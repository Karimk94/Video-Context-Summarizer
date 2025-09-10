import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from threading import Thread
import cv2
import whisper
import moviepy.editor as mp
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import logging
import base64
import io
from PIL import Image
from flask_cors import CORS
import requests
import re
from typing import List, Tuple
from werkzeug.serving import run_simple
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# --- Basic Setup ---
app = Flask(__name__)
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Set logging level to DEBUG to see detailed frame analysis logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


CORS(app, resources={r"/*": {"origins": "*"}})

# Use a temporary directory for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory "database" to track task status and results
tasks = {}

# --- AI Model Loading (Load once on startup) ---
logging.info("Loading AI models, please wait...")
YOLO_MODEL = YOLO('yolov8n.pt')
WHISPER_MODEL = whisper.load_model('base')
logging.info("AI models loaded successfully.")

# --- OCR Configuration ---
OCR_API_URL = os.getenv('OCR_API_URL')

# --- Smart Frame Capture Settings ---
OCR_FRAME_DIFFERENCE_THRESHOLD = 2000000 
MIN_OCR_INTERVAL_MS = 3000
MAX_OCR_INTERVAL_MS = 8000 
KEYFRAME_DIFFERENCE_THRESHOLD = 100000

# --- Core Processing Logic ---

def clean_ocr_text(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?()-:%]', '', text)
    return ' '.join(cleaned.split())

def analyze_video_for_visuals(video_path: str, task_id: str) -> Tuple[List, List, List]:
    tasks[task_id]['status'] = 'Analyzing video for objects, faces, and text...'
    
    detected_objects = set()
    known_face_embeddings = []
    unique_faces = []
    detected_texts = set()
    
    can_perform_ocr = OCR_API_URL is not None
    logging.info(f"Task {task_id}: OCR Service URL is set to '{OCR_API_URL}'. OCR enabled: {can_perform_ocr}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Task {task_id}: Could not open video for visual analysis.")
        return [], [], []

    ocr_prev_gray = None
    keyframe_prev_gray = None
    last_ocr_time = -MIN_OCR_INTERVAL_MS
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if frame_count % 15 != 0:
            continue

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if ocr_prev_gray is None:
            ocr_prev_gray = current_gray
            keyframe_prev_gray = current_gray

        time_since_last_ocr = current_time_ms - last_ocr_time
        should_process_for_ocr = False
        if can_perform_ocr and time_since_last_ocr >= MIN_OCR_INTERVAL_MS:
            h, w = current_gray.shape
            ch, cw = 480, int(480 * w / h)
            resized_ocr_prev = cv2.resize(ocr_prev_gray, (cw, ch))
            resized_current = cv2.resize(current_gray, (cw, ch))
            
            diff = cv2.absdiff(resized_ocr_prev, resized_current)
            non_zero_count = np.count_nonzero(diff)

            is_major_change = non_zero_count > OCR_FRAME_DIFFERENCE_THRESHOLD
            is_time_for_forced_check = time_since_last_ocr > MAX_OCR_INTERVAL_MS
            
            logging.debug(f"Task {task_id} OCR Check at {current_time_ms:.0f}ms: "
                          f"Time since last={time_since_last_ocr:.0f}ms (Min={MIN_OCR_INTERVAL_MS}, Max={MAX_OCR_INTERVAL_MS}). "
                          f"Frame diff={non_zero_count} (Threshold={OCR_FRAME_DIFFERENCE_THRESHOLD}). "
                          f"Is major change={is_major_change}. "
                          f"Is time for forced check={is_time_for_forced_check}.")

            if is_major_change or is_time_for_forced_check:
                should_process_for_ocr = True

        if should_process_for_ocr:
            last_ocr_time = current_time_ms
            ocr_prev_gray = current_gray
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                
                final_ocr_url = f"{OCR_API_URL.rstrip('/')}/translate_image_stream"
                logging.info(f"Task {task_id}: TRIGGER! Sending frame to OCR service at {final_ocr_url}")
                
                headers = {'Content-Type': 'application/octet-stream'}
                response = requests.post(final_ocr_url, data=buffer.tobytes(), headers=headers, timeout=15)

                if response.status_code == 200:
                    raw_ocr_result = response.json().get('text', '').strip()
                    cleaned_ocr_result = clean_ocr_text(raw_ocr_result)
                    
                    if cleaned_ocr_result and len(cleaned_ocr_result) > 3:
                        letters = sum(c.isalpha() for c in cleaned_ocr_result)
                        digits = sum(c.isdigit() for c in cleaned_ocr_result)
                        has_multiple_words = ' ' in cleaned_ocr_result.strip()

                        if has_multiple_words and letters > digits:
                            logging.info(f"Task {task_id}: OCR raw: '{raw_ocr_result}' -> Cleaned & Validated: '{cleaned_ocr_result}'")
                            detected_texts.add(cleaned_ocr_result)
                        else:
                            logging.warning(f"Task {task_id}: OCR text '{cleaned_ocr_result}' discarded for being non-prose.")
                else:
                    logging.error(f"Task {task_id}: OCR service returned error status {response.status_code}: {response.text}")

            except Exception as e:
                logging.error(f"Task {task_id}: OCR request failed: {e}")

        diff = cv2.absdiff(keyframe_prev_gray, current_gray)
        if np.count_nonzero(diff) > KEYFRAME_DIFFERENCE_THRESHOLD:
            keyframe_prev_gray = current_gray
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = YOLO_MODEL(rgb_frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    detected_objects.add(YOLO_MODEL.names[int(box.cls)])
            
            try:
                embedding_objs = DeepFace.represent(rgb_frame, model_name='VGG-Face', detector_backend='opencv', enforce_detection=True)
                for emb_obj in embedding_objs:
                    if 'embedding' not in emb_obj: continue
                    embedding = emb_obj['embedding']
                    is_new = all(np.linalg.norm(np.array(known) - np.array(embedding)) > 0.6 for known in known_face_embeddings)
                    if is_new:
                        known_face_embeddings.append(embedding)
                        facial_area = emb_obj['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        face_image = rgb_frame[y:y+h, x:x+w]
                        pil_img = Image.fromarray(face_image)
                        buff = io.BytesIO()
                        pil_img.save(buff, format="JPEG")
                        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
                        unique_faces.append(img_str)
            except Exception:
                pass

    cap.release()
    logging.info(f"Task {task_id}: Visual analysis finished. Found {len(detected_objects)} objects, {len(unique_faces)} faces, {len(detected_texts)} text snippets.")
    return sorted(list(detected_objects)), unique_faces, sorted(list(detected_texts))


def transcribe_full_audio(video_path: str, language: str, task_id: str) -> str:
    tasks[task_id]['status'] = 'Extracting full audio for transcription...'
    try:
        with mp.VideoFileClip(video_path) as video:
            temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_full_audio.mp3")
            video.audio.write_audiofile(temp_audio_path, codec='mp3', logger=None)

        tasks[task_id]['status'] = 'Transcribing full audio (this may take a while)...'
        
        result = WHISPER_MODEL.transcribe(temp_audio_path, language=language)
        full_transcript = result['text'].strip()

        os.remove(temp_audio_path)
        
        logging.info(f"Task {task_id}: Transcription complete.")
        return full_transcript

    except Exception as e:
        error_message = f"Full audio transcription failed: {e}"
        logging.error(f"ERROR for task {task_id}: {error_message}")
        tasks[task_id]['error'] = error_message
        return ""


def process_video_task(video_path: str, language: str, task_id: str):
    try:
        detected_objects, unique_faces, ocr_texts = analyze_video_for_visuals(video_path, task_id)
        full_transcript = transcribe_full_audio(video_path, language, task_id)

        tasks[task_id]['status'] = 'complete'
        tasks[task_id]['result'] = {
            "objects": detected_objects,
            "faces": unique_faces,
            "transcript": full_transcript,
            "ocr_texts": ocr_texts
        }
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        logging.error(f"Task {task_id}: Top-level processing error.", exc_info=True)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            
# --- Flask API Endpoints ---

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return render_template_string(f.read())

@app.route('/upload_stream', methods=['POST'])
def upload_stream():
    language = request.args.get('language', 'english')
    filename_header = request.headers.get('X-Filename', f"{uuid.uuid4()}.mp4")
    
    filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(filename_header))[1]
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        with open(video_path, 'wb') as f:
            chunk_size = 262144 
            while True:
                chunk = request.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        logging.error(f"Error receiving stream for file {filename}", exc_info=True)
        return jsonify(error=f"Error receiving stream: {e}"), 500

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'queued', 'result': None, 'error': None}
    
    thread = Thread(target=process_video_task, args=(video_path, language, task_id))
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    language = request.form.get('language', 'english')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        task_id = str(uuid.uuid4())
        tasks[task_id] = {'status': 'queued', 'result': None, 'error': None}
        
        thread = Thread(target=process_video_task, args=(video_path, language, task_id))
        thread.start()
        
        return jsonify({"task_id": task_id})

@app.route('/status/<task_id>')
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    response = {
        "status": task['status'],
        "result": task['result'],
        "error": task['error']
    }
    return jsonify(response)

if __name__ == '__main__':
    run_simple(
        '0.0.0.0',
        5008,
        app,
        use_reloader=False,
        use_debugger=True,
        threaded=True,
        exclude_patterns=['*venv*', '*__pycache__*']
    )