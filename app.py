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

# --- Basic Setup ---
app = Flask(__name__)
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Use a temporary directory for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory "database" to track task status and results
tasks = {}

# --- AI Model Loading (Load once on startup) ---
print("Loading AI models, please wait...")
YOLO_MODEL = YOLO('yolov8n.pt')
WHISPER_MODEL = whisper.load_model('base')
print("AI models loaded successfully.")

# --- Core Processing Logic (Adapted from desktop app) ---

def extract_keyframes(video_path: str, task_id: str):
    """Extracts keyframes and updates task status."""
    tasks[task_id]['status'] = 'Extracting keyframes...'
    keyframes = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video file.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, prev_frame = cap.read()
        if prev_frame is None: return []
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % 30 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)
                if np.count_nonzero(diff) > 100000:
                    keyframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                prev_gray = gray
        cap.release()
        return keyframes
    except Exception as e:
        tasks[task_id]['error'] = f"Keyframe extraction failed: {e}"
        return []

def analyze_frames_for_context(frames: list, task_id: str) -> (set, list):
    """Analyzes frames for objects and extracts unique faces."""
    tasks[task_id]['status'] = 'Analyzing keyframes for objects and faces...'
    detected_objects, known_face_embeddings, unique_faces = set(), [], []
    
    for i, frame in enumerate(frames):
        tasks[task_id]['status'] = f'Analyzing keyframe {i+1}/{len(frames)} for objects...'
        # Object Detection
        results = YOLO_MODEL(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                detected_objects.add(YOLO_MODEL.names[int(box.cls)])

    tasks[task_id]['status'] = 'Detecting and extracting unique faces...'
    for i, frame in enumerate(frames):
        try:
            # Use extract_faces to get face images and embeddings
            embedding_objs = DeepFace.represent(frame, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
            
            for emb_obj in embedding_objs:
                if 'embedding' not in emb_obj: continue
                embedding = emb_obj['embedding']
                
                # Check if the face is unique
                is_new = all(np.linalg.norm(np.array(known) - np.array(embedding)) > 0.6 for known in known_face_embeddings)
                
                if is_new:
                    known_face_embeddings.append(embedding)
                    
                    # Get the face image
                    facial_area = emb_obj['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    face_image = frame[y:y+h, x:x+w]
                    
                    # Convert to base64
                    pil_img = Image.fromarray(face_image)
                    buff = io.BytesIO()
                    pil_img.save(buff, format="JPEG")
                    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
                    
                    unique_faces.append(img_str)

        except Exception:
            pass # Skip frame if face analysis fails
            
    return detected_objects, unique_faces


def transcribe_audio_chunks(video_path: str, language: str, task_id: str) -> list:
    """Transcribes audio chunks and updates task status with better logging."""
    tasks[task_id]['status'] = 'Sampling audio for transcription...'
    transcribed_snippets = []
    try:
        with mp.VideoFileClip(video_path) as video:
            duration = video.duration
            sample_interval, chunk_duration = 180, 15
            num_chunks = int(duration // sample_interval) + 1
            for i, start_time in enumerate(range(0, int(duration), sample_interval)):
                tasks[task_id]['status'] = f'Transcribing audio chunk {i+1}/{num_chunks}...'
                end_time = min(start_time + chunk_duration, duration)
                temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_chunk_{i}.mp3")
                
                video.subclip(start_time, end_time).audio.write_audiofile(temp_audio_path, codec='mp3', logger=None)
                
                # Transcribe the chunk
                result = WHISPER_MODEL.transcribe(temp_audio_path, language=language)
                snippet = result['text'].strip()
                
                # --- NEW: Add console logging for debugging ---
                if snippet:
                    print(f"Task {task_id}: Found speech in chunk {i+1}: '{snippet[:50]}...'")
                    transcribed_snippets.append(f"[{int(start_time//60)}m {int(start_time%60)}s] {snippet}")
                else:
                    print(f"Task {task_id}: No speech detected in chunk {i+1}.")
                # --- END NEW ---
                
                os.remove(temp_audio_path)
        
        # --- NEW: Update final status if no speech was found at all ---
        if not transcribed_snippets:
             tasks[task_id]['status'] = 'Processing complete. No speech was detected in any audio samples.'
        # --- END NEW ---
             
        return transcribed_snippets
    except Exception as e:
        error_message = f"Audio transcription failed: {e}"
        print(f"ERROR for task {task_id}: {error_message}") # Print error to console
        tasks[task_id]['error'] = error_message
        return []

def process_video_task(video_path: str, language: str, task_id: str):
    """The main background task for processing a video."""
    try:
        # Step 1: Visual Analysis
        keyframes = extract_keyframes(video_path, task_id)
        detected_objects, unique_faces = analyze_frames_for_context(keyframes, task_id) if keyframes else (set(), [])
        
        # Step 2: Audio Analysis
        transcribed_snippets = transcribe_audio_chunks(video_path, language, task_id)

        # Step 3: Finalize
        tasks[task_id]['status'] = 'complete'
        tasks[task_id]['result'] = {
            "objects": sorted(list(detected_objects)),
            "faces": unique_faces,
            "snippets": transcribed_snippets
        }
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        # Clean up the uploaded video file
        if os.path.exists(video_path):
            os.remove(video_path)

# --- Flask API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    with open('index.html', 'r', encoding='utf-8') as f:
        return render_template_string(f.read())

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video upload and starts the background processing task."""
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
    """Allows the frontend to poll for the status of a task."""
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
    app.run(host='0.0.0.0', port=5008, debug=True)