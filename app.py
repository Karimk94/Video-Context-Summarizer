import os
import uuid
import time
import base64
import cv2
import requests
import logging
import concurrent.futures
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import moviepy.editor as mp
from dotenv import load_dotenv
import uvicorn
import shutil

load_dotenv()

app = FastAPI(title="Video Context Summarizer")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration
OCR_API_URL = os.getenv('OCR_API_URL')
GOVAI_API_URL = os.getenv('GOVAI_API_URL')
GOVAI_API_KEY = os.getenv('GOVAI_API_KEY')
GOVAI_VISION_MODEL = os.getenv('GOVAI_VISION_MODEL', 'qwen3-vl-30b')
GOVAI_AUDIO_MODEL = os.getenv('GOVAI_AUDIO_MODEL', 'whisper_asr')

tasks = {}

# Background cleanup
def cleanup_old_tasks():
    current_time = time.time()
    keys_to_delete = [
        task_id for task_id, task_data in tasks.items()
        if current_time - task_data.get('created_at', current_time) > 3600
    ]
    for task_id in keys_to_delete:
        del tasks[task_id]

def extract_keyframes(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0 or fps == 0:
        return []
        
    duration = total_frames / fps
    interval = max(1.0, duration / max_frames)
    
    for i in range(max_frames):
        target_time = i * interval
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
        ret, frame = cap.read()
        if ret:
            # Encode frame to JPEG base64
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(b64)
            
    cap.release()
    return frames

def analyze_video_for_visuals(video_path: str, task_id: str):
    tasks[task_id]['status'] = 'Analyzing video frames with GovAI...'
    try:
        keyframes = extract_keyframes(video_path, max_frames=5)
        if not keyframes:
            return "No frames extracted."
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze these keyframes from a video. Describe the objects, faces, and any context you see."}
                ]
            }
        ]
        
        for b64 in keyframes:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            
        payload = {
            "model": GOVAI_VISION_MODEL,
            "messages": messages,
            "max_tokens": 1000
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GOVAI_API_KEY}"
        }
        
        response = requests.post(GOVAI_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result_json = response.json()
        return result_json['choices'][0]['message']['content']
        
    except Exception as e:
        logging.error(f"Visual analysis failed: {e}")
        return f"Error: {str(e)}"

def transcribe_full_audio(video_path: str, task_id: str):
    tasks[task_id]['status'] = 'Extracting audio and transcribing...'
    audio_path = f"{video_path}.mp3"
    try:
        # Extract audio
        video = mp.VideoFileClip(video_path)
        if video.audio is None:
            return "No audio track found in video."
            
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        
        # Send to GovAI API
        headers = {
            "Authorization": f"Bearer {GOVAI_API_KEY}"
        }
        
        # We try to send it as a multipart form data request which is standard for whisper APIs.
        with open(audio_path, 'rb') as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
            data = {"model": GOVAI_AUDIO_MODEL}
            response = requests.post(GOVAI_API_URL, headers=headers, data=data, files=files, timeout=300)
            
        if response.status_code == 200:
            result_json = response.json()
            # If standard OpenAI format, transcript is in 'text'
            return result_json.get('text', str(result_json))
        else:
            return f"Audio API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        logging.error(f"Audio transcription failed: {e}")
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

def process_video_task(video_path: str, task_id: str):
    try:
        tasks[task_id]['status'] = 'Processing with GovAI models...'
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            visual_future = executor.submit(analyze_video_for_visuals, video_path, task_id)
            audio_future = executor.submit(transcribe_full_audio, video_path, task_id)
            
            visual_result = visual_future.result()
            audio_result = audio_future.result()
            
        tasks[task_id]['status'] = 'complete'
        tasks[task_id]['result'] = {
            "visual_analysis": visual_result,
            "audio_transcript": audio_result
        }
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass

@app.get("/", response_class=HTMLResponse)
async def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.post("/upload_stream")
async def upload_stream(request: Request, background_tasks: BackgroundTasks, language: str = 'english'):
    cleanup_old_tasks()
    
    filename_header = request.headers.get('X-Filename', f"{uuid.uuid4()}.mp4")
    filename = f"{uuid.uuid4()}_{filename_header}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(video_path, 'wb') as f:
            async for chunk in request.stream():
                f.write(chunk)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error receiving stream: {e}"})
        
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'queued', 'result': None, 'error': None, 'created_at': time.time()}
    
    background_tasks.add_task(process_video_task, video_path, task_id)
    return {"task_id": task_id}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...), language: str = Form('english')):
    cleanup_old_tasks()
    
    filename = f"{uuid.uuid4()}_{video.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error saving file: {e}"})
        
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'queued', 'result': None, 'error': None, 'created_at': time.time()}
    
    background_tasks.add_task(process_video_task, video_path, task_id)
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
        
    return {
        "status": task['status'],
        "result": task['result'],
        "error": task['error']
    }

if __name__ == '__main__':
    port = int(os.environ.get("HTTP_PLATFORM_PORT", 5008))
    print(f"Starting API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)