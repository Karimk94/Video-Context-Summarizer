# Video Context Summarizer

> **High-Performance Video Intelligence Microservice for the RTA Smart EDMS Ecosystem**  
> Powered by **FastAPI**, **GovAI LLM / Vision-Language Models**, and **Asynchronous Media Processing**.

---

## 1. Overview & Architecture

The **Video Context Summarizer** is an automated video intelligence microservice designed to ingest, process, analyze, and extract rich contextual metadata from video files within the **RTA Smart EDMS** (Electronic Document Management System) environment.

Originally prototyped with heavy local ML models (Flask, local YOLO, local DeepFace, and local Whisper), the service has been refactored into a high-concurrency **FastAPI** ASGI microservice. Heavy compute and inference are offloaded to **GovAI Multimodal LLM & ASR APIs**, while video pre-processing, temporal strategic sampling, and audio stream extraction remain local, lightweight, and efficient.

```
                              ┌───────────────────────────────────────────────┐
                              │            Smart EDMS Ecosystem               │
                              │             (..\EDMS API)                     │
                              └───────┬───────────────────────────────▲───────┘
                                      │                               │
                       Chunked Stream │ POST /upload_stream           │ Task Result
                           (Port 5008)│                               │ Polling /status/{id}
                                      ▼                               │
┌─────────────────────────────────────────────────────────────────────┴───────────────────────────────────┐
│                               Video Context Summarizer (FastAPI)                                        │
│                                                                                                         │
│  ┌───────────────────────┐             ┌─────────────────────────────────────────────────────────────┐  │
│  │   Stream / Multipart  │             │               Background Processing Pipeline                │  │
│  │    Ingestion Router   │ ──────────► │  (concurrent.futures.ThreadPoolExecutor - 2 Workers)        │  │
│  └───────────────────────┘             └──────────────┬──────────────────────────────┬───────────────┘  │
│                                                       │                              │                  │
│                                           Frame Set   │                  Audio (.mp3)│                  │
│                                                       ▼                              ▼                  │
│                                         ┌───────────────────────────┐  ┌─────────────────────────────┐  │
│                                         │      OpenCV Keyframes     │  │      MoviePy Audio Split    │  │
│                                         │    (Strategic Sampling)   │  │        (Track Extractor)    │  │
│                                         └─────────────┬─────────────┘  └─────────────┬───────────────┘  │
│                                                       │                              │                  │
└───────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────┘
                                                        │                              │
                                                        ▼                              ▼
                                     ┌─────────────────────────────────────────────────────────┐
                                     │                     GovAI Cloud / API                   │
                                     │                                                         │
                                     │  • Vision-Language (Qwen-VL-30B) Base64 Keyframes       │
                                     │  • Speech-to-Text ASR (Whisper ASR) Multipart MP3       │
                                     └─────────────────────────────────────────────────────────┘
```

### Key Highlights
- **High Concurrency & Async Architecture**: Built on FastAPI with asynchronous streaming, background tasks (`BackgroundTasks`), and thread pool workers (`ThreadPoolExecutor`).
- **Cloud-Native GovAI LLM Engine**: Offloads visual analysis to `qwen3-vl-30b` and audio transcription to `whisper_asr` through GovAI's secure endpoint (`https://llmapi.govai.ae`).
- **Memory-Efficient Large Video Streaming**: Accepts chunked octet streams directly from `..\EDMS API` via `POST /upload_stream`, bypassing in-memory buffer bloat.
- **Offline Self-Contained Deployment**: Includes automated offline packaging tools (`packages/` wheel cache, `download_packages.bat`, `install_packages.bat`) for secure air-gapped government environments.
- **Standalone Dashboard**: Embedded interactive Web UI (`index.html`) powered by bundled `static/tailwindcss.js` (zero CDN / external web dependencies required).

---

## 2. GovAI LLM Integration

The microservice integrates directly with the **GovAI LLM Platform** (`https://llmapi.govai.ae/chat/completions`) using standard OpenAI-compatible API specifications with Bearer Token authentication (`GOVAI_API_KEY`).

### A. Multimodal Visual Analysis (`analyze_video_for_visuals`)
- **Model**: `GOVAI_VISION_MODEL` (Default: `qwen3-vl-30b`).
- **Keyframe Extraction (`extract_keyframes`)**:
  - Leverages OpenCV (`cv2.VideoCapture`) to compute total frame count and FPS.
  - Dynamically calculates the temporal interval: `interval = max(1.0, duration / max_frames)` (defaults to 5 evenly distributed keyframes across the video duration).
  - Extracts each frame at target timestamps (`cv2.CAP_PROP_POS_MSEC`), encodes it into JPEG format, and converts it into a base64 string.
- **GovAI Request Payload**:
  Frames are submitted as multi-content message arrays formatted as data URLs:
  ```json
  {
    "model": "qwen3-vl-30b",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Please analyze these keyframes from a video. Describe the objects, faces, and any context you see." },
          { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,<frame_1_b64>" } },
          { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,<frame_2_b64>" } },
          { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,<frame_3_b64>" } }
        ]
      }
    ],
    "max_tokens": 1000
  }
  ```
- **Output**: Returns contextual visual description, identified subjects, actions, and key scene components.

### B. Audio Extraction & Transcription (`transcribe_full_audio`)
- **Model**: `GOVAI_AUDIO_MODEL` (Default: `whisper_asr`).
- **Audio Extraction**:
  - Uses MoviePy (`mp.VideoFileClip`) to separate the audio track from the uploaded video.
  - Generates a temporary local `.mp3` file on disk.
  - Handles silent videos gracefully (returns `"No audio track found in video"`).
- **GovAI Request Payload**:
  - Sends a multipart form-data request to GovAI:
    ```http
    POST https://llmapi.govai.ae/chat/completions
    Authorization: Bearer <GOVAI_API_KEY>
    Content-Type: multipart/form-data

    file=@<audio_file>.mp3;type=audio/mpeg
    model=whisper_asr
    ```
- **Output & Cleanup**: Parses the transcription text (`result.get('text')`) and reliably unlinks the temporary `.mp3` file in a `finally` block.

### C. Parallel Execution
Visual analysis and audio transcription run in parallel via `concurrent.futures.ThreadPoolExecutor(max_workers=2)`:
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    visual_future = executor.submit(analyze_video_for_visuals, video_path, task_id)
    audio_future = executor.submit(transcribe_full_audio, video_path, task_id)
    
    visual_result = visual_future.result()
    audio_result = audio_future.result()
```
This reduces overall processing latency by ~50% compared to sequential processing.

---

## 3. Integration with `..\EDMS API`

The Video Context Summarizer functions as an asynchronous media analysis engine for the primary **EDMS API** located in the sibling directory (`..\EDMS API`).

### A. Environment Configuration in EDMS API
In `..\EDMS API\.env` and `..\EDMS API\web2.config`:
```ini
VIDEO_SUMMARIZER_API_URL="http://localhost:5008/"
```

### B. Ingestion & Streaming Flow (`api_client.py` & `services/processor.py`)

1. **Document Detection**:
   When the background document worker in `..\EDMS API\services\processor.py` (`process_document`) receives a document from the Oracle queue with `media_type == 'video'`, it invokes `async_summarize_video()`.

2. **Chunked Streaming (`api_client.summarize_video`)**:
   Instead of buffering large videos in memory, `EDMS API` streams the video bytes in 256 KB chunks directly to the summarizer endpoint:
   ```python
   # ..\EDMS API\api_client.py
   def summarize_video(video_data, filename):
       api_url = os.getenv("VIDEO_SUMMARIZER_API_URL")
       upload_endpoint = f"{api_url}/upload_stream"
       
       headers = {'Content-Type': 'application/octet-stream', 'X-Filename': filename}
       response = requests.post(upload_endpoint, data=data_chunk_generator(video_data), headers=headers, timeout=300)
       task_id = response.json().get('task_id')
       ...
   ```

3. **Asynchronous Polling**:
   `EDMS API` polls the task endpoint `GET /status/{task_id}` every 5 seconds (up to 360 polls / 30 minutes):
   ```python
   status_endpoint = f"{api_url}/status/{task_id}"
   for _ in range(360):
       time.sleep(5)
       status_data = requests.get(status_endpoint).json()
       if status_data['status'] == 'complete':
           return status_data['result']
   ```

4. **Downstream Ingestion & Metadata Tagging**:
   Once `Video Context Summarizer` marks the task as `complete`, `EDMS API` parses the returned JSON:
   - **Visual Objects / Descriptions**: Passed to `_translate_and_collect` to generate English/Arabic tag pairs and written to the Oracle DMS database via `db_connector.insert_keywords_and_tags`.
   - **Detected Faces**: Cropped faces are forwarded to `Face Recognition py` (`http://localhost:5002`) to identify known personnel and populate the `VIPs:` section in the document abstract.
   - **Speech Transcript**: Forwarded to `Translator Rephraser py` (`http://localhost:5005`) for keyword tokenization, bilingual translation, and inclusion in the document caption.
   - **File Lifecycle**: Upon task completion or failure, the temporary video file stored in `uploads/` is immediately deleted. Old task records in memory are automatically purged after 1 hour (`cleanup_old_tasks`).

---

## 4. Ecosystem Port Reference

The Video Context Summarizer is part of the coordinated microservices architecture powering Smart EDMS:

| Service Name | Port | Description |
| :--- | :--- | :--- |
| **Video Context Summarizer** | **5008** | **Video frame analysis, speech transcription, context extraction (This service)** |
| **EDMS API** | **5000** | Core backend orchestrator, Oracle DB manager & WSDL interface |
| **Captioning Service** | **5001** | Image captioning service |
| **Face Recognition Service** | **5002** | Face detection, identification, and enrollment |
| **Embedding Service** | **5003** | Vector embedding generation for semantic search |
| **OCR Service** | **5004** | Optical Character Recognition for PDFs and scanned documents |
| **Translator & Rephraser** | **5005** | Bilingual translation and transcript tokenization |
| **Smart Search Service** | **5006** | Semantic vector search backend |
| **Smart EDMS Frontend** | **3000** | Next.js user web portal |

---

## 5. Architecture & Code Graph (Graphify Report)

Based on the static AST and dependency graph in `graphify-out`:

- **Extraction Metrics**: 616 nodes, 1,675 edges, across 38 community clusters.
- **Core Architecture Nodes**:
  - `FastAPI`, `BackgroundTasks`, `UploadFile`, `Request` — Asynchronous HTTP server and streaming endpoints.
  - `analyze_video_for_visuals` — Multimodal vision extraction handler.
  - `transcribe_full_audio` — ASR speech transcription handler.
  - `extract_keyframes` — Temporal keyframe sampler using OpenCV.
  - `cleanup_old_tasks` — In-memory task expiration manager.
  - `upload_stream` / `upload_video` — Ingestion routes for `..\EDMS API` and web clients.
- **Import Cycles**: **0 detected**. The codebase maintains strict decoupling between endpoints, media extraction, and AI client communication.

---

## 6. REST API Reference

### Base URL: `http://localhost:5008`

#### `GET /`
Serves the web dashboard (`index.html`).
- **Response**: `text/html; charset=utf-8`

---

#### `POST /upload_stream`
Used primarily by `..\EDMS API` for streaming large video files without buffering in RAM.
- **Headers**:
  - `Content-Type: application/octet-stream`
  - `X-Filename`: Target filename (e.g. `recording.mp4`).
- **Query Parameters**:
  - `language` (optional, default: `english`).
- **Response** (`200 OK`):
  ```json
  {
    "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
  }
  ```

---

#### `POST /upload`
Standard multipart form-data upload used by the web interface.
- **Request Body** (`multipart/form-data`):
  - `video`: Binary video file.
  - `language`: Target language string (e.g. `english` or `arabic`).
- **Response** (`200 OK`):
  ```json
  {
    "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
  }
  ```

---

#### `GET /status/{task_id}`
Checks the execution progress and fetches results.
- **Response** (`200 OK` - In Progress):
  ```json
  {
    "status": "Processing with GovAI models...",
    "result": null,
    "error": null
  }
  ```
- **Response** (`200 OK` - Completed):
  ```json
  {
    "status": "complete",
    "result": {
      "visual_analysis": "The video shows a conference room with three individuals seated around a table...",
      "audio_transcript": "Good morning everyone, welcome to the quarterly project status meeting..."
    },
    "error": null
  }
  ```
- **Response** (`404 Not Found`):
  ```json
  {
    "error": "Task not found"
  }
  ```

---

## 7. Interactive Web Interface (`index.html`)

The embedded dashboard provides an intuitive, interactive environment for testing and manual video ingestion:

1. **Drag-and-Drop Ingestion**:
   - Accepts all standard video formats (`.mp4`, `.avi`, `.mov`, `.mkv`, etc.).
   - Displays real-time file size and filename verification.
2. **Language Selection**:
   - Select between English and Arabic processing modes.
3. **Real-time Status Polling**:
   - Visual spinner and dynamic status messages mirroring backend task state.
4. **Summary & Keyword Extraction**:
   - Visually detected objects card.
   - Speech transcript display.
   - One-click transcript tokenization using `Translator Rephraser py` (`http://localhost:5005`).
5. **Interactive Face Recognition & Enrollment**:
   - Extracted face gallery.
   - Direct integration with `Face Recognition py` (`http://localhost:5002`) to identify faces and save new person enrollments (`/add_person`).
6. **100% Offline UI**:
   - Utilizes `static/tailwindcss.js` directly from the local filesystem with zero external CDN dependencies.

---

## 8. Offline & Air-Gapped Deployment Guide

The project is structured for self-contained, offline operation in isolated enterprise and government environments.

### Step 1: Package Download (On Internet-Connected Machine)
Run `download_packages.bat` to download Python wheel files (`.whl`) matching `requirements.txt`:
```bat
download_packages.bat
```
*This downloads 64-bit and 32-bit Windows binary wheels for Python 3.12 into the local `packages/` directory.*

### Step 2: Create Self-Contained Archive
Run `create_archive.bat` to bundle source files, `static/` assets, and offline wheels into a timestamped archive:
```bat
create_archive.bat
```
*Output: `..\Video_Context_Summarizer_YYYY-MM-DD_HH-mm-ss.zip`*

### Step 3: Install on Target Server (No Internet)
Transfer the zip to the production host, extract it, and execute:
```bat
install_packages.bat
```
This script automatically:
1. Creates a clean Python virtual environment (`venv`).
2. Creates the `logs/` directory.
3. Installs all packages exclusively from the local `packages/` directory using:
   ```cmd
   pip install --no-index --find-links=packages -r requirements.txt
   ```

### Step 4: Verify Environment
Run `check_env.bat` to test virtual environment integrity and FastAPI imports:
```bat
check_env.bat
```

### Step 5: Start the Service
Launch the Uvicorn server:
```bat
run.bat
```
*Starts Uvicorn on `0.0.0.0:5008` with 4 worker processes.*

---

## 9. Environment Variables Reference (`.env`)

Create or update the `.env` file in the project root:

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `HTTP_PLATFORM_PORT` | No | `5008` | Port on which the FastAPI / Uvicorn server listens |
| `GOVAI_API_URL` | Yes | `https://llmapi.govai.ae/chat/completions` | GovAI LLM chat completions endpoint |
| `GOVAI_API_KEY` | Yes | — | GovAI API Bearer authentication key |
| `GOVAI_VISION_MODEL` | No | `qwen3-vl-30b` | GovAI multimodal vision model name |
| `GOVAI_AUDIO_MODEL` | No | `whisper_asr` | GovAI speech recognition ASR model name |
| `OCR_API_URL` | No | `http://localhost:5004` | Optional endpoint for OCR text extraction |

---

## 10. File & Directory Structure

```
Video Context Summarizer/
│
├── .env                       # Local service environment configuration
├── app.py                     # FastAPI application & GovAI integration core
├── index.html                 # Modern Tailwind dark-mode web interface
├── requirements.txt           # Python dependency declarations
│
├── static/
│   └── tailwindcss.js         # Offline Tailwind CSS bundle (no internet needed)
│
├── packages/                  # Offline wheel cache (.whl files) for air-gapped install
├── uploads/                   # Temporary directory for video file processing
├── venv/                      # Python virtual environment
│
├── graphify-out/              # Static AST & code dependency graph analysis
│   ├── GRAPH_REPORT.md        # Code community and centrality report
│   ├── graph.json             # Raw node and edge graph data
│   └── graph.html             # Visual interactive graph viewer
│
├── archive_helper.ps1         # PowerShell staging & compression helper
├── check_env.bat              # Health-check script verifying virtual environment
├── create_archive.bat         # Automated deployment archive generator
├── download_packages.bat      # Pre-fetch wheels on connected workstation
├── install_packages.bat       # Offline pip installer for target server
└── run.bat                    # Production launch script (Uvicorn, 4 workers)
```

---

## 11. Maintenance & Troubleshooting

- **Task Timeouts**: If processing high-resolution or long videos takes longer than expected, verify that the `requests.post` timeout in `app.py` is set appropriately (currently 120s for visual analysis, 300s for audio transcription).
- **Temporary File Space**: The service immediately unlinks processed video files in `process_video_task` and temporary `.mp3` files in `transcribe_full_audio`. If disk space fills up unexpectedly, check the `uploads/` folder for interrupted tasks.
- **GovAI Connectivity**: If requests fail with `401 Unauthorized` or connection errors, verify `GOVAI_API_KEY` in `.env` and confirm the server has network routing to `https://llmapi.govai.ae`.
- **FFmpeg / Audio Extraction**: MoviePy utilizes imageio and FFmpeg. If audio extraction fails, ensure `ffmpeg` is available on the system PATH or embedded with the Python distribution.