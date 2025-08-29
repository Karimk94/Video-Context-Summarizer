Offline Video Context Summarizer (Web Version)
A web-based application that uses a Python backend and an HTML/JS frontend to generate a context summary for video files. It uses Strategic Sampling to quickly analyze key visual frames and audio chunks.

This application runs as a local server on your machine. You interact with it through your web browser.

Architecture
Backend (app.py): A Python server built with Flask. It handles video uploads and all the heavy processing (object detection, face detection, transcription) offline.

Frontend (index.html): A modern user interface built with HTML, Tailwind CSS, and JavaScript. You open this file in your browser to interact with the application.

Features
Web Interface: Modern, browser-based UI for uploading and viewing results.

Easy Installation: No C++ compilers needed.

Efficient & Fast: Analyzes videos using the strategic sampling method.

Fully Offline (after setup): The backend processing requires no internet.

Secure: Your video files are processed locally and are not sent over the internet.

Prerequisites
Python 3.8+

ffmpeg: Required for audio extraction.

Windows: Download from ffmpeg.org and add its bin directory to your system's PATH.

macOS (via Homebrew): brew install ffmpeg

Linux (via apt): sudo apt update && sudo apt install ffmpeg

How to Run the Application
Step 1: Install Dependencies (Windows)
On Windows, simply double-click the install.bat script. It will automatically:

Create a Python virtual environment in a folder named venv.

Activate the environment.

Install all the required Python packages from requirements.txt.

For macOS/Linux, run these commands in your terminal:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Important: The first time you run the backend, it will download the necessary AI models (Whisper, YOLO, DeepFace). This requires an internet connection. After this one-time download, the server will be fully offline.

Step 2: Run the Backend Server (Windows)
Double-click the run.bat script. This will start the local web server.

Keep the terminal window that appears open. This is your server. If you close it, the application will stop working.

For macOS/Linux, run this command in your terminal:

python app.py

Step 3: Use the Frontend
Open your web browser (like Chrome, Firefox, or Edge).

Go to the address shown in the terminal. It will be:
http://127.0.0.1:5000

The web interface will load, and you can now select a video file, choose the language, and generate a summary