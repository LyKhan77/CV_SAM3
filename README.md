# AI CV Monitoring Dashboard

This project is a prototype for a real-time AI Computer Vision monitoring system. It leverages **Meta's SAM 3 (Segment Anything Model 3)** to detect, segment, and count objects in video streams or static images based on user prompts.

## 🔄 System Workflow

The application operates on a Client-Server architecture designed for real-time performance:

1.  **Input Acquisition:**
    *   The backend (`main.py`) captures video frames using **OpenCV**.
    *   Sources can be a live **RTSP Stream** (IP Camera), an uploaded **Video File**, or a static **Image**.

2.  **User Interaction (Frontend):**
    *   The user interacts with the web interface (`index.html`).
    *   **Configuration:** Users set input modes, model parameters (Confidence, Threshold), and prompts.
    *   **Prompts:** Users provide a text description (e.g., "person with red hat") or click on the video feed (Point Prompt) to identify objects.

3.  **AI Processing (Backend):**
    *   The backend receives state updates via REST API endpoints.
    *   The `video_processing_loop` (in `model.py`) continuously runs in the background.
    *   **Inference:** Frames are passed to the **SAM 3 model** (`transformers` library).
    *   **Post-Processing:** Raw model outputs are filtered by confidence, subjected to Non-Maximum Suppression (NMS), and smoothed for display.

4.  **Real-time Streaming:**
    *   Processed frames (annotated with segmentation masks) are encoded as Base64 JPEGs.
    *   Analytics data (counts, status, detection info) is packaged into a JSON payload.
    *   Data is broadcasted to the frontend via **WebSocket** (`ws://.../ws/monitor`).

5.  **Visualization:**
    *   The frontend receives the WebSocket messages and updates the DOM in real-time (Video feed, Object counters, Status badges).

## 🌟 Key Features

*   **Multi-Modal Input Support:**
    *   **RTSP Streaming:** Connect to live surveillance cameras.
    *   **Video Upload:** Process local video files with seek and playback controls.
    *   **Image Analysis:** High-resolution segmentation of static images.
*   **Flexible AI Prompting:**
    *   **Text Prompts:** Natural language object detection (e.g., "safety helmet").
    *   **Point Prompts:** "Click-to-segment" functionality for precise object selection.
*   **Real-time Dashboard:**
    *   Live video feed with color-coded segmentation overlays.
    *   Dynamic object counting with user-defined limits.
    *   System status indicators (Ready, Processing, Approved).
*   **Export Capabilities:**
    *   **Snapshot Mode:** Save detected objects as individual transparent PNGs with bounding box metadata.

## 🚀 Installation & Usage

### Prerequisites
*   **Python 3.10+**
*   **CUDA-capable GPU** (Recommended for SAM 3 performance)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
**Crucial:** Install PyTorch with CUDA support first.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*Note: Adjust `cu121` (CUDA 12.1) based on your system configuration.*

Then install the remaining requirements:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

### 5. Access the Dashboard
Open your web browser and navigate to:
`http://127.0.0.1:8000`

## 📂 Project Structure

*   `main.py`: FastAPI backend server and WebSocket manager.
*   `model.py`: AI model loading, inference loop, and image processing logic.
*   `web_app/`: Frontend assets (HTML, CSS, JS).
*   `requirements.txt`: Project dependencies.
