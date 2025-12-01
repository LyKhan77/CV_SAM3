import asyncio
import uvicorn
import os
import shutil
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import logic from model.py
from model import load_model, video_processing_loop

# --- 1. Pydantic Models ---
class StreamRequest(BaseModel): url: str
class PromptRequest(BaseModel): object_name: str
class LimitRequest(BaseModel): value: int
class SoundRequest(BaseModel): enabled: bool
class ModelConfigRequest(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    display_mode: str

class InputModeRequest(BaseModel):
    mode: str  # "rtsp", "video", or "image"

class VideoSeekRequest(BaseModel):
    frame: int

class VideoPlaybackRequest(BaseModel):
    playing: bool

# --- 2. Application State & WebSocket Manager ---
app_state: Dict[str, Any] = {
    "rtsp_url": None,
    "uploaded_image_path": None,  # NEW field for local image support
    "prompt": None,
    "point_prompt": None,
    "max_limit": 100,
    "sound_enabled": False,
    "model": None,
    "processor": None,
    # New state for model configuration
    "confidence_threshold": 0.5,
    "mask_threshold": 0.5,
    "display_mode": "segmentation", # "segmentation" or "bounding_box"
    "select_object_mode": False,
    # New state for input mode management
    "input_mode": "rtsp",              # "rtsp", "video", or "image"
    "video_file_path": None,            # Path to uploaded video file
    "video_current_frame": 0,           # Current frame index
    "video_total_frames": None,         # Total frame count
    "video_fps": None,                  # Video frames per second
    "video_capture": None,              # VideoCapture instance for video files
    "video_playing": True,              # Video playback state
    "video_seek_request": None,         # Seek frame index
    "video_speed": 1.0,                 # Playback speed multiplier
}

class ConnectionManager:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket): self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections: await connection.send_text(message)

manager = ConnectionManager()

# --- 3. Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Application Startup ---")
    model, processor = load_model()
    app_state["model"] = model
    app_state["processor"] = processor
    
    print("Starting background video processing loop...")
    asyncio.create_task(video_processing_loop(manager, app_state))
    yield
    print("--- Application Shutdown ---")

# --- 4. FastAPI Application ---
app = FastAPI(title="AI CV Monitoring Dashboard Backend", lifespan=lifespan)

# --- 5. Mount Static Files ---
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# --- 6. API & Frontend Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("web_app/templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/config/stream")
async def set_stream_url(request: StreamRequest):
    app_state["rtsp_url"] = None 
    await asyncio.sleep(0.1)
    app_state["rtsp_url"] = request.url
    return {"status": "success", "message": f"Stream URL set to {request.url}"}

@app.post("/api/config/prompt")
async def set_prompt(request: PromptRequest):
    app_state["prompt"] = request.object_name
    app_state["point_prompt"] = None # Clear point prompt when text prompt is used
    return {"status": "success", "message": f"Prompt set to '{request.object_name}'"}

@app.post("/api/config/limit")
async def set_limit(request: LimitRequest):
    app_state["max_limit"] = request.value
    return {"status": "success", "message": f"Limit set to {request.value}"}

@app.post("/api/config/sound")
async def set_sound_toggle(request: SoundRequest):
    app_state["sound_enabled"] = request.enabled
    return {"status": "success", "message": f"Sound notification set to {request.enabled}"}

# New endpoint for model configuration
@app.post("/api/config/model")
async def set_model_config(request: ModelConfigRequest):
    app_state["confidence_threshold"] = request.confidence
    app_state["mask_threshold"] = request.mask_threshold
    app_state["display_mode"] = request.display_mode
    print(f"Model config updated: Confidence={request.confidence}, Mask={request.mask_threshold}, Display={request.display_mode}")
    return {"status": "success", "message": "Model config updated"}

@app.post("/api/config/input-mode")
async def set_input_mode(request: InputModeRequest):
    """
    Switch input mode and handle state conflicts.
    """
    mode = request.mode
    if mode not in ["rtsp", "video", "image"]:
        return {"status": "error", "message": "Invalid input mode"}

    # Clear conflicting states
    if mode == "rtsp":
        app_state["uploaded_image_path"] = None
        app_state["video_file_path"] = None
        # Release video capture if exists
        if app_state.get("video_capture"):
            app_state["video_capture"].release()
            app_state["video_capture"] = None

    elif mode == "video":
        app_state["rtsp_url"] = None
        app_state["uploaded_image_path"] = None

    elif mode == "image":
        app_state["rtsp_url"] = None
        app_state["video_file_path"] = None
        # Release video capture if exists
        if app_state.get("video_capture"):
            app_state["video_capture"].release()
            app_state["video_capture"] = None

    app_state["input_mode"] = mode
    app_state["prompt"] = None  # Clear existing prompts
    app_state["point_prompt"] = None

    print(f"Input mode switched to: {mode}")
    return {"status": "success", "message": f"Input mode set to {mode}"}

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for local processing.
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        return {"status": "error", "message": "File must be a video"}

    # Create uploads directory if not exists
    os.makedirs("uploads", exist_ok=True)

    # Save file
    file_path = f"uploads/{file.filename}"
    try:
        # Read and save video file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Test video file and get metadata
        video_capture = cv2.VideoCapture(file_path)
        if not video_capture.isOpened():
            return {"status": "error", "message": "Failed to open video file"}

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Release test capture
        video_capture.release()

        if total_frames <= 0:
            return {"status": "error", "message": "Invalid video file - no frames found"}

        # Update app state
        app_state["video_file_path"] = file_path
        app_state["video_current_frame"] = 0
        app_state["video_total_frames"] = total_frames
        app_state["video_fps"] = fps
        app_state["video_playing"] = True
        app_state["video_seek_request"] = None
        app_state["input_mode"] = "video"

        # Clear other input modes
        app_state["rtsp_url"] = None
        app_state["uploaded_image_path"] = None
        app_state["prompt"] = None
        app_state["point_prompt"] = None

        print(f"Video uploaded successfully: {file.filename}")
        print(f"Video metadata: {total_frames} frames, {fps:.2f} FPS")
        return {
            "status": "success",
            "message": f"Video uploaded: {file.filename}",
            "total_frames": total_frames,
            "fps": fps
        }

    except Exception as e:
        print(f"Error uploading video: {e}")
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

@app.post("/api/config/video/seek")
async def seek_video(request: VideoSeekRequest):
    """
    Seek to a specific frame in the video.
    """
    if not app_state.get("video_file_path"):
        return {"status": "error", "message": "No video file loaded"}

    frame_index = max(0, min(request.frame, app_state["video_total_frames"] - 1))
    app_state["video_seek_request"] = frame_index
    app_state["video_current_frame"] = frame_index

    return {"status": "success", "message": f"Seeked to frame {frame_index}"}

@app.post("/api/config/video/play-pause")
async def toggle_video_playback(request: VideoPlaybackRequest):
    """
    Toggle video playback state.
    """
    if not app_state.get("video_file_path"):
        return {"status": "error", "message": "No video file loaded"}

    app_state["video_playing"] = request.playing
    action = "playing" if request.playing else "paused"
    return {"status": "success", "message": f"Video {action}"}

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for local processing.
    Disables RTSP streaming when an image is uploaded.
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        return {"status": "error", "message": "File must be an image"}

    # Create uploads directory if not exists
    os.makedirs("uploads", exist_ok=True)

    # Save file
    file_path = f"uploads/{file.filename}"
    try:
        # Read file into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             return {"status": "error", "message": "Failed to decode image"}

        # Resize if too large (Max 1024px for better SAM performance)
        max_dim = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        # Save processed image
        cv2.imwrite(file_path, img)

        # Update app state
        app_state["uploaded_image_path"] = file_path
        app_state["input_mode"] = "image"
        app_state["rtsp_url"] = None  # Disable RTSP when using local image
        app_state["video_file_path"] = None  # Disable video when using image
        # Release video capture if exists
        if app_state.get("video_capture"):
            app_state["video_capture"].release()
            app_state["video_capture"] = None
        app_state["prompt"] = None  # Clear existing prompts
        app_state["point_prompt"] = None

        print(f"Image uploaded and processed successfully: {file.filename}")
        return {"status": "success", "message": f"Image uploaded: {file.filename}"}

    except Exception as e:
        print(f"Error uploading image: {e}")
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

@app.post("/api/config/clear-image")
async def clear_uploaded_image():
    """
    Clear the uploaded image and restore RTSP capability.
    """
    app_state["uploaded_image_path"] = None
    app_state["prompt"] = None
    app_state["point_prompt"] = None

    # Optionally delete the uploaded file
    # (Implement file cleanup if needed)

    print("Uploaded image cleared")
    return {"status": "success", "message": "Local image cleared"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: 
            # Handle incoming messages for point prompts
            data = await websocket.receive_json()
            if data.get("type") == "point_prompt":
                app_state["point_prompt"] = data.get("points")
                app_state["prompt"] = None # Clear text prompt
                print(f"Received point prompt: {app_state['point_prompt']}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- 7. Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)