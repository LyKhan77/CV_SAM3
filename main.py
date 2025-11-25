import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

# Import logic from model.py
from model import load_model, video_processing_loop

# --- 1. Pydantic Models ---
class StreamRequest(BaseModel): url: str
class PromptRequest(BaseModel): object_name: str
class LimitRequest(BaseModel): value: int
class SoundRequest(BaseModel): enabled: bool

# --- 2. Application State & WebSocket Manager ---
app_state: Dict = {
    "rtsp_url": None,
    "prompt": None,
    "max_limit": 100,
    "sound_enabled": False,
    "model": None,
    "processor": None,
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
    """Handles application startup and shutdown events."""
    print("--- Application Startup ---")
    # Load the model and processor and store them in the app_state
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
    return {"status": "success", "message": f"Prompt set to '{request.object_name}'"}

@app.post("/api/config/limit")
async def set_limit(request: LimitRequest):
    app_state["max_limit"] = request.value
    return {"status": "success", "message": f"Limit set to {request.value}"}

@app.post("/api/config/sound")
async def set_sound_toggle(request: SoundRequest):
    app_state["sound_enabled"] = request.enabled
    return {"status": "success", "message": f"Sound notification set to {request.enabled}"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- 7. Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
