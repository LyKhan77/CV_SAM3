# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AI Computer Vision monitoring system** built with FastAPI (backend) and vanilla JavaScript (frontend). The application uses **Meta's SAM 3 (Segment Anything Model 3)** for real-time object detection and segmentation from multiple input sources: RTSP streams (IP cameras/webcams), uploaded videos, or static images.

**Core Architecture:** Client-Server with WebSocket streaming
- **Backend:** Python FastAPI server with async processing loop
- **Frontend:** HTML/JS single-page application
- **AI Engine:** SAM 3 via Hugging Face Transformers
- **Real-time Communication:** WebSocket for streaming processed frames and analytics

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install PyTorch with CUDA first (CRITICAL - must be done before other deps)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start server (runs on http://127.0.0.1:8000)
python main.py
```

### Testing
```bash
# Test imports and CUDA availability
python test_imports.py
```

## System Architecture

### Core Components

1. **main.py** - FastAPI application and API endpoints
   - Manages `app_state`: global dictionary containing all runtime state (model, prompts, input modes, video metadata)
   - `ConnectionManager`: WebSocket manager for broadcasting processed frames
   - REST API endpoints for configuration (`/api/config/*`, `/api/upload/*`)
   - WebSocket endpoint (`/ws/monitor`) for bidirectional communication

2. **model.py** - AI inference and video processing
   - `load_model()`: Initializes SAM 3 model and processor (runs on GPU if available)
   - `video_processing_loop()`: Async background loop that continuously:
     - Acquires frames from active input source
     - Runs SAM 3 inference when prompted
     - Post-processes masks (NMS, morphology, smoothing)
     - Encodes frames as base64 JPEG and broadcasts via WebSocket
   - `process_sam3_outputs_optimized()`: Post-processing pipeline with confidence filtering, NMS, and mask refinement
   - `draw_masks()`: Renders segmentation contours with smoothing on frames

3. **web_app/** - Frontend assets
   - `templates/index.html`: Single-page UI
   - `static/app.js`: WebSocket client, DOM updates, user interactions
   - `static/logo.png`: Application logo

### Data Flow

```
User Action (Frontend)
  → REST API Call (e.g., /api/config/prompt)
  → Updates app_state
  → video_processing_loop detects state change
  → Acquires frame from input source
  → Runs SAM 3 inference (if should_segment=True)
  → Post-processes masks
  → Encodes frame + analytics as JSON
  → Broadcasts via WebSocket
  → Frontend receives message
  → Updates video feed, counters, status badges
```

### Input Modes

The system supports three mutually exclusive input modes (controlled via `app_state["input_mode"]`):

1. **RTSP Mode** (`"rtsp"`)
   - Connects to RTSP streams or local webcams (device index like "0")
   - Uses OpenCV's `VideoCapture` with `CAP_FFMPEG`
   - Processes every 5th frame for performance

2. **Video Upload Mode** (`"video"`)
   - Processes uploaded video files from `uploads/` directory
   - Supports seek/play/pause controls
   - Loops playback automatically
   - Frame rate controlled by video's native FPS

3. **Image Mode** (`"image"`)
   - Processes single static images from `uploads/` directory
   - Images resized to max 1024px for SAM performance
   - Uses caching to avoid reprocessing unless state changes

### State Management

All application state is stored in the `app_state` dictionary (defined in main.py:38). Key fields:

- **Input Sources:** `rtsp_url`, `uploaded_image_path`, `video_file_path`
- **AI Configuration:** `model`, `processor`, `confidence_threshold`, `mask_threshold`
- **Prompts:** `prompt` (text), `point_prompt` (click coordinates)
- **Processing Control:** `should_segment` (Run/Clear Mask toggle)
- **Video Playback:** `video_current_frame`, `video_total_frames`, `video_fps`, `video_playing`
- **Export Cache:** `last_processed_frame`, `last_raw_masks` (for snapshot feature)

State changes are detected in the `video_processing_loop` via state hashing for efficient caching.

### AI Processing Pipeline

When `should_segment=True` and a prompt exists:

1. **Frame Acquisition:** Get frame from current input source
2. **Preprocessing:** Resize to max 1024px (VRAM optimization)
3. **Inference:** Pass to SAM 3 with text or point prompt
4. **Post-processing** (in `process_sam3_outputs_optimized`):
   - Filter by confidence threshold
   - Resize masks to original frame size
   - Apply NMS (Non-Maximum Suppression) to remove overlaps
   - Morphological operations (opening, closing) to clean masks
   - Filter small artifacts (< 200px area)
5. **Rendering:** Draw smoothed contours on frame
6. **Export:** Store raw masks + frame in `app_state` for snapshot feature

### Snapshot Export System

When `/api/snapshot/save` is called:
- Creates `ObjectList/{input_name}/` directory
- Saves each detected object as transparent PNG (BGRA format)
- Crops to bounding box with 5px padding
- Generates `data.json` with metadata (bbox coordinates)
- Returns base64 thumbnails for UI display

## Important Implementation Details

### GPU Memory Management
- Model loads to CUDA if available, CPU fallback
- Input frames resized to 1024px max dimension to reduce VRAM usage
- `torch.cuda.empty_cache()` called on inference errors
- Inference runs every 5th frame for RTSP/video modes

### WebSocket Communication
- Frontend connects to `/ws/monitor`
- Server broadcasts JSON payloads with:
  - `video_frame`: base64-encoded JPEG (quality=70)
  - `analytics`: object count, status, warnings, video metadata
- Bidirectional: Frontend can send point prompts via WebSocket

### Prompt System
- **Text Prompts:** Natural language (e.g., "person with red hat")
- **Point Prompts:** Click coordinates (normalized 0-1), converted to absolute pixels based on current frame size
- Text and point prompts are mutually exclusive (setting one clears the other)

### State Caching (Image Mode)
- Static images use hash-based caching to avoid reprocessing
- Cache invalidates when:
  - Prompts change
  - Model config changes
  - `should_segment` toggles
  - Image path changes
- RTSP/video modes always regenerate frames

### File Upload Handling
- Images: Decoded, resized if >1024px, saved to `uploads/`
- Videos: Validated, metadata extracted (fps, total_frames), saved to `uploads/`
- Switching input modes automatically cleans up conflicting resources (e.g., releasing VideoCapture)

## Common Workflows

### Modifying Model Inference
- Edit `model.py:process_sam3_outputs_optimized()` for post-processing changes
- Adjust `confidence_threshold` or `mask_threshold` in API calls or `app_state`
- NMS threshold: hardcoded at 0.5 in model.py:197

### Adding New API Endpoints
- Define Pydantic models in main.py (e.g., `class NewRequest(BaseModel)`)
- Create endpoint with `@app.post("/api/config/new")`
- Update `app_state` dictionary
- Frontend will detect changes in next `video_processing_loop` iteration

### Debugging Inference Issues
- Check console logs for `[ERROR]` messages
- Look for "INFO: Inference finished but 0 objects" (indicates filtering too aggressive)
- Verify CUDA availability: `test_imports.py`
- Monitor VRAM usage if getting OOM errors

### Working with Masks
- Raw masks stored in `app_state["last_raw_masks"]` as numpy arrays (uint8, 0-255)
- Smoothing applied only during rendering (`draw_masks`) for UI display
- Export uses raw masks to preserve fidelity

## Dependencies

- **FastAPI/Uvicorn:** Web server and async framework
- **OpenCV (cv2):** Frame capture, image processing, morphology
- **PyTorch/Torchvision:** Deep learning backend, NMS operations
- **Transformers:** SAM 3 model loading and inference
- **Pillow:** Image format conversions for SAM processor

## Critical Notes

- **CUDA Installation:** PyTorch MUST be installed with CUDA support before other dependencies
- **Model Loading:** SAM 3 is loaded from Hugging Face main branch (bleeding edge)
- **Windows Path Handling:** Use raw strings or forward slashes in paths
- **Async Context:** `video_processing_loop` runs as background task via `asyncio.create_task()` in lifespan event
- **Resource Cleanup:** VideoCapture instances must be released when switching modes to avoid leaks
