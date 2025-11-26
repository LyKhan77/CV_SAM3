# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered computer vision monitoring dashboard that performs real-time object segmentation and detection using Meta's SAM-3 (Segment Anything Model) on RTSP video streams. The application consists of a FastAPI backend with WebSocket support and a responsive HTML/JavaScript frontend.

## Architecture

### Backend Structure
- **main.py**: FastAPI application with API endpoints, WebSocket handling, and application state management
- **model.py**: SAM-3 model loading, video processing pipeline, and AI inference logic
- **Centralized State**: All configuration stored in `app_state` dictionary (RTSP URL, prompts, limits, model settings)

### Frontend Structure
- **web_app/templates/index.html**: Responsive dashboard using Tailwind CSS and Inter font
- **web_app/static/app.js**: Client-side WebSocket communication, UI updates, and user interactions

### Core Components
1. **Video Processing Loop**: Background async task in `model.py:32` that processes RTSP frames every 10 frames
2. **Real-time Communication**: WebSocket endpoint `/ws/monitor` for streaming video frames and analytics
3. **Object Detection**: Text-based prompts or point-and-click selection for SAM-3 segmentation
4. **Configuration System**: API endpoints for stream URL, prompts, limits, sound, and model settings

## Development Commands

```bash
# Setup virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
# Server runs on http://127.0.0.1:8000
```

## Key Technical Details

### Model Configuration
- Uses `facebook/sam3` from HuggingFace transformers
- CUDA GPU support automatically detected and preferred
- Model loading happens on application startup in `main.py:52`

### Video Processing Pipeline
- RTSP streams processed using OpenCV with FFMPEG backend
- Frames processed every 10 iterations for performance (`PROCESS_EVERY_N_FRAMES = 10`)
- Base64 encoding for WebSocket video frame transmission
- SAM-3 inference with separate image and text input processing

### WebSocket Communication
- Real-time updates include video frames and analytics data
- Point-and-click object selection sends normalized coordinates via WebSocket
- Bidirectional communication for configuration updates

### State Management Patterns
- All settings stored in `app_state` dictionary in `main.py:23`
- Frontend maintains `frontendState` object for UI state in `app.js:15`
- Configuration changes propagate through API calls to backend state

## Critical Files for Modifications

- **main.py:51-58**: Application lifespan and model initialization
- **model.py:71-100**: AI inference logic and mask processing
- **model.py:102-119**: Frame encoding and WebSocket broadcasting
- **app.js:90-120**: Point-and-click coordinate handling
- **app.js:164-211**: Dashboard update logic for real-time UI

## Development Notes

- No formal testing framework - use manual testing via web interface
- Async/await patterns used throughout for performance
- Frontend assets served directly by FastAPI static file mounting
- Point coordinates normalized (0-1) for cross-resolution compatibility
- Error handling present but could be enhanced for production use

## Common Issues and Solutions

- **Model Loading**: First startup may be slow due to SAM-3 model download
- **CUDA Memory**: Monitor GPU memory usage with high-resolution streams
- **RTSP Connectivity**: Ensure network access to camera streams and proper URL format
- **WebSocket Disconnections**: Frontend handles reconnection attempts automatically