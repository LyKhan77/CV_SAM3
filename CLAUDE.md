# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Computer Vision Monitoring Dashboard that uses Meta's SAM 3 (Segment Anything Model) to detect and segment objects in real-time video streams or static images. The system provides a web-based interface for configuring prompts, setting detection limits, and monitoring analytics.

## Architecture

### Core Components

- **main.py**: FastAPI backend server that handles:
  - WebSocket connections for real-time video streaming
  - REST API endpoints for configuration management
  - File upload handling for static image processing
  - State management for RTSP streams, prompts, and model configuration

- **model.py**: Computer vision processing module containing:
  - SAM 3 model loading and inference logic
  - Video processing loop with RTSP stream handling
  - Optimized post-processing for mask generation and filtering
  - Support for both text prompts and point-based selection

- **web_app/**: Frontend web application
  - **templates/index.html**: Main dashboard UI using Tailwind CSS
  - **static/app.js**: Client-side JavaScript for WebSocket communication, UI updates, and file upload handling

### Data Flow

1. **Input Sources**:
   - RTSP video streams (IP cameras)
   - Static image uploads

2. **Processing Pipeline**:
   - Frame capture and resizing (max 1024px for efficiency)
   - SAM 3 inference with text or point prompts
   - Mask optimization with morphological operations
   - Object counting and visualization

3. **Output**:
   - Real-time video feed with segmentation overlays
   - Analytics data (object count, status, limits)
   - WebSocket broadcast to connected clients

## Common Commands

### Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Test imports (run before starting)
python test_imports.py

# Start the development server
python main.py
```

### Testing and Debugging

```bash
# Test model imports and CUDA availability
python test_imports.py

# Hugging Face login (if needed for model access)
python login_helper.py

# Run with debug batch file (Windows)
run_server_debug.bat
```

### Key Configuration

- **Server**: Runs on `http://127.0.0.1:8000` by default
- **WebSocket**: Connect to `ws://127.0.0.1:8000/ws/monitor`
- **Upload Directory**: `uploads/` (created automatically)
- **Max Input Size**: 1024px (for SAM 3 performance)
- **Frame Processing**: Every 5th frame for RTSP streams (for performance)

## Model Configuration

The system supports dynamic model configuration through API endpoints:

- **Confidence Threshold**: Filter masks by confidence score (0.0-1.0)
- **Mask Threshold**: Binary threshold for mask generation (0.0-1.0)
- **Display Mode**: "segmentation" (contours) or "bounding_box" (rectangles)

## Important Notes

- CUDA support automatically detected and utilized when available
- RTSP streams require proper camera authentication and network access
- Uploaded images are automatically resized to max 1024px for optimal performance
- The system includes aggressive memory management and cache clearing for stable operation
- Morphological operations (opening/closing) are applied to clean up noisy masks

## API Endpoints

- `POST /api/config/stream` - Set RTSP URL
- `POST /api/config/prompt` - Set text prompt for object detection
- `POST /api/config/limit` - Set detection limit
- `POST /api/config/sound` - Toggle notification sound
- `POST /api/config/model` - Update model configuration
- `POST /api/upload/image` - Upload static image for processing
- `POST /api/config/clear-image` - Clear uploaded image
- `WebSocket /ws/monitor` - Real-time video and analytics stream