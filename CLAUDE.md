# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Vision Monitor Dashboard - A Flask-based web application for object detection and counting using Meta's Segment Anything Model (SAM). Built for GSPE (Grahasumber Prima Elektronik) to monitor compliance by counting objects in uploaded images against configurable thresholds.

## Claude Code Storage System

**IMPORTANT:** This repository uses specific folders for Claude Code data storage:

### Storage Locations
- **Plans:** `.claude/plans/` - Implementation plans created during plan mode
- **History:** `.claude/history/` - Conversation history and session data

### Notes
- These folders should be added to `.gitignore` to avoid committing Claude Code internal data
- Plans are created when using plan mode to organize complex implementation tasks
- History preserves context across sessions for continuity

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install SAM library
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Running the Application
```bash
# Development mode (auto-reload enabled)
python app.py

# Access at http://localhost:5000
```

### Download Model Weights
```bash
# Use the provided script
python download_weights.py

# Or manually download from:
# https://github.com/facebookresearch/segment-anything#model-checkpoints
# Place .pth file in model/weights/
```

## Architecture

### Application Flow
1. **Frontend (templates/index.html + static/js/dashboard.js)**: Single-page UI with drag-and-drop image upload, prompt input, and real-time status display
2. **Backend (app.py)**: Flask server with single POST endpoint `/analyze` that processes images
3. **AI Engine (model/sam_engine.py)**: SAM wrapper that performs segmentation, falls back to demo mode if SAM unavailable
4. **Data Flow**: User uploads image + prompt → Backend saves to `static/uploads/` → SAM processes → Returns count + annotated image URL → Frontend updates UI

### Demo Mode vs Production Mode
- **Demo Mode**: Activated automatically when SAM model weights are missing or `segment-anything` library not installed. Returns mock counts and basic image overlays.
- **Production Mode**: Requires SAM model checkpoint in `model/weights/sam_vit_h_4b8939.pth` (2.4GB). Performs actual object segmentation.

The app gracefully handles both modes - check console output for "Demo Mode" warnings.

### Status Logic (Client-Side)
Status determination happens in `static/js/dashboard.js`:
- **Approved (Green)**: `detected_count >= maximum_limit` AND `maximum_limit > 0`
- **Waiting (Orange)**: All other cases

This allows instant UI updates without server round-trips.

### File Upload Flow
1. Client validates file type (JPG/JPEG/PNG) and size (16MB max)
2. FileReader creates preview thumbnail + displays main preview
3. Image card component shows thumbnail + filename with remove button
4. On "Run", sends multipart/form-data to `/analyze`
5. Server saves as `raw_{timestamp}.{ext}` and `processed_{timestamp}.{ext}`
6. Response includes processed image URL which replaces preview

## Configuration Notes

### Flask Configuration (app.py)
- `UPLOAD_FOLDER`: `static/uploads/` - processed images stored here
- `MAX_CONTENT_LENGTH`: 16MB file size limit
- `ALLOWED_EXTENSIONS`: Only PNG, JPG, JPEG accepted

### UI Theme
Primary color: `#003473` (Deep Navy) - used consistently across:
- Headers and navigation
- Primary buttons
- Progress bars
- Modal titles

Tailwind CSS via CDN - no build step required.

### SAM Model Configuration
Default model type: `vit_h` (highest accuracy, 2.4GB)
Alternative models: `vit_l` (1.2GB) or `vit_b` (375MB)

To use different model, modify `model_type` parameter in `SamPredictorWrapper.__init__()` and download corresponding checkpoint.

## Key Implementation Details

### Image Card Component
After file upload, two visual representations exist:
- **Main Container**: Full-size preview/result image (`templates/index.html:69-71`)
- **Image Section**: Thumbnail card with filename + remove button (`templates/index.html:74-95`)

Both update simultaneously via shared FileReader result.

### Modal System
"How To Prompt" modal (`templates/index.html:191-279`) provides SAM prompting guidance:
- Opens via button next to "Prompt" heading
- Closes via X button, backdrop click, or Escape key
- Contains 6 practical tips for effective object detection prompts

### Frontend State Management
`static/js/dashboard.js` manages two global state variables:
- `currentImageFile`: Selected File object (null when no image)
- `isProcessing`: Boolean flag preventing duplicate analysis requests

State synchronization ensures upload area, preview, image card, and remove button stay consistent.

## API Contract

### POST /analyze
**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `file`: Image binary (required)
  - `prompt`: Text string describing object to detect (required)

**Success Response (200):**
```json
{
  "success": true,
  "message": "Analysis complete",
  "data": {
    "detected_count": 5,
    "prompt_used": "helmet",
    "result_image_url": "/static/uploads/processed_12345.jpg?t=timestamp"
  }
}
```

**Error Response (400/500):**
```json
{
  "success": false,
  "error": "Error message description"
}
```

## Model Integration Notes

SAM integration is designed for future enhancement with GroundingDINO for text-to-box conversion. Current implementation:
- Text prompts are accepted but not fully utilized by base SAM
- `_demo_segmentation()` generates random circular masks for demonstration
- Production integration would require GroundingDINO to convert text prompts to bounding boxes, then feed to SAM

For full text-prompt functionality, integrate GroundingDINO:
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

Then implement `_text_to_boxes()` method in `sam_engine.py`.

## Troubleshooting

### "Demo Mode" Warning
SAM model not loaded. Either:
1. Model weights missing from `model/weights/`
2. `segment-anything` library not installed
3. CUDA/GPU issues preventing model initialization

Application continues working with mock detection results.

### Image Upload Fails
- Verify `static/uploads/` directory exists and is writable
- Check file size under 16MB
- Confirm file extension is .jpg, .jpeg, or .png

### GPU Not Detected
Check CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

PyTorch may need reinstallation with correct CUDA version (see requirements.txt).
