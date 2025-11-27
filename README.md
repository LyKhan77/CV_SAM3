# AI Vision Monitor Dashboard

A web-based Computer Vision application for object detection and counting using Meta's Segment Anything Model (SAM).

## Features

- Upload and analyze images for specific object detection
- Natural language prompts for object identification
- Real-time object counting and visualization
- Status monitoring based on configurable thresholds
- Modern, responsive UI built with Tailwind CSS

## Project Structure

```
Segmentation_image/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── model/
│   ├── __init__.py
│   ├── sam_engine.py          # SAM model wrapper
│   └── weights/               # Model checkpoint storage
├── static/
│   ├── css/
│   ├── js/
│   │   └── dashboard.js       # Frontend logic
│   ├── images/
│   │   └── logo.png          # GSPE logo
│   └── uploads/              # Temporary image storage
└── templates/
    └── index.html            # Main dashboard UI
```

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install SAM (Segment Anything Model)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. Download SAM Model Weights

Download the model checkpoint from:
https://github.com/facebookresearch/segment-anything#model-checkpoints

For best performance, use `sam_vit_h_4b8939.pth` and place it in:
```
model/weights/sam_vit_h_4b8939.pth
```

### 4. (Optional) GPU Support

For CUDA GPU support, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Running the Application

### Development Mode

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Usage

1. **Upload Image**: Click or drag an image file into the upload area
2. **Enter Prompt**: Type the object name you want to detect (e.g., "helmet", "person", "box")
3. **Set Maximum Limit**: Enter the expected maximum count
4. **Run Analysis**: Click the "Run" button to process the image
5. **View Results**: The processed image will show detected objects with segmentation masks

## Status Logic

- **Approved (Green)**: Detected count ≥ Maximum limit
- **Waiting (Orange)**: Detected count < Maximum limit

## Demo Mode

The application includes a demo mode that works without the full SAM model installation. In demo mode:
- The system will generate mock detection results
- Useful for testing the UI and workflow without GPU requirements
- To use the full SAM functionality, complete the model setup steps above

## API Endpoint

### POST /analyze

Analyzes an uploaded image for object detection.

**Request:**
- `Content-Type`: multipart/form-data
- `file`: Image file (JPG, JPEG, PNG)
- `prompt`: Text description of object to detect

**Response:**
```json
{
  "success": true,
  "message": "Analysis complete",
  "data": {
    "detected_count": 5,
    "prompt_used": "helmet",
    "result_image_url": "/static/uploads/processed_12345.jpg"
  }
}
```

## Technical Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **AI Engine**: Meta SAM (Segment Anything Model)
- **Image Processing**: OpenCV, NumPy
- **Deep Learning**: PyTorch

## Configuration

- **Max Upload Size**: 16MB (configurable in app.py)
- **Allowed File Types**: JPG, JPEG, PNG
- **Default Port**: 5000
- **Theme Color**: #003473 (Deep Navy)

## Troubleshooting

### Model Loading Issues

If you see "Demo Mode" warnings:
1. Verify SAM model weights are in `model/weights/`
2. Check that segment-anything is installed
3. Ensure CUDA is available if using GPU

### Image Upload Errors

- Check file size is under 16MB
- Verify file format is JPG, JPEG, or PNG
- Ensure the `static/uploads/` directory exists and is writable

## Future Enhancements

- Integration with GroundingDINO for text-to-box conversion
- Support for video stream processing
- Batch image processing
- Export detection results as JSON/CSV
- User authentication and session management

## License

This project is developed for GSPE (Grahasumber Prima Elektronik).

## Support

For issues or questions, please contact the development team.
