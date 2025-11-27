from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import time
import base64
import json
import numpy as np
from model.sam_engine import SamPredictorWrapper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize SAM model
sam_model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_model():
    global sam_model
    try:
        sam_model = SamPredictorWrapper()
        print("SAM model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load SAM model - {str(e)}")
        print("The application will run in demo mode")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only JPG, JPEG, and PNG are allowed.'
            }), 400

        # Get mode and parameters
        mode = request.form.get('mode', 'text')  # 'text' or 'click'
        confidence = float(request.form.get('confidence', 0.3))

        # Save uploaded file
        timestamp = str(int(time.time() * 1000))
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        raw_filename = f'raw_{timestamp}.{file_ext}'
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        file.save(raw_path)

        # Process based on mode
        result_filename = f'processed_{timestamp}.{file_ext}'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        if mode == 'click':
            # Click mode - use point prompts
            points_json = request.form.get('points', '[]')
            import json
            points = json.loads(points_json)

            if not points:
                return jsonify({
                    'success': False,
                    'error': 'No points provided for click mode'
                }), 400

            if sam_model is None:
                # Demo mode
                result_filename = raw_filename
                detected_count = len(points)
            else:
                # Real processing with SAM
                detected_count = sam_model.predict_with_points(
                    raw_path,
                    result_path,
                    points,
                    confidence_threshold=confidence
                )

            # Build response
            result_url = f'/static/uploads/{result_filename}?t={timestamp}'

            return jsonify({
                'success': True,
                'message': 'Analysis complete',
                'data': {
                    'detected_count': detected_count,
                    'result_image_url': result_url
                }
            }), 200

        else:
            # Text mode - use text prompt
            prompt = request.form.get('prompt', '').strip()

            if not prompt:
                return jsonify({
                    'success': False,
                    'error': 'Prompt is required for text mode'
                }), 400

            if sam_model is None:
                # Demo mode - return mock data
                result_filename = raw_filename
                detected_count = 3
            else:
                # Real processing
                detected_count = sam_model.predict_and_visualize(raw_path, result_path, prompt)

            # Build response
            result_url = f'/static/uploads/{result_filename}?t={timestamp}'

            return jsonify({
                'success': True,
                'message': 'Analysis complete',
                'data': {
                    'detected_count': detected_count,
                    'prompt_used': prompt,
                    'result_image_url': result_url
                }
            }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/analyze/instant', methods=['POST'])
def analyze_instant():
    """
    Instant segmentation endpoint for Select Object mode
    Processes single click and returns mask + classification instantly
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400

        # Get click point coordinates
        point_x = request.form.get('x')
        point_y = request.form.get('y')

        if point_x is None or point_y is None:
            return jsonify({
                'success': False,
                'error': 'Click coordinates required'
            }), 400

        # Save uploaded file temporarily
        timestamp = str(int(time.time() * 1000))
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        temp_filename = f'temp_{timestamp}.{file_ext}'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        # Create point dictionary
        point = {
            'x': int(point_x),
            'y': int(point_y),
            'label': 1  # Foreground
        }

        # Process with SAM 2 + CLIP
        if sam_model is None:
            # Demo mode
            result = {
                'mask': np.zeros((100, 100), dtype=bool),
                'object_label': 'demo_object',
                'confidence': 0.85,
                'bbox': [int(point_x)-50, int(point_y)-50, int(point_x)+50, int(point_y)+50],
                'top_3': [
                    {'label': 'demo_object', 'confidence': 0.85},
                    {'label': 'unknown', 'confidence': 0.10},
                    {'label': 'item', 'confidence': 0.05}
                ]
            }
        else:
            # Real SAM 2 + CLIP processing
            result = sam_model.predict_single_point_instant(temp_path, point)

        # Encode mask as base64 for transmission
        mask_uint8 = (result['mask'].astype(np.uint8) * 255)
        mask_bytes = mask_uint8.tobytes()
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Build response
        return jsonify({
            'success': True,
            'data': {
                'mask': mask_base64,
                'mask_shape': result['mask'].shape,
                'object_label': result['object_label'],
                'confidence': float(result['confidence']),
                'bbox': result['bbox'],
                'top_3': result['top_3']
            }
        }), 200

    except Exception as e:
        # Clean up temporary file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': False,
            'error': f'Instant analysis failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize model
    init_model()

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
