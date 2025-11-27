from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import time
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
        prompt = request.form.get('prompt', '').strip()

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only JPG, JPEG, and PNG are allowed.'
            }), 400

        # Save uploaded file
        timestamp = str(int(time.time() * 1000))
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        raw_filename = f'raw_{timestamp}.{file_ext}'
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        file.save(raw_path)

        # Process with SAM model
        if sam_model is None:
            # Demo mode - return mock data
            result_filename = raw_filename  # Use the same image in demo mode
            detected_count = 3  # Mock count
        else:
            # Real processing
            result_filename = f'processed_{timestamp}.{file_ext}'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
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

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize model
    init_model()

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
