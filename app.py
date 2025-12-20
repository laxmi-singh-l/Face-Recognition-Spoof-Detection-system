"""
Flask Web Application for Face Recognition + Spoof Detection System
Complete backend with REST API endpoints
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import json

# Import custom modules
from config import Config
from inference.pipeline import SecurityPipeline

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Ensure required directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# Initialize security pipeline
try:
    pipeline = SecurityPipeline()
    print("✓ Security pipeline initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Could not initialize pipeline - {e}")
    pipeline = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ROUTES - Web Pages
# ============================================================================

@app.route('/')
def index():
    """Main page - Face detection interface"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page - View registered faces and logs"""
    if pipeline is None:
        return "Pipeline not initialized", 500
    
    known_faces = pipeline.face_database.list_faces()
    return render_template('dashboard.html', 
                         faces=known_faces,
                         total_faces=len(known_faces))


@app.route('/analytics')
def analytics():
    """Analytics page - Detection statistics"""
    return render_template('analytics.html')


# ============================================================================
# API ENDPOINTS - Image Processing
# ============================================================================

@app.route('/api/process_image', methods=['POST'])
def process_image():
    """
    Process uploaded image for face detection and recognition
    Returns: Detection results with bounding boxes and labels
    """
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400
    
    try:
        # Save uploaded image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Run detection pipeline
        annotated_frame, results = pipeline.process_frame(image)
        
        # Save annotated result
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_frame)
        
        # Convert numpy types to native Python types for JSON
        results_json = []
        for result in results:
            results_json.append({
                'box': [float(x) for x in result['box']],
                'confidence': float(result['confidence']),
                'liveness_score': float(result['liveness_score']),
                'is_real': bool(result['is_real']),
                'name': str(result['name']),
                'similarity': float(result['similarity']),
                'access_granted': bool(result['access_granted'])
            })
        
        return jsonify({
            'success': True,
            'results': results_json,
            'original_image': filename,
            'output_image': result_filename,
            'timestamp': timestamp
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_video_frame', methods=['POST'])
def process_video_frame():
    """
    Process single frame from video stream
    Used for real-time webcam detection
    """
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        # Get image data from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image data'}), 400
        
        file = request.files['image']
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Process frame
        annotated_frame, results = pipeline.process_frame(image)
        
        # Convert results for JSON
        results_json = []
        for result in results:
            results_json.append({
                'name': str(result['name']),
                'is_real': bool(result['is_real']),
                'liveness_score': float(result['liveness_score']),
                'access_granted': bool(result['access_granted'])
            })
        
        return jsonify({
            'success': True,
            'results': results_json
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - Face Database Management
# ============================================================================

@app.route('/api/add_face', methods=['POST'])
def add_face():
    """
    Register new face in the database
    Requires: name (string), image (file)
    """
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    # Validate inputs
    if 'name' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Name and image required'}), 400
    
    name = request.form['name'].strip()
    file = request.files['image']
    
    if not name:
        return jsonify({'error': 'Name cannot be empty'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(filepath)
        
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Detect face
        boxes, confidences = pipeline.face_detector.detect_faces(image)
        
        if len(boxes) == 0:
            os.remove(filepath)
            return jsonify({'error': 'No face detected in image'}), 400
        
        if len(boxes) > 1:
            os.remove(filepath)
            return jsonify({'error': 'Multiple faces detected. Use image with single face'}), 400
        
        # Crop face
        face = pipeline.face_detector.crop_face(image, boxes[0])
        
        # Check if face is real (liveness check)
        from liveness_model.preprocessing import preprocess_face
        face_tensor = preprocess_face(face).to(pipeline.device)
        with pipeline.liveness_model.eval():
            liveness_score = pipeline.liveness_model(face_tensor).item()
        
        if liveness_score < Config.LIVENESS_THRESHOLD:
            os.remove(filepath)
            return jsonify({
                'error': 'Spoof detected! Please use a real photo, not a screen or printout',
                'liveness_score': float(liveness_score)
            }), 400
        
        # Add to database
        pipeline.face_database.add_face(name, face)
        
        # Save face image
        face_filename = f"{name.replace(' ', '_')}.jpg"
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'faces', face_filename)
        os.makedirs(os.path.dirname(face_path), exist_ok=True)
        cv2.imwrite(face_path, face)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Successfully registered {name}',
            'face_image': face_filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remove_face', methods=['POST', 'DELETE'])
def remove_face():
    """Remove face from database"""
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error': 'Name required'}), 400
    
    name = data['name']
    
    try:
        pipeline.face_database.remove_face(name)
        return jsonify({
            'success': True,
            'message': f'Successfully removed {name}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_faces', methods=['GET'])
def list_faces():
    """Get list of all registered faces"""
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        faces = pipeline.face_database.list_faces()
        return jsonify({
            'success': True,
            'faces': faces,
            'count': len(faces)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_face_info/<name>', methods=['GET'])
def get_face_info(name):
    """Get information about specific face"""
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        if name not in pipeline.face_database.database:
            return jsonify({'error': 'Face not found'}), 404
        
        face_filename = f"{name.replace(' ', '_')}.jpg"
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'faces', face_filename)
        
        return jsonify({
            'success': True,
            'name': name,
            'has_image': os.path.exists(face_path),
            'image_path': face_filename if os.path.exists(face_path) else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - System Info
# ============================================================================

@app.route('/api/system_status', methods=['GET'])
def system_status():
    """Get system status and statistics"""
    try:
        import torch
        
        status = {
            'pipeline_initialized': pipeline is not None,
            'device': str(pipeline.device) if pipeline else 'N/A',
            'cuda_available': torch.cuda.is_available(),
            'registered_faces': len(pipeline.face_database.list_faces()) if pipeline else 0,
            'model_loaded': os.path.exists(os.path.join(Config.MODEL_DIR, 'best_model.pth')),
            'version': '1.0.0'
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    try:
        info = {
            'face_detector': 'MTCNN',
            'face_recognizer': 'FaceNet (InceptionResnetV1)',
            'liveness_model': 'MobileNetV2',
            'input_size': Config.IMG_SIZE,
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            'recognition_threshold': Config.RECOGNITION_THRESHOLD,
            'liveness_threshold': Config.LIVENESS_THRESHOLD
        }
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# STATIC FILES
# ============================================================================

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/faces/<filename>')
def face_image(filename):
    """Serve face images"""
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], 'faces'),
        filename
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🔐 Face Recognition + Spoof Detection System")
    print("=" * 60)
    print(f"✓ Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"✓ Model directory: {Config.MODEL_DIR}")
    
    if pipeline:
        print(f"✓ Pipeline initialized on {pipeline.device}")
        print(f"✓ Registered faces: {len(pipeline.face_database.list_faces())}")
    else:
        print("⚠ Pipeline not initialized - limited functionality")
    
    print("\n🚀 Starting Flask server...")
    print("📱 Access at: http://localhost:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )