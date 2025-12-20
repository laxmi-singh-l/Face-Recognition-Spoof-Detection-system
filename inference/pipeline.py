# ============================================================================
# FILE: inference/pipeline.py
# PURPOSE: Complete detection pipeline
# ============================================================================

import torch
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from face_recognition.face_detector import FaceDetector
from face_recognition.face_embeddings import FaceRecognizer
from face_recognition.database import FaceDatabase
from liveness_model.model import get_model
from liveness_model.preprocessing import preprocess_face
from inference.utils import draw_box

class SecurityPipeline:
    """
    Complete Face Recognition + Spoof Detection Pipeline
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.face_detector = FaceDetector(self.device)
        self.face_recognizer = FaceRecognizer(self.device)
        self.face_database = FaceDatabase(Config.FACE_DB_PATH)
        
        # Load liveness model
        self.liveness_model = get_model('mobilenet', pretrained=False).to(self.device)
        model_path = os.path.join(Config.MODEL_DIR, 'best_model.pth')
        if os.path.exists(model_path):
            self.liveness_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.liveness_model.eval()
            print("✓ Loaded liveness model")
        else:
            print("⚠ Liveness model not found. Train the model first.")
    
    def process_frame(self, frame):
        """
        Process single frame through complete pipeline
        Returns: annotated frame, results dict
        """
        results = []
        
        # Step 1: Face Detection
        boxes, confidences = self.face_detector.detect_faces(frame)
        
        for box, conf in zip(boxes, confidences):
            if conf < Config.CONFIDENCE_THRESHOLD:
                continue
            
            # Crop face
            face = self.face_detector.crop_face(frame, box)
            
            # Step 2: Liveness Detection
            face_tensor = preprocess_face(face).to(self.device)
            with torch.no_grad():
                liveness_score = self.liveness_model(face_tensor).item()
            
            is_real = liveness_score > Config.LIVENESS_THRESHOLD
            
            # Step 3: Face Recognition (only if real)
            if is_real:
                embedding = self.face_recognizer.get_embedding(face)
                name, similarity = self.face_recognizer.match_face(
                    embedding, 
                    self.face_database.database,
                    Config.RECOGNITION_THRESHOLD
                )
            else:
                name = "SPOOF DETECTED"
                similarity = 0.0
            
            # Prepare result
            result = {
                'box': box,
                'confidence': conf,
                'liveness_score': liveness_score,
                'is_real': is_real,
                'name': name,
                'similarity': similarity,
                'access_granted': is_real and name != 'Unknown'
            }
            results.append(result)
            
            # Draw on frame
            color = (0, 255, 0) if result['access_granted'] else (0, 0, 255)
            label = f"{name} ({similarity:.2f})" if is_real else "SPOOF"
            frame = draw_box(frame, box, label, color)
            
            # Add liveness score
            x1, y1 = int(box[0]), int(box[1])
            cv2.putText(frame, f"Live: {liveness_score:.2f}", 
                       (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        return frame, results
    
    def process_video(self, video_path=0):
        """
        Process video stream (0 for webcam)
        """
        cap = cv2.VideoCapture(video_path)
        
        print("Press 'q' to quit, 's' to save snapshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, results = self.process_frame(frame)
            
            # Display
            cv2.imshow('Security System', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('snapshot.jpg', annotated_frame)
                print("✓ Snapshot saved")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pipeline = SecurityPipeline()
    pipeline.process_video(0)  # Use webcam
