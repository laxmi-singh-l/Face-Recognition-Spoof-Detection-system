# ============================================================================
# FILE: face_recognition/face_detector.py
# PURPOSE: Face detection using MTCNN
# ============================================================================

import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

class FaceDetector:
    """
    Face Detection using MTCNN
    Detects faces and returns bounding boxes
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            image_size=224,
            margin=0,
            min_face_size=80,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device,
            keep_all=True
        )
    
    def detect_faces(self, image):
        """
        Detect faces in image
        Args:
            image: BGR image from OpenCV
        Returns:
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            confidences: List of detection confidences
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(rgb_image)
        
        if boxes is None:
            return [], []
        
        return boxes.tolist(), probs.tolist()
    
    def crop_face(self, image, box, margin=20):
        """
        Crop face from image with margin
        Args:
            image: BGR image
            box: (x1, y1, x2, y2)
            margin: Extra pixels around face
        Returns:
            cropped_face: BGR image
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # Add margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        return image[y1:y2, x1:x2]
