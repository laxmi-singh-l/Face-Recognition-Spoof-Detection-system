# ============================================================================
# FILE: dataset_preparation.py
# PURPOSE: Helper script to prepare and organize dataset
# ============================================================================

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from facenet_pytorch import MTCNN
import torch

class DatasetPreparation:
    """Prepare and organize dataset for training"""
    
    def __init__(self, source_dir, output_dir='dataset'):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(device=self.device, select_largest=True)
    
    def extract_faces(self, img_path, output_path):
        """Extract and save face from image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb_img)
            
            if boxes is None or len(boxes) == 0:
                return False
            
            # Get first face
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Add margin
            margin = 20
            h, w = img.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            # Crop face
            face = img[y1:y2, x1:x2]
            
            # Resize to 224x224
            face_resized = cv2.resize(face, (224, 224))
            
            # Save
            cv2.imwrite(output_path, face_resized)
            return True
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False
    
    def organize_dataset(self, real_dir, fake_dir, split_ratio=(0.7, 0.15, 0.15)):
        """
        Organize images into train/val/test splits
        Args:
            real_dir: Directory containing real face images
            fake_dir: Directory containing fake/spoof images
            split_ratio: (train, val, test) ratios
        """
        print("="*60)
        print("Dataset Organization")
        print("="*60)
        
        # Get all images
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(real_images)} real images")
        print(f"Found {len(fake_images)} fake images")
        
        # Split data
        train_ratio, val_ratio, test_ratio = split_ratio
        
        def split_data(images, label):
            # Train/temp split
            train, temp = train_test_split(images, 
                                          test_size=(val_ratio + test_ratio),
                                          random_state=42)
            # Val/test split
            val, test = train_test_split(temp,
                                        test_size=test_ratio/(val_ratio + test_ratio),
                                        random_state=42)
            return train, val, test
        
        real_train, real_val, real_test = split_data(real_images, 'real')
        fake_train, fake_val, fake_test = split_data(fake_images, 'fake')
        
        # Process and save
        splits = {
            'train': {
                'real': real_train,
                'fake': fake_train
            },
            'val': {
                'real': real_val,
                'fake': fake_val
            },
            'test': {
                'real': real_test,
                'fake': fake_test
            }
        }
        
        for split_name, data in splits.items():
            print(f"\nProcessing {split_name} set...")
            
            for label, images in data.items():
                output_dir = os.path.join(self.output_dir, split_name, label)
                os.makedirs(output_dir, exist_ok=True)
                
                success = 0
                for i, img_path in enumerate(tqdm(images, desc=f"{split_name}/{label}")):
                    output_path = os.path.join(output_dir, f"{label}_{i:04d}.jpg")
                    if self.extract_faces(img_path, output_path):
                        success += 1
                
                print(f"  {label}: {success}/{len(images)} images processed")
        
        print("\n✓ Dataset organization complete!")
        print(f"  Train: {len(real_train + fake_train)} images")
        print(f"  Val: {len(real_val + fake_val)} images")
        print(f"  Test: {len(real_test + fake_test)} images")


# Usage example
if __name__ == '__main__':
    print("Dataset Preparation Tool")
    print("="*60)
    
    real_dir = input("Enter path to real images directory: ").strip()
    fake_dir = input("Enter path to fake images directory: ").strip()
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("✗ Invalid directories")
        sys.exit(1)
    
    prep = DatasetPreparation(source_dir='')
    prep.organize_dataset(real_dir, fake_dir)