#!/usr/bin/env python3
# ============================================================================
# TEST YOUR MODEL ON NEW SATELLITE DATA
# Validates model performance on unseen/different data
# File: test_on_new_data.py
# Run: python test_on_new_data.py
# ============================================================================

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import csv

print("\n" + "#"*70)
print("# TEST MODEL ON NEW SATELLITE DATA")
print("#"*70 + "\n")

# ============================================================================
# STEP 1: LOAD TRAINED MODEL
# ============================================================================

class ViTDetector(nn.Module):
    """Your trained Vision Transformer model"""
    
    def __init__(self, model_name='vit_small_patch16_224', num_queries=10):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.embed_dim
        self.num_queries = num_queries
        
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        if len(features.shape) == 3:
            class_token = features[:, 0, :]
        else:
            features = features.mean(dim=(2, 3))
            class_token = features
        
        pred_boxes = self.bbox_head(class_token).unsqueeze(1)
        pred_logits = self.class_head(class_token).unsqueeze(1)
        
        return {
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits
        }

# ============================================================================
# STEP 2: NEW DATA TESTER
# ============================================================================

class NewDataTester:
    """Test model on new/different satellite data"""
    
    def __init__(self, model_path, test_data_dir, device='cpu'):
        self.device = torch.device(device)
        print(f"Loading trained model from: {model_path}")
        print(f"Test data directory: {test_data_dir}")
        print(f"Using device: {self.device}\n")
        
        # Load model
        self.model = ViTDetector('vit_small_patch16_224', num_queries=10).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.test_data_dir = Path(test_data_dir)
        
        # Transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print("✓ Model loaded successfully\n")
    
    def find_test_images(self):
        """Find all test images"""
        images = []
        
        # Search for images recursively
        for ext in ['*.jpg', '*.JPG', '*.tif', '*.TIF', '*.png', '*.PNG']:
            images.extend(self.test_data_dir.rglob(ext))
        
        # Remove duplicates (keep jpg if both jpg and tif exist)
        unique_images = {}
        for img in images:
            stem = img.stem
            if stem not in unique_images:
                unique_images[stem] = img
            elif img.suffix.lower() == '.jpg':
                unique_images[stem] = img
        
        return sorted(list(unique_images.values()))
    
    def predict_single_image(self, image_path, confidence_threshold=0.5):
        """Predict on single image"""
        
        # Read image
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            try:
                import tifffile
                img = tifffile.imread(str(image_path))
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = img[:, :, :3]
            except:
                img = cv2.imread(str(image_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img_rgb is None:
            return None, None, None
        
        original_h, original_w = img_rgb.shape[:2]
        
        # Transform
        augmented = self.transform(image=img_rgb)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        pred_boxes = predictions['pred_boxes'][0]
        pred_logits = predictions['pred_logits'][0]
        
        probs = F.softmax(pred_logits, dim=-1)[:, 1]
        
        # Filter by confidence
        mask = probs > confidence_threshold
        filtered_boxes = pred_boxes[mask]
        filtered_probs = probs[mask]
        
        # Denormalize
        detections = []
        for box, prob in zip(filtered_boxes, filtered_probs):
            x_min = int(max(0, box[0].item() * original_w))
            y_min = int(max(0, box[1].item() * original_h))
            x_max = int(min(original_w, box[2].item() * original_w))
            y_max = int(min(original_h, box[3].item() * original_h))
            
            if x_max > x_min and y_max > y_min:
                detections.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': float(prob.item()),
                    'area': (x_max - x_min) * (y_max - y_min)
                })
        
        return detections, img_rgb, (original_w, original_h)
    
    def test_all_images(self, confidence_threshold=0.5):
        """Test on all new images"""
        
        images = self.find_test_images()
        
        if len(images) == 0:
            print("❌ No test images found!")
            print(f"Please place test images in: {self.test_data_dir}")
            print("Supported formats: .jpg, .tif, .png")
            return None
        
        print("="*70)
        print(f"TESTING ON {len(images)} NEW IMAGES")
        print("="*70 + "\n")
        
        os.makedirs('results/new_data_tests', exist_ok=True)
        os.makedirs('results/new_data_tests/visualizations', exist_ok=True)
        os.makedirs('results/new_data_tests/annotations', exist_ok=True)
        
        results_csv = Path('results/new_data_tests/test_results.csv')
        
        with open(results_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image_Name', 'Detections', 'Avg_Confidence', 'Status'])
            
            total_detections = 0
            successful = 0
            
            for img_path in tqdm(images, desc="Testing"):
                try:
                    # Predict
                    detections, img_rgb, size = self.predict_single_image(
                        img_path, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    if detections is None:
                        writer.writerow([img_path.name, 0, 0, 'Error: Cannot read image'])
                        continue
                    
                    total_detections += len(detections)
                    successful += 1
                    
                    # Visualize
                    self.visualize_predictions(img_path, detections, img_rgb)
                    
                    # Save annotations
                    ann_data = {
                        "image_name": img_path.name,
                        "image_size": list(size),
                        "detections": detections,
                        "total_detections": len(detections),
                        "model": "ViT-Solar-Panel-Detector",
                        "test_timestamp": datetime.now().isoformat(),
                        "confidence_threshold": confidence_threshold
                    }
                    
                    ann_path = Path('results/new_data_tests/annotations') / f"{img_path.stem}.json"
                    with open(ann_path, 'w') as f:
                        json.dump(ann_data, f, indent=2)
                    
                    # Log results
                    avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
                    writer.writerow([
                        img_path.name,
                        len(detections),
                        f"{avg_conf:.3f}",
                        'Success'
                    ])
                
                except Exception as e:
                    writer.writerow([img_path.name, 0, 0, f'Error: {str(e)}'])
        
        # Summary
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        print(f"\nTotal test images: {len(images)}")
        print(f"Successfully processed: {successful}")
        print(f"Total solar panels detected: {total_detections}")
        
        if successful > 0:
            print(f"Average detections per image: {total_detections / successful:.1f}")
        
        print(f"\nResults saved to:")
        print(f"  - Test CSV: results/new_data_tests/test_results.csv")
        print(f"  - Visualizations: results/new_data_tests/visualizations/")
        print(f"  - Annotations: results/new_data_tests/annotations/")
        print("\n" + "="*70 + "\n")
        
        return {
            'total_images': len(images),
            'successful': successful,
            'total_detections': total_detections
        }
    
    def visualize_predictions(self, image_path, detections, img_rgb):
        """Visualize predictions"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(img_rgb)
        
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            conf = det['confidence']
            
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{conf:.2f}"
            ax.text(x_min, y_min - 5, label, fontsize=8, color='lime',
                   bbox=dict(facecolor='black', alpha=0.7))
        
        ax.set_title(f"{image_path.name} - {len(detections)} panels detected")
        ax.axis('off')
        
        output_path = Path('results/new_data_tests/visualizations') / f"{image_path.stem}_test.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

# ============================================================================
# STEP 3: COMPARISON WITH TRAINING DATA
# ============================================================================

def compare_with_training_data():
    """Compare test results with training results"""
    
    print("\n" + "="*70)
    print("COMPARISON: TRAINING vs TEST DATA")
    print("="*70 + "\n")
    
    # Read training data stats (from final_dataset)
    training_stats = {
        'total_images': 80,
        'total_detections': 3390,
        'avg_detections': 42.4
    }
    
    # Read test data stats
    test_csv = Path('results/new_data_tests/test_results.csv')
    if test_csv.exists():
        import csv
        detections = []
        with open(test_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    detections.append(int(row['Detections']))
                except:
                    pass
        
        test_stats = {
            'total_detections': sum(detections),
            'avg_detections': np.mean(detections) if detections else 0
        }
        
        print("METRIC COMPARISON:")
        print(f"Training Data - Avg panels per image: {training_stats['avg_detections']:.1f}")
        print(f"Test Data     - Avg panels per image: {test_stats['avg_detections']:.1f}")
        
        difference = abs(test_stats['avg_detections'] - training_stats['avg_detections'])
        percentage = (difference / training_stats['avg_detections'] * 100)
        
        print(f"\nDifference: {difference:.1f} panels ({percentage:.1f}%)")
        
        if percentage < 20:
            print("\n✅ MODEL GENERALIZES WELL!")
            print("   Test data shows similar detection patterns to training data")
        elif percentage < 50:
            print("\n⚠️  MODEL SHOWS SOME VARIATION")
            print("   Test data may have different characteristics")
        else:
            print("\n❌ MODEL BEHAVIOR DIFFERS SIGNIFICANTLY")
            print("   Test data may be fundamentally different from training data")
        
        print("\n" + "="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Check if model exists
    if not Path('models/best_model.pth').exists():
        print("❌ Trained model not found!")
        print("Please train the model first using: python scripts/complete_system.py")
        exit(1)
    
    # Create test data directory if it doesn't exist
    test_dir = Path('data/test_new_data')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Test data directory: {test_dir}")
    print(f"\nTo test your model on NEW DATA:")
    print(f"1. Place new satellite images in: {test_dir}/")
    print(f"2. Run this script again")
    print(f"3. Check results in: results/new_data_tests/\n")
    
    # Check if test images exist
    images = []
    for ext in ['*.jpg', '*.JPG', '*.tif', '*.TIF', '*.png', '*.PNG']:
        images.extend(test_dir.rglob(ext))
    
    if len(images) == 0:
        print("="*70)
        print("NO TEST IMAGES FOUND")
        print("="*70)
        print("\nTo test your model on new data:")
        print(f"1. Create folder: {test_dir}/")
        print(f"2. Copy your new satellite images there")
        print(f"3. Run this script again: python test_on_new_data.py")
        print("\nSupported formats: .jpg, .tif, .png")
        print("\n" + "="*70 + "\n")
        return
    
    # Run tests
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = NewDataTester(
        'models/best_model.pth',
        test_dir,
        device=device
    )
    
    results = tester.test_all_images(confidence_threshold=0.5)
    
    # Compare with training
    if results:
        compare_with_training_data()
    
    print("="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print("\nCheck results:")
    print("  - Predictions: results/new_data_tests/test_results.csv")
    print("  - Visualizations: results/new_data_tests/visualizations/")
    print("  - Annotations: results/new_data_tests/annotations/")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()