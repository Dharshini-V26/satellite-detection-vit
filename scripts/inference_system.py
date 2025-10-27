#!/usr/bin/env python3
# ============================================================================
# INFERENCE AND PREDICTION SYSTEM
# Uses trained model to detect solar panels and generate predictions
# File: inference_system.py
# Run: python inference_system.py
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
import csv
from datetime import datetime

print("\n" + "#"*70)
print("# INFERENCE AND PREDICTION SYSTEM")
print("#"*70 + "\n")

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

class ViTDetector(nn.Module):
    """Vision Transformer for detection"""
    
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
# INFERENCE CLASS
# ============================================================================

class SolarPanelInference:
    """Run inference with trained model"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}\n")
        
        # Initialize model
        self.model = ViTDetector('vit_small_patch16_224', num_queries=10).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print("✓ Model loaded successfully\n")
        
        # Transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def predict(self, image_path, confidence_threshold=0.5, nms_threshold=0.4):
        """Predict solar panels in image"""
        
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
            return [], img_rgb, (0, 0)
        
        original_h, original_w = img_rgb.shape[:2]
        
        # Transform
        augmented = self.transform(image=img_rgb)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        pred_boxes = predictions['pred_boxes'][0]  # [1, 4]
        pred_logits = predictions['pred_logits'][0]  # [1, 2]
        
        # Get probabilities
        probs = F.softmax(pred_logits, dim=-1)[:, 1]  # Probability of being solar panel
        
        # Filter by confidence
        mask = probs > confidence_threshold
        filtered_boxes = pred_boxes[mask]
        filtered_probs = probs[mask]
        
        # Denormalize boxes
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
                    'class': 'solar_panel',
                    'area': (x_max - x_min) * (y_max - y_min)
                })
        
        # NMS
        detections = self._nms(detections, nms_threshold)
        
        return detections, img_rgb, (original_w, original_h)
    
    def _nms(self, detections, threshold=0.4):
        """Non-maximum suppression"""
        if len(detections) == 0:
            return detections
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            remaining = []
            for det in detections:
                iou = self._iou(current['bbox'], det['bbox'])
                if iou < threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _iou(self, box1, box2):
        """Calculate IoU"""
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2
        
        inter_xmin = max(x_min1, x_min2)
        inter_ymin = max(y_min1, y_min2)
        inter_xmax = min(x_max1, x_max2)
        inter_ymax = min(y_max1, y_max2)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
        box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def visualize(self, image_path, detections, output_path=None):
        """Visualize predictions"""
        
        _, img_rgb, _ = self.predict(image_path)
        
        if img_rgb is None:
            return None
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(img_rgb)
        
        # Draw boxes
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"Solar Panel: {conf:.3f}"
            ax.text(x_min, y_min - 5, label, fontsize=10, color='lime',
                   bbox=dict(facecolor='black', alpha=0.7))
        
        ax.set_title(f"{Path(image_path).name} - {len(detections)} panels detected")
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        plt.close()
        
        return fig

# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def main():
    # Setup
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/predictions/visualizations', exist_ok=True)
    os.makedirs('results/predictions/annotations', exist_ok=True)
    os.makedirs('data/clean_dataset/images', exist_ok=True)
    os.makedirs('data/clean_dataset/annotations', exist_ok=True)
    
    # Check model exists
    if not Path('models/best_model.pth').exists():
        print("❌ Model not found! Run training first.")
        exit(1)
    
    # Initialize inference
    print("="*70)
    print("STEP 1: LOADING TRAINED MODEL")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference = SolarPanelInference('models/best_model.pth', device=device)
    
    # Find test images
    print("="*70)
    print("STEP 2: FINDING TEST IMAGES")
    print("="*70 + "\n")
    
    mock_dataset = Path("data/raw/Mock_Dataset/mock-dataset")
    if mock_dataset.exists():
        data_dir = mock_dataset
    else:
        data_dir = Path("data/raw")
    
    images = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.JPG")) +
                   list(data_dir.glob("*.tif")) + list(data_dir.glob("*.TIF")))
    
    # Remove duplicates
    unique_images = {}
    for img in images:
        stem = img.stem
        if stem not in unique_images:
            unique_images[stem] = img
        elif img.suffix.lower() == '.jpg':
            unique_images[stem] = img
    
    images = sorted(list(unique_images.values()))
    print(f"Found {len(images)} test images\n")
    
    # Run inference
    print("="*70)
    print("STEP 3: RUNNING INFERENCE")
    print("="*70 + "\n")
    
    results_csv = Path("results/predictions/predictions.csv")
    
    with open(results_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Detections', 'Confidence_Scores', 'Status'])
        
        total_detections = 0
        
        for img_path in tqdm(images, desc="Processing"):
            try:
                # Predict
                detections, img_rgb, size = inference.predict(img_path, confidence_threshold=0.5)
                total_detections += len(detections)
                
                # Visualize
                vis_path = Path('results/predictions/visualizations') / f"{img_path.stem}_detected.png"
                inference.visualize(img_path, detections, str(vis_path))
                
                # Save annotations
                ann_data = {
                    "image_name": img_path.name,
                    "image_size": list(size),
                    "detections": detections,
                    "total_detections": len(detections),
                    "model": "ViT-Solar-Panel-Detector",
                    "timestamp": datetime.now().isoformat()
                }
                
                ann_path = Path('results/predictions/annotations') / f"{img_path.stem}.json"
                with open(ann_path, 'w') as f:
                    json.dump(ann_data, f, indent=2)
                
                # Add to clean dataset if high confidence
                if len(detections) > 0:
                    avg_confidence = np.mean([d['confidence'] for d in detections])
                    if avg_confidence > 0.6:
                        # Copy image
                        import shutil
                        clean_img_path = Path('data/clean_dataset/images') / img_path.name
                        shutil.copy(img_path, clean_img_path)
                        
                        # Save annotation
                        clean_ann_path = Path('data/clean_dataset/annotations') / f"{img_path.stem}.json"
                        with open(clean_ann_path, 'w') as f:
                            json.dump(ann_data, f, indent=2)
                
                # Log
                conf_scores = '; '.join([f"{d['confidence']:.3f}" for d in detections])
                writer.writerow([img_path.name, len(detections), conf_scores, 'Success'])
                
            except Exception as e:
                writer.writerow([img_path.name, 0, '', f'Error: {str(e)}'])
    
    # Summary
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"\nTotal solar panels detected: {total_detections}")
    print(f"Average per image: {total_detections / len(images):.1f}")
    print(f"\nResults saved to:")
    print(f"  - Predictions CSV: results/predictions/predictions.csv")
    print(f"  - Visualizations: results/predictions/visualizations/")
    print(f"  - Annotations: results/predictions/annotations/")
    print(f"  - Clean dataset: data/clean_dataset/")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()