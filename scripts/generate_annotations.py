#!/usr/bin/env python3
# ============================================================================
# AUTO SOLAR PANEL DETECTION AND LABELED DATASET GENERATION
# Detects solar panels from satellite images and creates JSON annotations
# File: scripts/01_generate_annotations.py
# Run: python scripts/01_generate_annotations.py
# ============================================================================

import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("\n" + "#"*70)
print("# AUTO SOLAR PANEL DETECTION - GENERATING LABELED DATASET")
print("#"*70 + "\n")

# ============================================================================
# SOLAR PANEL DETECTOR USING IMAGE PROCESSING
# ============================================================================

class SolarPanelDetector:
    """Automatically detect solar panels in satellite images"""
    
    def __init__(self, min_area=50, max_area=50000):
        self.min_area = min_area
        self.max_area = max_area
        self.detections = []
    
    def detect_panels(self, image_path):
        """Detect solar panels in image"""
        
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
            return []
        
        height, width = img_rgb.shape[:2]
        
        # Convert to HSV for better color detection
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Create masks for blue-ish colors (typical solar panels)
        # Lower blue range
        lower_blue1 = np.array([90, 50, 50])
        upper_blue1 = np.array([130, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
        
        # Grayscale approach - solar panels are typically darker
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Threshold to find dark regions (solar panels)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_blue, thresh)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (solar panels are roughly rectangular)
            aspect_ratio = float(w) / h if h != 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Check if region has enough variation (not just empty space)
            roi = gray[y:y+h, x:x+w]
            if roi.std() < 5:
                continue
            
            detections.append({
                'x_min': max(0, int(x)),
                'y_min': max(0, int(y)),
                'x_max': min(width, int(x + w)),
                'y_max': min(height, int(y + h)),
                'confidence': 0.75,  # Default confidence
                'area': int(area)
            })
        
        # Remove overlapping detections (keep larger ones)
        detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections, threshold=0.5):
        """Non-maximum suppression to remove overlapping boxes"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            remaining = []
            for det in detections:
                iou = self._iou(current, det)
                if iou < threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x_min1, y_min1 = box1['x_min'], box1['y_min']
        x_max1, y_max1 = box1['x_max'], box1['y_max']
        x_min2, y_min2 = box2['x_min'], box2['y_min']
        x_max2, y_max2 = box2['x_max'], box2['y_max']
        
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

# ============================================================================
# MAIN PROCESSING
# ============================================================================

# Find dataset
raw_dir = Path("data/raw")
dataset_dirs = []

# Check for nested structure
mock_dataset = raw_dir / "Mock_Dataset" / "mock-dataset"
if mock_dataset.exists():
    dataset_dirs.append(mock_dataset)

if not dataset_dirs:
    mock_dataset = raw_dir / "Mock_Dataset"
    if mock_dataset.exists():
        dataset_dirs.append(mock_dataset)

if not dataset_dirs:
    dataset_dirs = [raw_dir]

print(f"Searching in: {dataset_dirs}\n")

# Initialize detector
detector = SolarPanelDetector(min_area=50, max_area=100000)

# Find all images
all_images = []
for data_dir in dataset_dirs:
    all_images.extend(list(data_dir.glob("*.jpg")))
    all_images.extend(list(data_dir.glob("*.JPG")))
    all_images.extend(list(data_dir.glob("*.tif")))
    all_images.extend(list(data_dir.glob("*.TIF")))

# Remove duplicates
unique_images = {}
for img in all_images:
    stem = img.stem
    if stem not in unique_images:
        unique_images[stem] = img
    elif img.suffix.lower() == '.jpg':
        unique_images[stem] = img

images = sorted(list(unique_images.values()))

print(f"Found {len(images)} images\n")

if len(images) == 0:
    print("❌ No images found!")
    exit(1)

# Create output directory
output_dir = Path("data/raw/annotations")
output_dir.mkdir(parents=True, exist_ok=True)

vis_dir = Path("data/raw/detections_preview")
vis_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("DETECTING SOLAR PANELS IN SATELLITE IMAGES")
print("="*70 + "\n")

total_panels = 0
processed = 0

for img_path in tqdm(images, desc="Processing images"):
    try:
        # Detect panels
        detections = detector.detect_panels(img_path)
        total_panels += len(detections)
        
        # Create JSON annotation
        annotation = {
            "image_name": img_path.name,
            "image_path": str(img_path),
            "objects": [
                {
                    "class": "solar_panel",
                    "bbox": [
                        d['x_min'], 
                        d['y_min'], 
                        d['x_max'], 
                        d['y_max']
                    ],
                    "confidence": d['confidence'],
                    "area": d['area']
                }
                for d in detections
            ],
            "detection_method": "image_processing",
            "total_detections": len(detections)
        }
        
        # Save JSON
        json_path = output_dir / img_path.with_suffix('.json').name
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        processed += 1
        
        # Create visualization (every 5th image to save time)
        if processed % 5 == 0:
            try:
                if str(img_path).lower().endswith(('.tif', '.tiff')):
                    import tifffile
                    img_data = tifffile.imread(str(img_path))
                    if len(img_data.shape) == 2:
                        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = img_data[:, :, :3]
                else:
                    img_data = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                
                # Draw detections
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(img_rgb)
                
                for det in detections:
                    rect = patches.Rectangle(
                        (det['x_min'], det['y_min']),
                        det['x_max'] - det['x_min'],
                        det['y_max'] - det['y_min'],
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                
                ax.set_title(f"{img_path.name} - {len(detections)} panels detected")
                ax.axis('off')
                
                vis_path = vis_dir / img_path.with_suffix('.png').name
                plt.savefig(vis_path, dpi=100, bbox_inches='tight')
                plt.close()
            except:
                pass
    
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

print("\n" + "="*70)
print("DETECTION COMPLETE")
print("="*70)
print(f"\nProcessed: {processed} images")
print(f"Total solar panels detected: {total_panels}")
print(f"Average panels per image: {total_panels / processed:.1f}")
print(f"\nAnnotations saved to: {output_dir}")
print(f"Preview visualizations: {vis_dir}")

# Summary
print("\n" + "="*70)
print("DATASET READY FOR TRAINING")
print("="*70)
print(f"\nNext step:")
print(f"  python scripts/complete_training_pipeline.py")
print("\n" + "="*70 + "\n")