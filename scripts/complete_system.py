#!/usr/bin/env python3
# ============================================================================
# COMPLETE END-TO-END SYSTEM
# 1. Auto-detect solar panels
# 2. Generate annotations
# 3. Train Vision Transformer
# File: complete_system.py
# Run: python complete_system.py
# ============================================================================

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split

print("\n" + "#"*70)
print("# COMPLETE END-TO-END SOLAR PANEL DETECTION SYSTEM")
print("#"*70 + "\n")

# ============================================================================
# STEP 1: AUTO DETECT SOLAR PANELS
# ============================================================================

class SolarPanelDetector:
    """Automatically detect solar panels in satellite images"""
    
    def __init__(self, min_area=100, max_area=100000):
        self.min_area = min_area
        self.max_area = max_area
    
    def detect_panels(self, image_path):
        """Detect solar panels in image"""
        
        # Read image
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            try:
                import tifffile
                img = tifffile.imread(str(image_path))
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                else:
                    img_rgb = img[:, :, :3]
            except:
                img = cv2.imread(str(image_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    return []
        else:
            img = cv2.imread(str(image_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return []
        
        height, width = img_rgb.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
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
            
            # Filter by aspect ratio
            aspect_ratio = float(w) / h if h != 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Solidity check
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.4:
                continue
            
            detections.append({
                'x_min': max(0, int(x)),
                'y_min': max(0, int(y)),
                'x_max': min(width, int(x + w)),
                'y_max': min(height, int(y + h)),
                'area': int(area)
            })
        
        # NMS
        detections = self._nms(detections)
        return detections
    
    def _nms(self, detections, threshold=0.4):
        """Non-maximum suppression"""
        if len(detections) == 0:
            return detections
        
        detections = sorted(detections, key=lambda x: x['area'], reverse=True)
        
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
        """Calculate IoU"""
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
# STEP 2: DATASET CLASS
# ============================================================================

class SatelliteDataset(Dataset):
    """Dataset for satellite images"""
    
    def __init__(self, img_dir, ann_dir, transforms=None, img_size=224):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.images = sorted(list(self.img_dir.glob('*.jpg')) + 
                           list(self.img_dir.glob('*.tif')) +
                           list(self.img_dir.glob('*.JPG')) +
                           list(self.img_dir.glob('*.TIF')))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Read image
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            try:
                import tifffile
                img = tifffile.imread(str(img_path))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = img[:,:,:3]
            except:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            return {
                'image': torch.zeros(3, self.img_size, self.img_size),
                'boxes': torch.zeros(0, 4),
                'image_id': 'error'
            }
        
        original_h, original_w = img.shape[:2]
        
        # Load annotations
        ann_path = self.ann_dir / img_path.with_suffix('.json').name
        boxes = []
        
        if ann_path.exists():
            try:
                with open(ann_path, encoding='utf-8') as f:
                    data = json.load(f)
                    for obj in data.get('objects', []):
                        bbox = obj.get('bbox', [])
                        if bbox and len(bbox) == 4:
                            boxes.append([
                                max(0, bbox[0] / original_w),
                                max(0, bbox[1] / original_h),
                                min(1, bbox[2] / original_w),
                                min(1, bbox[3] / original_h)
                            ])
            except:
                pass
        
        # Apply transforms
        if self.transforms and len(boxes) > 0:
            try:
                augmented = self.transforms(image=img, bboxes=boxes)
                img = augmented['image']
                boxes = augmented['bboxes']
            except:
                augmented = self.transforms(image=img, bboxes=[])
                img = augmented['image']
                boxes = []
        elif self.transforms:
            augmented = self.transforms(image=img, bboxes=[])
            img = augmented['image']
            boxes = []
        
        return {
            'image': img,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'image_id': img_path.stem
        }

# ============================================================================
# STEP 3: VISION TRANSFORMER MODEL
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
# STEP 4: TRAINER
# ============================================================================

class Trainer:
    """Training pipeline"""
    
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def compute_loss(self, predictions, targets):
        """Compute loss"""
        pred_boxes = predictions['pred_boxes']
        pred_logits = predictions['pred_logits']
        
        batch_size = pred_logits.shape[0]
        
        # Classification loss
        target_classes = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                target_classes[i, 0] = 1
        
        class_loss = nn.CrossEntropyLoss()(
            pred_logits.view(-1, 2),
            target_classes.view(-1)
        )
        
        # BBox loss
        bbox_loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                target_boxes = target['boxes'].to(self.device)
                bbox_loss += nn.L1Loss()(pred_boxes[i, 0], target_boxes[0])
                count += 1
        
        if count > 0:
            bbox_loss = bbox_loss / count
        
        total_loss = class_loss + 0.5 * bbox_loss
        return total_loss
    
    def train_epoch(self, epoch):
        """Train epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = [{'boxes': batch['boxes'][i]} for i in range(len(batch['boxes']))]
            
            predictions = self.model(images)
            loss = self.compute_loss(predictions, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = [{'boxes': batch['boxes'][i]} for i in range(len(batch['boxes']))]
                
                predictions = self.model(images)
                loss = self.compute_loss(predictions, targets)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        self.history['val_loss'].append(avg_loss)
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(self.model.state_dict(), 'models/best_model.pth')
            print(f"  ✓ Best model saved (loss: {avg_loss:.4f})")
        
        return avg_loss
    
    def train(self, num_epochs=30):
        """Train model"""
        print("\n" + "="*70)
        print("STEP 4: TRAINING VISION TRANSFORMER")
        print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}\n")
        
        print("✅ Training completed!")
        self.plot_curves()
    
    def plot_curves(self):
        """Plot curves"""
        if len(self.history['train_loss']) == 0:
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/train_output/training_curves.png', dpi=150)
        print("✓ Training curves saved\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Setup directories
    print("Setting up directories...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/train_output', exist_ok=True)
    os.makedirs('data/raw/annotations', exist_ok=True)
    os.makedirs('data/processed/train/images', exist_ok=True)
    os.makedirs('data/processed/train/annotations', exist_ok=True)
    os.makedirs('data/processed/val/images', exist_ok=True)
    os.makedirs('data/processed/val/annotations', exist_ok=True)
    print("✓ Directories ready\n")
    
    # STEP 1: Find images
    print("="*70)
    print("STEP 1: FINDING SATELLITE IMAGES")
    print("="*70 + "\n")
    
    raw_dir = Path("data/raw")
    mock_dataset = raw_dir / "Mock_Dataset" / "mock-dataset"
    
    if mock_dataset.exists():
        data_dir = mock_dataset
    else:
        data_dir = raw_dir
    
    all_images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.JPG")) + \
                 list(data_dir.glob("*.tif")) + list(data_dir.glob("*.TIF"))
    
    # Remove duplicates
    unique_images = {}
    for img in all_images:
        stem = img.stem
        if stem not in unique_images:
            unique_images[stem] = img
        elif img.suffix.lower() == '.jpg':
            unique_images[stem] = img
    
    images = sorted(list(unique_images.values()))
    print(f"Found {len(images)} satellite images\n")
    
    if len(images) == 0:
        print("❌ No images found!")
        exit(1)
    
    # STEP 2: Detect solar panels
    print("="*70)
    print("STEP 2: AUTO-DETECTING SOLAR PANELS")
    print("="*70 + "\n")
    
    detector = SolarPanelDetector(min_area=100, max_area=100000)
    
    annotations_dir = Path("data/raw/annotations")
    total_panels = 0
    
    for img_path in tqdm(images, desc="Detecting panels"):
        try:
            detections = detector.detect_panels(img_path)
            total_panels += len(detections)
            
            # Save annotation
            annotation = {
                "image_name": img_path.name,
                "objects": [
                    {
                        "class": "solar_panel",
                        "bbox": [d['x_min'], d['y_min'], d['x_max'], d['y_max']]
                    }
                    for d in detections
                ]
            }
            
            json_path = annotations_dir / img_path.with_suffix('.json').name
            with open(json_path, 'w') as f:
                json.dump(annotation, f, indent=2)
        except:
            pass
    
    print(f"\n✓ Detected {total_panels} solar panels across {len(images)} images")
    print(f"✓ Average: {total_panels / len(images):.1f} panels per image\n")
    
    # STEP 3: Prepare dataset
    print("="*70)
    print("STEP 3: PREPARING TRAINING DATASET")
    print("="*70 + "\n")
    
    train_images, val_images = train_test_split(images, train_size=0.8, random_state=42)
    
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}\n")
    
    # Copy training data
    for idx, img_path in enumerate(tqdm(train_images, desc="Copying train")):
        img_out = Path('data/processed/train/images') / f"train_{idx:04d}.tif"
        
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            try:
                import tifffile
                img = tifffile.imread(str(img_path))
                tifffile.imwrite(str(img_out), img)
            except:
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(img_out), img)
        else:
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(img_out), img)
        
        ann_in = annotations_dir / img_path.with_suffix('.json').name
        if ann_in.exists():
            ann_out = Path('data/processed/train/annotations') / f"train_{idx:04d}.json"
            shutil.copy(ann_in, ann_out)
    
    # Copy validation data
    for idx, img_path in enumerate(tqdm(val_images, desc="Copying val")):
        img_out = Path('data/processed/val/images') / f"val_{idx:04d}.tif"
        
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            try:
                import tifffile
                img = tifffile.imread(str(img_path))
                tifffile.imwrite(str(img_out), img)
            except:
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(img_out), img)
        else:
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(img_out), img)
        
        ann_in = annotations_dir / img_path.with_suffix('.json').name
        if ann_in.exists():
            ann_out = Path('data/processed/val/annotations') / f"val_{idx:04d}.json"
            shutil.copy(ann_in, ann_out)
    
    print("✓ Dataset preparation complete\n")
    
    # STEP 3.5: Create dataloaders
    print("="*70)
    print("STEP 3.5: CREATING DATA LOADERS")
    print("="*70 + "\n")
    
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2))
    
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2))
    
    train_dataset = SatelliteDataset('data/processed/train/images',
                                     'data/processed/train/annotations',
                                     transforms=train_transforms)
    val_dataset = SatelliteDataset('data/processed/val/images',
                                   'data/processed/val/annotations',
                                   transforms=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"✓ Train dataset: {len(train_dataset)} images")
    print(f"✓ Validation dataset: {len(val_dataset)} images\n")
    
    # STEP 3.75: Initialize model
    print("="*70)
    print("STEP 3.75: INITIALIZING MODEL")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = ViTDetector('vit_small_patch16_224', num_queries=10).to(device)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # STEP 4: Train
    trainer = Trainer(model, train_loader, val_loader, device, lr=1e-4)
    trainer.train(num_epochs=30)
    
    # Summary
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*70)
    print(f"\nResults:")
    print(f"  ✓ Trained model: models/best_model.pth")
    print(f"  ✓ Annotations: data/raw/annotations/")
    print(f"  ✓ Training curves: results/train_output/training_curves.png")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()