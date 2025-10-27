#!/usr/bin/env python3
# ============================================================================
# COMPLETE TRAINING PIPELINE - CORRECTED FOR YOUR MOCK DATASET
# Uses Vision Transformer with correct image size (224x224)
# ============================================================================

import os
import json
import cv2
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
print("# COMPLETE TRAINING PIPELINE - YOUR GeoJSON SATELLITE DATA")
print("#"*70 + "\n")

# ============================================================================
# STEP 0: Create necessary directories
# ============================================================================

print("Creating directories...")
os.makedirs('models', exist_ok=True)
os.makedirs('results/train_output', exist_ok=True)
os.makedirs('data/processed/train/images', exist_ok=True)
os.makedirs('data/processed/train/annotations', exist_ok=True)
os.makedirs('data/processed/val/images', exist_ok=True)
os.makedirs('data/processed/val/annotations', exist_ok=True)
os.makedirs('data/raw_converted', exist_ok=True)
print("✓ Directories created\n")

# ============================================================================
# STEP 1: GeoJSON to BBox Converter
# ============================================================================

class GeoJSONToBBoxConverter:
    """Convert your GeoJSON format to bounding boxes"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converted = 0
        self.total_objects = 0
    
    def extract_bbox_from_polygon(self, polygon_coords):
        """Extract bbox from polygon coordinates"""
        try:
            if not polygon_coords or len(polygon_coords) < 2:
                return None
            
            x_coords = [coord[0] for coord in polygon_coords]
            y_coords = [coord[1] for coord in polygon_coords]
            
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            
            if x_max <= x_min or y_max <= y_min:
                return None
            
            return [int(x_min), int(y_min), int(x_max), int(y_max)]
        except:
            return None
    
    def convert_file(self, geojson_path):
        """Convert single GeoJSON file"""
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            objects = []
            features = geojson_data.get('features', [])
            
            for feature in features:
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                
                class_name = properties.get('Class Name', 'unknown')
                geom_type = geometry.get('type')
                coordinates = geometry.get('coordinates', [])
                
                # Handle MultiPolygon (your format)
                if geom_type == 'MultiPolygon':
                    for polygon in coordinates:
                        for ring in polygon:
                            bbox = self.extract_bbox_from_polygon(ring)
                            if bbox:
                                objects.append({
                                    'class': class_name.lower().replace(' ', '_'),
                                    'bbox': bbox
                                })
                
                # Handle Polygon
                elif geom_type == 'Polygon':
                    for ring in coordinates:
                        bbox = self.extract_bbox_from_polygon(ring)
                        if bbox:
                            objects.append({
                                'class': class_name.lower().replace(' ', '_'),
                                'bbox': bbox
                            })
            
            if objects:
                output_data = {
                    'image_name': geojson_path.stem + '.tif',
                    'objects': objects
                }
                return output_data
            return None
            
        except Exception as e:
            print(f"Error converting {geojson_path.name}: {e}")
            return None
    
    def convert_all(self):
        """Convert all GeoJSON files"""
        print("\n" + "="*70)
        print("STEP 1: CONVERTING YOUR GeoJSON FILES")
        print("="*70)
        
        geojson_files = list(self.input_dir.glob('*.json'))
        
        if not geojson_files:
            print("⚠ No JSON files found")
            return 0
        
        print(f"Found {len(geojson_files)} GeoJSON files\n")
        
        for geojson_path in tqdm(geojson_files, desc="Converting"):
            output_data = self.convert_file(geojson_path)
            if output_data:
                output_path = self.output_dir / geojson_path.name
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                self.converted += 1
                self.total_objects += len(output_data['objects'])
        
        print(f"\n✓ Converted {self.converted} GeoJSON files")
        print(f"✓ Extracted {self.total_objects} total objects (solar panels)")
        return self.converted

# ============================================================================
# STEP 2: Dataset Class - CORRECTED FOR CORRECT IMAGE SIZE
# ============================================================================

class SatelliteDataset(Dataset):
    """Dataset for satellite images - uses 224x224 for ViT"""
    
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
        
        # Read image (handle both JPG and TIF)
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            # TIF files
            import tifffile
            try:
                img = tifffile.imread(str(img_path))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif len(img.shape) == 3:
                    img = img[:,:,:3]
            except:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # JPG files
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            return {
                'image': torch.zeros(3, self.img_size, self.img_size),
                'boxes': torch.zeros(0, 4),
                'image_id': 'error'
            }
        
        original_h, original_w = img.shape[:2]
        
        # Read annotations
        ann_path = self.ann_dir / img_path.with_suffix('.json').name
        boxes = []
        
        if ann_path.exists():
            try:
                with open(ann_path, encoding='utf-8') as f:
                    data = json.load(f)
                    for obj in data.get('objects', []):
                        bbox = obj.get('bbox', [])
                        if bbox and len(bbox) == 4:
                            # Normalize to [0, 1]
                            boxes.append([
                                max(0, bbox[0] / original_w),
                                max(0, bbox[1] / original_h),
                                min(1, bbox[2] / original_w),
                                min(1, bbox[3] / original_h)
                            ])
            except:
                pass
        
        # Apply augmentation
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
# STEP 3: Vision Transformer Model - CORRECTED
# ============================================================================

class ViTDetector(nn.Module):
    """Vision Transformer for object detection"""
    
    def __init__(self, model_name='vit_small_patch16_224', num_queries=10):
        super().__init__()
        
        # Load pretrained ViT (224x224 input)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.embed_dim
        
        self.num_queries = num_queries
        
        # Detection heads
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
        # x shape: [batch_size, 3, 224, 224]
        features = self.backbone.forward_features(x)
        
        # Get class token (first token)
        if len(features.shape) == 3:
            # Shape: [batch_size, num_patches + 1, feat_dim]
            class_token = features[:, 0, :]  # [batch_size, feat_dim]
        else:
            features = features.mean(dim=(2, 3))
            class_token = features
        
        batch_size = class_token.size(0)
        
        # Generate predictions
        pred_boxes = self.bbox_head(class_token)  # [batch_size, 4]
        pred_logits = self.class_head(class_token)  # [batch_size, 2]
        
        return {
            'pred_boxes': pred_boxes.unsqueeze(1),  # [batch_size, 1, 4]
            'pred_logits': pred_logits.unsqueeze(1)  # [batch_size, 1, 2]
        }

# ============================================================================
# STEP 4: Trainer Class
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
        pred_boxes = predictions['pred_boxes']  # [batch_size, 1, 4]
        pred_logits = predictions['pred_logits']  # [batch_size, 1, 2]
        
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
        """Train one epoch"""
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
        print("STEP 5: TRAINING VISION TRANSFORMER MODEL")
        print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n")
        
        print("✅ Training completed!")
        self.plot_curves()
    
    def plot_curves(self):
        """Plot training curves"""
        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
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
        print("✓ Training curves saved to: results/train_output/training_curves.png\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Find the data directory (could be in subfolder)
    data_raw_path = Path('data/raw')
    
    # Check if files are in subfolder
    subfolders = [f for f in data_raw_path.glob('*') if f.is_dir()]
    json_files_root = list(data_raw_path.glob('*.json'))
    tif_files_root = list(data_raw_path.glob('*.tif'))
    
    if (not json_files_root and not tif_files_root) and subfolders:
        # Files are in a subfolder
        print(f"Found subfolder(s): {[f.name for f in subfolders]}")
        data_raw_path = subfolders[0]
        print(f"Using: {data_raw_path}\n")
    
    # STEP 1: Convert GeoJSON
    print("="*70)
    print("STEP 1: CONVERTING YOUR GeoJSON FILES")
    print("="*70)
    
    converter = GeoJSONToBBoxConverter(str(data_raw_path), 'data/raw_converted')
    converted_count = converter.convert_all()
    
    if converted_count == 0:
        print("\n⚠ No GeoJSON files converted!")
        print("Make sure your files are in: data/raw/ or data/raw/<subfolder>/")
        return
    
    # Copy converted files back
    print("\nCopying converted files...")
    for json_file in Path('data/raw_converted').glob('*.json'):
        shutil.copy(json_file, data_raw_path / json_file.name)
    print("✓ Conversion complete\n")
    
    # STEP 2: Prepare dataset
    print("="*70)
    print("STEP 2: PREPARING DATASET")
    print("="*70)
    
    images = sorted(list(data_raw_path.glob('*.jpg')) + 
                   list(data_raw_path.glob('*.tif')) +
                   list(data_raw_path.glob('*.JPG')) +
                   list(data_raw_path.glob('*.TIF')))
    
    # Remove duplicates
    unique_images = {}
    for img in images:
        stem = img.stem
        if stem not in unique_images:
            unique_images[stem] = img
        elif img.suffix.lower() == '.jpg':
            unique_images[stem] = img
    
    images = list(unique_images.values())
    
    if len(images) < 2:
        print(f"\n⚠ ERROR: Only {len(images)} image(s) found!")
        print("You need at least 2 images for training")
        return
    
    if len(images) < 10:
        print(f"\n⚠ WARNING: Only {len(images)} images found!")
        print("Recommended: at least 50 images for good results\n")
    
    # Split data
    if len(images) > 2:
        train_images, val_images = train_test_split(images, train_size=0.8, random_state=42)
    else:
        train_images = images[:1]
        val_images = images[1:]
    
    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}\n")
    
    # Copy training images
    print("Copying training images...")
    for idx, img_path in enumerate(tqdm(train_images)):
        img_out = Path('data/processed/train/images') / f"train_{idx:04d}.tif"
        
        # Handle both JPG and TIF
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            import tifffile
            try:
                img = tifffile.imread(str(img_path))
                tifffile.imwrite(str(img_out), img)
            except:
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(img_out), img)
        else:
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(img_out), img)
        
        ann_in = img_path.with_suffix('.json')
        if ann_in.exists():
            ann_out = Path('data/processed/train/annotations') / f"train_{idx:04d}.json"
            shutil.copy(ann_in, ann_out)
    
    # Copy validation images
    print("Copying validation images...")
    for idx, img_path in enumerate(tqdm(val_images)):
        img_out = Path('data/processed/val/images') / f"val_{idx:04d}.tif"
        
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            import tifffile
            try:
                img = tifffile.imread(str(img_path))
                tifffile.imwrite(str(img_out), img)
            except:
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(img_out), img)
        else:
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(img_out), img)
        
        ann_in = img_path.with_suffix('.json')
        if ann_in.exists():
            ann_out = Path('data/processed/val/annotations') / f"val_{idx:04d}.json"
            shutil.copy(ann_in, ann_out)
    
    print("✓ Data preparation complete\n")
    
    # STEP 3: Create dataloaders - CORRECTED SIZE
    print("="*70)
    print("STEP 3: CREATING DATA LOADERS")
    print("="*70)
    
    train_transforms = A.Compose([
        A.Resize(224, 224),  # CORRECTED: ViT uses 224x224
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2))
    
    val_transforms = A.Compose([
        A.Resize(224, 224),  # CORRECTED: ViT uses 224x224
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2))
    
    train_dataset = SatelliteDataset('data/processed/train/images', 
                                     'data/processed/train/annotations',
                                     transforms=train_transforms,
                                     img_size=224)
    val_dataset = SatelliteDataset('data/processed/val/images',
                                   'data/processed/val/annotations',
                                   transforms=val_transforms,
                                   img_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    print(f"✓ Train dataset: {len(train_dataset)} images")
    print(f"✓ Validation dataset: {len(val_dataset)} images\n")
    
    # STEP 4: Initialize model
    print("="*70)
    print("STEP 4: INITIALIZING VISION TRANSFORMER MODEL")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ViTDetector('vit_small_patch16_224', num_queries=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # STEP 5: Train
    trainer = Trainer(model, train_loader, val_loader, device, lr=1e-4)
    trainer.train(num_epochs=30)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print("\nYour trained model:")
    print(f"  Location: models/best_model.pth")
    print(f"  Size: ~88 MB")
    print("\nTraining visualization:")
    print(f"  Location: results/train_output/training_curves.png")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()