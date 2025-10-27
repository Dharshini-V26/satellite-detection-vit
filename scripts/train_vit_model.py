import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# DATASET CLASS
# ============================================================================

class SatelliteDetectionDataset(Dataset):
    """Dataset for satellite object detection"""
    
    def __init__(self, img_dir, ann_dir, transforms=None, img_size=512):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.images = sorted(list(self.img_dir.glob('*.jpg')))
        self.class_to_idx = {'solar_panel': 1, 'background': 0}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        # Load annotations
        ann_path = self.ann_dir / img_path.with_suffix('.json').name
        boxes = []
        
        if ann_path.exists():
            with open(ann_path) as f:
                data = json.load(f)
                for obj in data.get('objects', []):
                    bbox = obj.get('bbox', [])
                    if bbox and len(bbox) == 4:
                        # Normalize to [0, 1]
                        boxes.append([
                            bbox[0] / original_w, bbox[1] / original_h,
                            bbox[2] / original_w, bbox[3] / original_h
                        ])
        
        if self.transforms:
            augmented = self.transforms(image=img, bboxes=boxes)
            img = augmented['image']
            boxes = augmented['bboxes']
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return {
            'image': img,
            'boxes': boxes,
            'image_id': img_path.stem
        }

# ============================================================================
# VISION TRANSFORMER MODEL
# ============================================================================

class ViTObjectDetector(nn.Module):
    """Vision Transformer for object detection"""
    
    def __init__(self, model_name='vit_small_patch16_224', num_queries=100, num_classes=2):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Load pretrained ViT
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.embed_dim
        
        # Detection heads
        self.query_embeddings = nn.Embedding(num_queries, feat_dim)
        
        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)
        )
        
        # Class prediction head
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, feat_dim, H, W]
        
        # Global average pooling
        if len(features.shape) == 4:
            features = features.mean(dim=(2, 3))  # [B, feat_dim]
        
        batch_size = features.size(0)
        
        # Generate predictions for each query
        pred_boxes = self.bbox_head(features.unsqueeze(1).expand(-1, self.num_queries, -1))
        pred_logits = self.class_head(features.unsqueeze(1).expand(-1, self.num_queries, -1))
        
        return {
            'pred_boxes': pred_boxes,      # [B, num_queries, 4]
            'pred_logits': pred_logits     # [B, num_queries, num_classes]
        }

# ============================================================================
# TRAINING LOOP
# ============================================================================

class DetectionTrainer:
    """Training pipeline for object detection"""
    
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'train_cls': [], 'train_bbox': []}
        self.log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        os.makedirs('logs', exist_ok=True)
    
    def compute_loss(self, predictions, targets):
        """Compute detection loss"""
        pred_boxes = predictions['pred_boxes']  # [B, num_queries, 4]
        pred_logits = predictions['pred_logits']  # [B, num_queries, num_classes]
        
        batch_size = pred_logits.shape[0]
        
        # Classification loss
        target_classes = torch.zeros(
            batch_size, self.model.num_queries,
            dtype=torch.long, device=self.device
        )
        
        # Mark positive queries (first detected object)
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                target_classes[i, 0] = 1
        
        class_loss = nn.CrossEntropyLoss()(
            pred_logits.view(-1, self.model.num_classes),
            target_classes.view(-1)
        )
        
        # Bounding box regression loss
        bbox_loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                target_boxes = target['boxes'].to(self.device)
                bbox_loss += nn.L1Loss()(pred_boxes[i, 0], target_boxes[0])
                count += 1
        
        if count > 0:
            bbox_loss = bbox_loss / count
        else:
            bbox_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = class_loss + 0.5 * bbox_loss
        
        return total_loss, class_loss, bbox_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = [{'boxes': batch['boxes'][i]} for i in range(len(batch['boxes']))]
            
            # Forward pass
            predictions = self.model(images)
            loss, cls_loss, bbox_loss = self.compute_loss(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'bbox': f'{bbox_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_bbox_loss = total_bbox_loss / len(self.train_loader)
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_cls'].append(avg_cls_loss)
        self.history['train_bbox'].append(avg_bbox_loss)
        
        return avg_loss, avg_cls_loss, avg_bbox_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = [{'boxes': batch['boxes'][i]} for i in range(len(batch['boxes']))]
                
                predictions = self.model(images)
                loss, _, _ = self.compute_loss(predictions, targets)
                total_loss += loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs=30):
        """Train model"""
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        
        for epoch in range(num_epochs):
            train_loss, train_cls, train_bbox = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls:.4f}, BBox: {train_bbox:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_model.pth')
                print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), f'models/checkpoints/checkpoint_epoch_{epoch+1:03d}.pth')
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        self.plot_history()
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid()
        
        axes[1].plot(self.history['train_cls'], label='Class Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Classification Loss')
        axes[1].legend()
        axes[1].grid()
        
        axes[2].plot(self.history['train_bbox'], label='BBox Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('BBox Regression Loss')
        axes[2].legend()
        axes[2].grid()
        
        plt.tight_layout()
        plt.savefig('results/train_output/training_curves.png', dpi=150)
        print("Training curves saved to results/train_output/training_curves.png")
