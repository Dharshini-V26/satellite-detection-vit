import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

class DatasetExplorer:
    """Explore and analyze satellite imagery dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.stats = {
            'total_images': 0,
            'image_formats': defaultdict(int),
            'image_sizes': [],
            'annotations_count': 0,
            'object_classes': defaultdict(int),
            'bbox_sizes': [],
            'images_per_annotation': defaultdict(int)
        }
        self.images_info = []
    
    def explore(self):
        """Analyze dataset structure"""
        print("\n" + "="*70)
        print("DATASET EXPLORATION STARTED")
        print("="*70)
        
        images = list(self.dataset_path.glob('*.jpg')) + list(self.dataset_path.glob('*.tif'))
        self.stats['total_images'] = len(images)
        
        for img_path in images:
            self.stats['image_formats'][img_path.suffix] += 1
            
            # Read image
            try:
                if img_path.suffix.lower() in ['.tif', '.tiff']:
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if len(img.shape) == 3 and img.shape[2] > 3:
                        img = img[:, :, :3]
                else:
                    img = cv2.imread(str(img_path))
                
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                self.stats['image_sizes'].append((w, h))
                
                # Check for annotation
                json_path = img_path.with_suffix('.json')
                annotation_count = 0
                
                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)
                        objects = data.get('objects', [])
                        annotation_count = len(objects)
                        self.stats['annotations_count'] += 1
                        
                        for obj in objects:
                            class_name = obj.get('class', 'unknown')
                            self.stats['object_classes'][class_name] += 1
                            
                            bbox = obj.get('bbox', [])
                            if len(bbox) == 4:
                                bbox_w = bbox[2] - bbox[0]
                                bbox_h = bbox[3] - bbox[1]
                                self.stats['bbox_sizes'].append((bbox_w, bbox_h))
                
                self.stats['images_per_annotation'][annotation_count] += 1
                self.images_info.append({
                    'name': img_path.name,
                    'width': w,
                    'height': h,
                    'format': img_path.suffix,
                    'annotations': annotation_count
                })
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        return self.stats
    
    def print_report(self):
        """Print detailed statistics"""
        print("\n" + "="*70)
        print("DATASET STATISTICS REPORT")
        print("="*70)
        
        print(f"\nBasic Statistics:")
        print(f"  Total Images: {self.stats['total_images']}")
        print(f"  Image Formats: {dict(self.stats['image_formats'])}")
        print(f"  Annotated Images: {self.stats['annotations_count']}")
        
        if self.stats['image_sizes']:
            widths = [w for w, h in self.stats['image_sizes']]
            heights = [h for w, h in self.stats['image_sizes']]
            print(f"\nImage Dimensions:")
            print(f"  Width Range: {min(widths)} - {max(widths)} px")
            print(f"  Height Range: {min(heights)} - {max(heights)} px")
            print(f"  Average Size: {np.mean(widths):.0f} x {np.mean(heights):.0f} px")
        
        print(f"\nObject Classes:")
        for class_name, count in self.stats['object_classes'].items():
            print(f"  {class_name}: {count} objects")
        
        print(f"\nAnnotation Distribution:")
        for ann_count in sorted(self.stats['images_per_annotation'].keys()):
            img_count = self.stats['images_per_annotation'][ann_count]
            print(f"  {ann_count} objects: {img_count} images")
        
        if self.stats['bbox_sizes']:
            bbox_widths = [w for w, h in self.stats['bbox_sizes']]
            bbox_heights = [h for w, h in self.stats['bbox_sizes']]
            print(f"\nBounding Box Statistics:")
            print(f"  Average Width: {np.mean(bbox_widths):.0f} px")
            print(f"  Average Height: {np.mean(bbox_heights):.0f} px")
        
        print("\n" + "="*70)
    
    def visualize_samples(self, output_dir, num_samples=5):
        """Visualize sample images with annotations"""
        os.makedirs(output_dir, exist_ok=True)
        
        sample_images = np.random.choice(self.images_info, min(num_samples, len(self.images_info)), replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
        
        for idx, img_info in enumerate(sample_images):
            img_path = self.dataset_path / img_info['name']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw annotations
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                    for obj in data.get('objects', []):
                        bbox = obj.get('bbox', [])
                        if len(bbox) == 4:
                            x_min, y_min, x_max, y_max = bbox
                            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_info['name']}\n({img_info['annotations']} objects)")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to {output_dir}")