import shutil
from sklearn.model_selection import train_test_split

class DatasetPreparator:
    """Prepare and split dataset into train/val/test"""
    
    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.split_info = {}
    
    def prepare(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test"""
        print("\n" + "="*70)
        print("DATASET PREPARATION STARTED")
        print("="*70)
        
        # Get all images
        images = sorted(list(self.raw_data_dir.glob('*.jpg')) + 
                       list(self.raw_data_dir.glob('*.tif')))
        
        # Remove duplicates (keep jpg if both jpg and tif exist)
        unique_images = {}
        for img in images:
            stem = img.stem
            if stem not in unique_images:
                unique_images[stem] = img
            elif img.suffix.lower() == '.jpg':
                unique_images[stem] = img
        
        images = list(unique_images.values())
        print(f"Found {len(images)} unique images")
        
        # Split data
        train_images, temp_images = train_test_split(
            images, train_size=train_ratio, random_state=42
        )
        
        val_images, test_images = train_test_split(
            temp_images, 
            train_size=val_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        self.split_info = {
            'train': len(train_images),
            'validation': len(val_images),
            'test': len(test_images)
        }
        
        print(f"\nDataset Split:")
        print(f"  Train: {len(train_images)} images ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(val_images)} images ({val_ratio*100:.0f}%)")
        print(f"  Test: {len(test_images)} images ({test_ratio*100:.0f}%)")
        
        # Copy to processed directories
        self._copy_images('train', train_images)
        self._copy_images('validation', val_images)
        self._copy_images('test', test_images)
        
        # Save split info
        split_file = self.processed_data_dir / 'split_info.json'
        with open(split_file, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        
        print(f"\nData preparation completed!")
        print(f"Processed data saved to: {self.processed_data_dir}")
    
    def _copy_images(self, split_name, images):
        """Copy images and annotations to split directories"""
        img_dir = self.processed_data_dir / split_name / 'images'
        ann_dir = self.processed_data_dir / split_name / 'annotations'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        for idx, img_path in enumerate(images):
            # Rename with sequential number
            new_name = f"{split_name}_{idx:04d}"
            
            # Copy image
            new_img_path = img_dir / f"{new_name}.jpg"
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(new_img_path), img)
            
            # Copy annotation
            ann_path = img_path.with_suffix('.json')
            if ann_path.exists():
                new_ann_path = ann_dir / f"{new_name}.json"
                shutil.copy2(str(ann_path), str(new_ann_path))