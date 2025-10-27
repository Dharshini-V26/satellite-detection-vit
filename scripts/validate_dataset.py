class DatasetValidator:
    """Validate dataset consistency and quality"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.issues = {
            'missing_images': [],
            'missing_annotations': [],
            'invalid_bboxes': [],
            'empty_annotations': [],
            'corrupted_images': []
        }
    
    def validate(self):
        """Validate all data"""
        print("\n" + "="*70)
        print("DATASET VALIDATION STARTED")
        print("="*70)
        
        for split in ['train', 'validation', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
            
            img_dir = split_dir / 'images'
            ann_dir = split_dir / 'annotations'
            
            images = list(img_dir.glob('*.jpg'))
            print(f"\nValidating {split} set ({len(images)} images)...")
            
            for img_path in images:
                # Check if image is readable
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        self.issues['corrupted_images'].append(img_path.name)
                        continue
                except:
                    self.issues['corrupted_images'].append(img_path.name)
                    continue
                
                h, w = img.shape[:2]
                
                # Check annotation
                ann_path = ann_dir / img_path.with_suffix('.json').name
                if not ann_path.exists():
                    self.issues['missing_annotations'].append(img_path.name)
                    continue
                
                # Validate annotation content
                try:
                    with open(ann_path) as f:
                        data = json.load(f)
                        objects = data.get('objects', [])
                        
                        if len(objects) == 0:
                            self.issues['empty_annotations'].append(img_path.name)
                        
                        for obj in objects:
                            bbox = obj.get('bbox', [])
                            if len(bbox) != 4:
                                self.issues['invalid_bboxes'].append(f"{img_path.name}: {bbox}")
                            else:
                                x_min, y_min, x_max, y_max = bbox
                                if not (0 <= x_min < x_max <= w and 0 <= y_min < y_max <= h):
                                    self.issues['invalid_bboxes'].append(
                                        f"{img_path.name}: bbox {bbox} out of bounds ({w}x{h})"
                                    )
                except Exception as e:
                    self.issues['missing_annotations'].append(img_path.name)
        
        self._print_report()
    
    def _print_report(self):
        """Print validation report"""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        total_issues = sum(len(v) for v in self.issues.values())
        
        if total_issues == 0:
            print("\n✓ All validations passed! Dataset is clean.")
        else:
            print(f"\n✗ Found {total_issues} issues:")
            
            for issue_type, items in self.issues.items():
                if items:
                    print(f"\n  {issue_type.replace('_', ' ').title()}: {len(items)}")
                    for item in items[:5]:
                        print(f"    - {item}")
                    if len(items) > 5:
                        print(f"    ... and {len(items) - 5} more")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Step 1: Explore dataset
    print("\n" + "#"*70)
    print("# STEP 1: EXPLORE DATASET")
    print("#"*70)
    
    explorer = DatasetExplorer('data/raw')
    explorer.explore()
    explorer.print_report()
    explorer.visualize_samples('data/exploratory', num_samples=5)
    
    # Step 2: Prepare dataset
    print("\n" + "#"*70)
    print("# STEP 2: PREPARE DATASET")
    print("#"*70)
    
    preparator = DatasetPreparator('data/raw', 'data/processed')
    preparator.prepare(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Step 3: Validate dataset
    print("\n" + "#"*70)
    print("# STEP 3: VALIDATE DATASET")
    print("#"*70)
    
    validator = DatasetValidator('data/processed')
    validator.validate()
    
    print("\n" + "="*70)
    print("DATA PREPARATION PIPELINE COMPLETED!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Run 04_train_vit_model.py to start training")
    print("2. Monitor training progress in logs/")
    print("3. Evaluate model with 05_inference.py")
    print("="*70)