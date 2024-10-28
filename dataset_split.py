import json
import os
import shutil
import random
from typing import Dict, List, Set, Tuple

class DatasetSplitter:
    def __init__(
        self,
        json_path: str,
        image_folder: str,
        train_folder: str,
        val_folder: str,
        train_ratio: float = 0.8
    ):
        self.json_path = json_path
        self.image_folder = image_folder
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.train_ratio = train_ratio
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Create image_id to filename mapping
        self.image_map = {img['id']: img['file_name'] for img in self.data['images']}
        
    def split_dataset(self) -> Tuple[Dict, Dict]:
        """Split the dataset into training and validation sets."""
        
        # Get all unique image IDs
        all_image_ids = set(img['id'] for img in self.data['images'])
        
        # Calculate split sizes
        n_train = int(len(all_image_ids) * self.train_ratio)
        
        # Randomly split image IDs
        train_image_ids = set(random.sample(list(all_image_ids), n_train))
        val_image_ids = all_image_ids - train_image_ids
        
        # Split images and annotations
        train_data = self._create_split_data(train_image_ids)
        val_data = self._create_split_data(val_image_ids)
        
        return train_data, val_data
    
    def _create_split_data(self, image_ids: Set[int]) -> Dict:
        """Create a new data dictionary for the given image IDs."""
        
        # Filter images
        images = [img for img in self.data['images'] if img['id'] in image_ids]
        
        # Filter annotations
        annotations = [
            ann for ann in self.data['annotations'] 
            if ann['image_id'] in image_ids
        ]
        
        # Create new data dictionary
        return {
            'images': images,
            'annotations': annotations,
            'categories': self.data['categories']
        }
    
    def save_splits(self, train_data: Dict, val_data: Dict):
        """Save the split data and copy images to respective folders."""
        
        # Save JSON files
        train_json = os.path.join(os.path.dirname(self.json_path), 'train.json')
        val_json = os.path.join(os.path.dirname(self.json_path), 'val.json')
        
        with open(train_json, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(val_json, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        # Copy images
        self._copy_images(train_data['images'], self.train_folder)
        self._copy_images(val_data['images'], self.val_folder)
    
    def _copy_images(self, images: List[Dict], target_folder: str):
        """Copy images to the target folder."""
        os.makedirs(target_folder, exist_ok=True)
        
        for img in images:
            src = os.path.join(self.image_folder, img['file_name'])
            dst = os.path.join(target_folder, img['file_name'])
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: Image not found - {img['file_name']}")

def main():
    # Configuration
    json_path = r"C:\Users\armaa\Downloads\CV_assignment\random_sample_mavi_2_gt.json"
    image_folder = r"C:\Users\armaa\Downloads\CV_assignment\dataset"
    train_folder = r"C:\Users\armaa\Downloads\CV_assignment\training"
    val_folder = r"C:\Users\armaa\Downloads\CV_assignment\validation"
    
    # Create splitter instance
    splitter = DatasetSplitter(
        json_path=json_path,
        image_folder=image_folder,
        train_folder=train_folder,
        val_folder=val_folder,
        train_ratio=0.8  # 160-40 split ≈ 0.8
    )
    
    # Split dataset
    train_data, val_data = splitter.split_dataset()
    
    # Save splits
    splitter.save_splits(train_data, val_data)
    
    # Print statistics
    print(f"Training set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Validation set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")

if __name__ == "__main__":
    main()