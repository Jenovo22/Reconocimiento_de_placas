import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, output_dir, split_ratio=0.8):
    """
    Splits a flat Label Studio export into YOLO train/val structure.
    
    Args:
        source_dir (str): Path containing 'images' and 'labels' folders from LS export.
        output_dir (str): Path where the new structured dataset will be created.
        split_ratio (float): Ratio of training data (default 0.8 for 80%).
    """
    
    # 1. Setup paths based on your screenshot structure
    images_path = os.path.join(source_dir, 'images')
    labels_path = os.path.join(source_dir, 'labels')
    
    # Verify source exists
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"Error: Could not find 'images' or 'labels' folders in {source_dir}")
        return

    # 2. Get all image files (supported extensions)
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(images_path) if f.lower().endswith(supported_ext)]
    
    # Shuffle to ensure random distribution
    random.shuffle(all_images)
    
    # Calculate split index
    train_count = int(len(all_images) * split_ratio)
    
    # 3. Create Destination Structure
    subsets = ['train', 'val']
    folders = ['images', 'labels']
    
    for subset in subsets:
        for folder in folders:
            os.makedirs(os.path.join(output_dir, subset, folder), exist_ok=True)
            
    print(f"Total images found: {len(all_images)}")
    print(f"Training set: {train_count} images")
    print(f"Validation set: {len(all_images) - train_count} images")
    print("-" * 30)

    # 4. Move files
    missing_labels = 0
    
    for i, image_file in enumerate(all_images):
        # Determine if this image goes to 'train' or 'val'
        subset = 'train' if i < train_count else 'val'
        
        # Define source and destination for Image
        src_img = os.path.join(images_path, image_file)
        dst_img = os.path.join(output_dir, subset, 'images', image_file)
        
        # Define source and destination for Label
        # Use Path(image_file).stem to get filename without extension
        label_file = Path(image_file).stem + '.txt'
        src_label = os.path.join(labels_path, label_file)
        dst_label = os.path.join(output_dir, subset, 'labels', label_file)
        
        # Copy Image
        shutil.copy(src_img, dst_img)
        
        # Copy Label (if it exists)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            missing_labels += 1
            # If no label exists, it's considered a "background image" (no objects)
            # YOLO accepts this, but we warn just in case.
            
    print("-" * 30)
    print(f"Process complete. Dataset ready at: {output_dir}")
    if missing_labels > 0:
        print(f"Warning: {missing_labels} images had no corresponding .txt label file.")

if __name__ == "__main__":
    # CONFIGURATION
    # Use '.' if running inside the downloaded folder, or provide full path
    SOURCE_DIRECTORY = r'C:\Users\jeron\OneDrive\Documentos\A_Semillero\Data\project-10-at-2026-06' 
    
    # Where you want the final dataset ready for YOLO
    OUTPUT_DIRECTORY = 'dataset_final' 
    
    split_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
