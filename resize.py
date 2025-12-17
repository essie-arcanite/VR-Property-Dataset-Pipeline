# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:11:54 2025

@author: sijin
"""

import os
from PIL import Image
from pathlib import Path

def resize_frames_in_coho_folders(coho_base_path):
    """
    Loop through each folder in the COHO directory, find the _frames folder,
    and create a corresponding resize folder in the same location
    """
    # Get all directories in the COHO base folder
    subfolders = [f for f in os.listdir(coho_base_path) 
                  if os.path.isdir(os.path.join(coho_base_path, f))]
    
    total_success = 0
    total_errors = 0
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(coho_base_path, subfolder)
        
        # Look for the _frames folder
        frames_folder_name = f"{subfolder}_frames"
        frames_folder_path = os.path.join(subfolder_path, frames_folder_name)
        
        # Create resize folder path
        resize_folder_path = os.path.join(subfolder_path, "resize")
        
        # Check if frames folder exists
        if os.path.exists(frames_folder_path) and os.path.isdir(frames_folder_path):
            print(f"\nProcessing frames in: {frames_folder_path}")
            
            # Process images in the frames folder
            success, errors = process_images(frames_folder_path, resize_folder_path)
            
            total_success += success
            total_errors += errors
            
            print(f"Completed processing frames for {subfolder}")
        else:
            print(f"No frames folder found for {subfolder}, skipping")
    
    print(f"\nAll folders processed!")
    print(f"Total successfully resized: {total_success} images")
    print(f"Total errors: {total_errors} images")
    return total_success, total_errors

def process_images(frames_folder_path, resize_folder_path):
    """
    Process all images in the frames folder and save resized versions to the resize folder
    """
    success_count = 0
    error_count = 0
    
    # Create resize folder if it doesn't exist
    os.makedirs(resize_folder_path, exist_ok=True)
    
    # Walk through all directories in the frames folder
    for root, dirs, files in os.walk(frames_folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Get relative path from the frames folder
                rel_path = os.path.relpath(root, frames_folder_path)
                input_path = os.path.join(root, file)
                
                # Create corresponding output path in the resize folder
                output_dir = os.path.join(resize_folder_path, rel_path)
                output_path = os.path.join(output_dir, file)
                
                # Resize the image
                if resize_image(input_path, output_path):
                    success_count += 1
                    print(f"Successfully resized: {file}")
                else:
                    error_count += 1
    
    print(f"Resizing complete for this folder!")
    print(f"Successfully resized: {success_count} images")
    print(f"Errors encountered: {error_count} images")
    return success_count, error_count

def resize_image(image_path, output_path, max_size=(640, 360)):
    """
    Resize image while maintaining aspect ratio and save to output path
    """
    try:
        img = Image.open(image_path).convert('RGB')
        ratio = min(max_size[0] / img.size[0], max_size[1] / img.size[1])
        new_size = tuple(int(dim * ratio) for dim in img.size)
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resized_img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":
    coho_base_path = r"G:\Arcanite\ARC-PENTHOUSE"
    resize_frames_in_coho_folders(coho_base_path)