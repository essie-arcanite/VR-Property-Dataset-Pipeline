# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:03:00 2025

@author: sijin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_perspective_frame(img, heading, fov, pitch, output_size, perspective_adjust=1.0):
    """Generate a frame with adjustable perspective"""
    height, width = img.shape[:2]
    
    # Convert angles to radians
    fov_rad = np.radians(fov)
    heading_rad = np.radians(heading)
    pitch_rad = np.radians(pitch)
    
    # Adjust output dimensions with perspective
    output_width, output_height = output_size
    
    # Create perspective-adjusted meshgrid
    x = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), output_width) * perspective_adjust
    y = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), output_height)
    xv, yv = np.meshgrid(x, y)

    # Apply perspective distortion
    z = np.ones_like(xv) + xv**2 * 0.1  # Adjust the 0.1 factor to control perspective strength
    norm = np.sqrt(xv**2 + yv**2 + z**2)
    
    x = xv / norm
    y = yv / norm
    z = z / norm

    # Enhanced rotation matrices with perspective consideration
    rot_mat_h = np.array([
        [np.cos(heading_rad), 0, -np.sin(heading_rad)],
        [0, 1, 0],
        [np.sin(heading_rad), 0, np.cos(heading_rad)]
    ])
    
    rot_mat_p = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    rot_mat = rot_mat_h @ rot_mat_p
    
    # Apply rotation
    coords = np.stack([x, y, z], axis=-1)
    rotated_coords = np.einsum('ij,klj->kli', rot_mat, coords)

    # Convert to spherical coordinates with perspective adjustment
    phi = np.arctan2(rotated_coords[..., 0], rotated_coords[..., 2])
    theta = np.arcsin(np.clip(rotated_coords[..., 1], -1, 1))

    # Convert to image coordinates
    u = (phi / (2 * np.pi) + 0.5) * width
    v = (theta / np.pi + 0.5) * height

    # Remap image with perspective consideration
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    frame = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    return frame

def generate_main_frames(input_path, output_folder, num_frames=12):
    """Generate only main view frames"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Read input image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read the image file")
    
    # Calculate viewing angles
    headings = np.linspace(0, 360, num_frames, endpoint=False)
    
    # Main view parameters
    main_params = {
        'size': (1920, 1680),  # HD resolution
        'fov': 90,             # Field of view
        'perspective': 1.2     # Perspective adjustment
    }
    
    frames = []
    
    for i, heading in enumerate(headings):
        try:
            frame = generate_perspective_frame(
                img, 
                heading, 
                main_params['fov'],
                pitch=-5,  # Slight downward tilt
                output_size=main_params['size'],
                perspective_adjust=main_params['perspective']
            )
            
            # Save frame
            output_path = os.path.join(output_folder, f'frame_{i:03d}.jpg')
            cv2.imwrite(output_path, frame)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            print(f"Generated frame {i+1}/{num_frames}")
            
        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")
            continue
    
    # Show sample frame
    if frames:
        plt.figure(figsize=(15, 8))
        plt.imshow(frames[0])
        plt.title('Sample Frame')
        plt.axis('off')
        plt.show()
    
    return len(frames)

def generate_main_frames(input_path, output_folder, num_frames=72):
    """Generate only main view frames"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Read input image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read the image file: {input_path}")
    
    # Calculate viewing angles
    headings = np.linspace(0, 360, num_frames, endpoint=False)
    
    # Main view parameters
    main_params = {
        'size': (1920, 1680),  # HD resolution
        'fov': 90,             # Field of view
        'perspective': 1.2     # Perspective adjustment
    }
    
    frames = []
    
    for i, heading in enumerate(headings):
        try:
            frame = generate_perspective_frame(
                img, 
                heading, 
                main_params['fov'],
                pitch=-5,  # Slight downward tilt
                output_size=main_params['size'],
                perspective_adjust=main_params['perspective']
            )
            
            # Save frame
            output_path = os.path.join(output_folder, f'frame_{i:03d}.jpg')
            cv2.imwrite(output_path, frame)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            print(f"Generated frame {i+1}/{num_frames} for {os.path.basename(input_path)}")
            
        except Exception as e:
            print(f"Error processing frame {i} for {os.path.basename(input_path)}: {str(e)}")
            continue
    
    # Show sample frame
    if frames:
        plt.figure(figsize=(15, 8))
        plt.imshow(frames[0])
        plt.title(f'Sample Frame - {os.path.basename(input_path)}')
        plt.axis('off')
        plt.show()
    
    return len(frames)

def process_all_images(input_base_folder):
    """Process all panoramic images in all subfolders of the input base folder"""
    # Get all directories in the input base folder
    subfolders = [f for f in os.listdir(input_base_folder) 
                  if os.path.isdir(os.path.join(input_base_folder, f))]
    
    total_processed_images = 0
    total_processed_folders = 0
    
    for subfolder in subfolders:
        input_folder = os.path.join(input_base_folder, subfolder)
        output_folder_name = f"{subfolder}_frames"
        # Create frames folder INSIDE each subfolder
        output_folder = os.path.join(input_folder, output_folder_name)
        
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing folder: {subfolder}")
        
        # Get all jpg files in the input folder
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith('.jpg')]
        
        folder_processed = 0
        
        for image_file in image_files:
            # Create image-specific output subfolder
            image_output_folder_name = os.path.splitext(image_file)[0]
            image_output_folder = os.path.join(output_folder, image_output_folder_name)
            
            # Full paths
            input_path = os.path.join(input_folder, image_file)
            
            print(f"  Processing {image_file}...")
            try:
                num_generated = generate_main_frames(
                    input_path=input_path,
                    output_folder=image_output_folder,
                    num_frames=6  # Number of frames per image
                )
                print(f"  Successfully generated {num_generated} frames for {image_file}")
                folder_processed += 1
                total_processed_images += 1
            except Exception as e:
                print(f"  Error processing {image_file}: {str(e)}")
        
        print(f"Processed {folder_processed} images in {subfolder}")
        if folder_processed > 0:
            total_processed_folders += 1
    
    print(f"\nTotal: Processed {total_processed_images} images across {total_processed_folders} folders")
    return total_processed_images
input_base_folder = r"G:\Arcanite\ARC-PENTHOUSE"

# Run the processing
total_processed = process_all_images(input_base_folder)