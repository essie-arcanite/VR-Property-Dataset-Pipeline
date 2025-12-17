# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:45:09 2025

@author: sijin
"""

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

import json
from pathlib import Path
from datasets import Dataset, Features, Image, Value
import pandas as pd
from PIL import Image as PILImage
import numpy as np



client = genai.Client(api_key='YOUR API')
import json
from pathlib import Path
from datasets import Dataset, Features, Image, Value
import pandas as pd
from PIL import Image as PILImage
import numpy as np



def load_captions(directory):
    """Load captions from the captions.json file in the given directory."""
    captions_file = directory / "captions.json"
    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load captions from {captions_file}: {e}")
        return {}

def load_intro_captions(file_path):
    """Load intro captions from the intro_captions.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load intro captions from {file_path}: {e}")
        return {}

def create_radiology_style_dataset(base_dir, output_path, intro_captions_file):
    """
    Create a dataset similar to Radiology_mini format with direct image loading.
    
    Args:
        base_dir: Base directory containing numbered folders (0-17)
        output_path: Path where dataset will be saved
        intro_captions_file: Path to intro_captions.json
    """
    base_dir = Path(base_dir)
    output_path = Path(output_path)
    
    # Load intro captions
    intro_captions = load_intro_captions(intro_captions_file)
    
    # Prepare data
    data = {
        'image': [],
        'image_id': [],
        'video_id': [],
        'source_dir': [],
        'content': [],
        'response': []
    }
    
    # Get all numbered directories
    numbered_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()], 
                         key=lambda x: int(x.name))
    
    print(f"Found {len(numbered_dirs)} numbered directories")
    
    # Process each numbered directory
    for num_dir in numbered_dirs:
        print(f"\nProcessing directory {num_dir.name}...")
        
        # Get intro caption for this directory
        intro_data = intro_captions.get(num_dir.name, {})
        intro_caption = intro_data.get("caption", "")
        intro_duration = intro_data.get("duration", "")

        # Load room captions for this directory
        captions_data = load_captions(num_dir)
        
        # Process each category directory within the numbered directory
        for category_dir in num_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                category_display = category.replace('_', ' ')
                if category_display == 'temp':
                    continue
                print(f"  Processing category: {category}")
                
                # Get caption data for this category
                category_caption = captions_data.get(category_display, [{"caption": "", "time_range": ""}])[0]
                
                # Process all images in this category
                image_files = list(category_dir.glob("*"))
                print(f"    Found {len(image_files)} images")
                if len(image_files)>=2:
                    image_files = image_files[0:2]
                for idx, img_path in enumerate(image_files):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Create unique image ID including directory number
                        img_id = f"dir{num_dir.name}_{category}_{idx:04d}"
                        
                        # Load and convert image to array
                        img = PILImage.open(img_path)
                        text = f'''Apartment Introduction: {{<image>}}Here is the introduction for Apartment {num_dir.name} : {intro_caption}
                        Detailed Description: {category_caption.get("caption", "")}
                        Analyze the given image and craft a compelling, immersive one-small-paragraph VR narrative about the {category_display} in this apartment that fully engages the reader’s senses. The narrative should transport the reader into the scene, incorporating vivid descriptions, dynamic action, and emotional depth. Focus on creating a sense of presence and realism, making the experience feel truly lifelike.
                        '''
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[text, img])
                        print(response)
                        image_path_without_extension = str(img_path).rsplit('.', 1)[0]
                        with open(image_path_without_extension+".json", "w") as f:
                            json.dump(response.text, f, indent=4)
                        data['image'].append(img_path)
                        data['image_id'].append(img_id)
                        data['video_id'].append(f"video_{num_dir.name}")
                        data['source_dir'].append(num_dir.name)

                        data['content'].append(text)
                        data['response'].append(response.text)
    print("\nCreating dataset...")
    # Create dataset with image feature
    features = Features({
        'image': Value('string'),
        'image_id': Value('string'),
        'video_id': Value('string'),
        'source_dir': Value('string'),
        'content': Value('string'),
        'response': Value('string')
    })
    
    dataset = Dataset.from_dict(data, features=features)
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total images: {len(dataset)}")
    print("\nImages per directory:")
    dir_counts = pd.Series(data['source_dir']).value_counts().sort_index()
    for dir_num, count in dir_counts.items():
        print(f"Directory {dir_num}: {count} images")
    
    return dataset

base_dir = Path(r"G:\Arcanite\all_video_snapshots")
output_path = Path("room_dataset")
intro_captions_file = "G:/Arcanite/video_intros/intro_captions.json"

# Create the dataset
print("Starting dataset creation...")
dataset = create_radiology_style_dataset(base_dir, output_path, intro_captions_file)

# Save the dataset
print("\nSaving dataset...")
dataset.save_to_disk("yt_dataset_gemni")

print(f"\nDataset saved successfully to {output_path / 'dataset'}")
print("\nTo load and view the dataset:")
print("from datasets import load_from_disk")
print('dataset = load_from_disk("room_dataset/dataset")')
print('# View first image and captions:')
print('example = dataset[0]')
print('print(f"Room caption: {example["caption"]}")')
print('print(f"Intro caption: {example["intro_caption"]}")')
                            