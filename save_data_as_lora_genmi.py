# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 18:15:00 2025

@author: sijin
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, load_from_disk
import json

def load_descriptions(excel_path):
    """Load room descriptions from Excel file."""
    if not os.path.exists(excel_path):
        print(f"Warning: Description file not found at {excel_path}")
        return {}
    
    try:
        df = pd.read_excel(excel_path)
        descriptions = {row['Place'].lower(): row['Depscription'] 
                       for _, row in df.iterrows() if pd.notna(row['Place'])}
        return descriptions
    except Exception as e:
        print(f"Error loading descriptions from {excel_path}: {str(e)}")
        return {}

def load_materials(excel_path):
    """Load materials information from Excel file."""
    if not os.path.exists(excel_path):
        print(f"Warning: Material file not found at {excel_path}")
        return {}
    
    try:
        df = pd.read_excel(excel_path)
        materials = {}
        for _, row in df.iterrows():
            if pd.notna(row['Place']):
                place = row['Place'].lower()
                if place not in materials:
                    materials[place] = []
                material_info = {
                    'product': row['Product'] if pd.notna(row.get('Product', '')) else '',
                    'type': row['Type'] if pd.notna(row.get('Type', '')) else '',
                    'colour': row['Colour'] if pd.notna(row.get('Colour', '')) else '',
                    'arc_code': row['Arc_code'] if pd.notna(row.get('Arc_code', '')) else ''
                }
                materials[place].append(material_info)
        return materials
    except Exception as e:
        print(f"Error loading materials from {excel_path}: {str(e)}")
        return {}

def format_materials_text(materials_list):
    """Format materials information into readable text."""
    if not materials_list:
        return "No specific materials information available."
    
    text_parts = ["Materials Specifications:"]
    for material in materials_list:
        parts = []
        if material['product']:
            parts.append(f"Product: {material['product']}")
        if material['type']:
            parts.append(f"Type: {material['type']}")
        if material['colour']:
            parts.append(f"Colour: {material['colour']}")
        if material['arc_code']:
            parts.append(f"Code: {material['arc_code']}")
        text_parts.append(" | ".join(parts))
    
    return "\n".join(text_parts)

def create_llama_dataset(base_path, descriptions_path, materials_path, project_name, coho_base_path):
    """Create dataset in LLaMA Factory format"""
    descriptions = load_descriptions(descriptions_path)
    materials = load_materials(materials_path)
    
    transformed_examples = []
    
    # Walk through all directories in the resize folder
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Get path components
                relative_path = os.path.relpath(root, base_path)
                
                # Extract room type from the path or filename
                # This may need adjustment based on your exact folder structure
                if '_' in relative_path:
                    room_type = relative_path.split('_')[0].lower()
                else:
                    room_type = relative_path.lower()
                
                # Get image path using the base folder name from coho_base_path
                base_folder_name = coho_base_path.split('\\')[-1]
                image_path = f"/srv/scratch/dbgcse/sijin/LLaMA-Factory/{base_folder_name}/{project_name}/{relative_path}/{file}"
                
                
                # Get descriptions
                room_description = descriptions.get(room_type, "No specific description available.")
                room_materials = materials.get(room_type, [])
                materials_text = format_materials_text(room_materials)
                
                # Create prompt text
                prompt_text = f"""Create a compelling, immersive VR narrative (one-short paragraph, under 100 words) using the **second person perspective** to give viewers an immersive experience, focusing on the {room_type} within {project_name}. Analyze the given image and room context:

                                Room Context:
                                - The room is described as: {room_description}
                                - Materials used for the {room_type}: {materials_text}

                                Narrative Instructions:
                                - Analyze the image and room context to craft a vivid, one-paragraph narrative (under 100 words) in **second person** that transports *you*, the viewer, into the scene.
                                - Engage *your* senses (sight, sound, smell, touch) to create realism and presence for *your* experience.
                                - Incorporate dynamic action and emotional depth, considering *your* potential feelings within the space as **you** move through the room.
                                - Focus the narrative on the {room_type} as the central element, described from **your** perspective as **you** are virtually present.
                                - Consider the feelings that the objects and the room evoke in **you** as **you** virtually experience it.

                                **- Room-Specific Guidance:**
                                    * **For Kitchens or Bathrooms:** Primarily focus your description on the materials used and practical items present (appliances, fixtures), and the *feelings* these functional aspects evoke in **you** (e.g., efficiency, cleanliness, luxury).
                                    * **For Balconies, Living Rooms, or Bedrooms:**  Consider and describe any outside views visible in the image, and how they influence the room's atmosphere and the *feelings* they inspire in **you** (e.g., tranquility, openness, coziness) as **you** look at them."""
                
                # Create example in LLaMA Factory format
                example = {
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt_text
                        }
                    ],
                    'image_path': image_path
                }
                
                transformed_examples.append(example)
    
    # Create dataset
    features = Features({
        'messages': Sequence({
            'role': Value('string'),
            'content': Value('string')
        }),
        'image_path': Value('string')
    })
    
    dataset = Dataset.from_list(transformed_examples, features=features)
    return dataset

def process_coho_folders_for_dataset(coho_base_path):
    """
    Loop through each folder in the COHO directory, find the resize folder and Excel files,
    create datasets, and save them in a lora_dataset folder
    """
    # Create the main lora_dataset folder
    lora_dataset_path = os.path.join(coho_base_path, "lora_dataset")
    os.makedirs(lora_dataset_path, exist_ok=True)
    
    # Get all directories in the COHO base folder
    subfolders = [f for f in os.listdir(coho_base_path) 
                 if os.path.isdir(os.path.join(coho_base_path, f)) and 
                 not f in ["lora_dataset"]]  # Exclude the lora_dataset folder itself
    
    total_datasets = 0
    total_examples = 0
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(coho_base_path, subfolder)
        
        # Define paths
        resize_folder_path = os.path.join(subfolder_path, "resize")
        descriptions_path = os.path.join(subfolder_path, "dep.xlsx")
        materials_path = os.path.join(subfolder_path, "material.xlsx")
        
        # Check if resize folder exists
        if os.path.exists(resize_folder_path) and os.path.isdir(resize_folder_path):
            print(f"\nProcessing {subfolder} for dataset creation...")
            
            # Define output dataset path
            dataset_output_path = os.path.join(lora_dataset_path, f"{subfolder}_lora_dataset")
            
            # Create dataset
            try:
                dataset = create_llama_dataset(
                    resize_folder_path, 
                    descriptions_path, 
                    materials_path,
                    project_name=subfolder,
                    coho_base_path=coho_base_path)
                
                # Save dataset
                dataset.save_to_disk(dataset_output_path)
                
                # Print statistics
                print(f"Dataset for {subfolder} created with {len(dataset)} examples")
                total_datasets += 1
                total_examples += len(dataset)
                
            except Exception as e:
                print(f"Error creating dataset for {subfolder}: {str(e)}")
        else:
            print(f"No resize folder found for {subfolder}, skipping dataset creation")
    
    print(f"\nAll folders processed for dataset creation!")
    print(f"Created {total_datasets} datasets with a total of {total_examples} examples")

# Main execution
if __name__ == "__main__":
    coho_base_path = r"G:\Arcanite\ARC-PENTHOUSE"
    process_coho_folders_for_dataset(coho_base_path)