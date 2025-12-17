# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:12:27 2025

@author: sijin
"""

from datasets import Dataset, Features, Sequence, Value, load_from_disk
from datasets.features.features import Image as DatasetImage  # Changed this import
from PIL import Image
import io
import huggingface_hub

huggingface_hub.login("")

def resize_image(image_path, max_size=(518, 336)):
    """
    Resize image while maintaining aspect ratio
    """
    img = Image.open(image_path).convert('RGB')
    ratio = min(max_size[0] / img.size[0], max_size[1] / img.size[1])
    new_size = tuple(int(dim * ratio) for dim in img.size)
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    return resized_img

def transform_dataset_for_llama(dataset):
    transformed_examples = []
    
    for entry in dataset:
        if not entry.get('response') or entry['response'] == '':
            continue
            
        try:
            # Resize image
            image = entry['image']
            
            # Create messages list with explicit image token
            messages = [
                {
                    "role": "user",
                    "content": f"{entry.get('content', '')}"  # Image token at specific position
                },
                {
                    "role": "assistant",
                    "content": entry['response']
                }
            ]
            
            example = {
                'messages': messages,
                'images': [image]  # Must match number of <image> tokens
            }
            transformed_examples.append(example)
            
        except Exception as e:
            print(f"Error processing image {entry.get('image')}: {str(e)}")
            continue
    
    return transformed_examples

def transform_material_for_llama(dataset):
    transformed_examples = []
    
    for entry in dataset:
        if not entry.get('description') or entry['description'] == '':
            continue
            
        try:
            # Resize image
            image = entry['image']
            
            # Format description
            description = entry['description']
            if isinstance(description, list):
                description = '\n'.join(description)
            
            # Create messages list with explicit image token
            messages = [
                {
                    "role": "user",
                    "content": f"<image>{entry['introduction']}"  # Image token at specific position
                },
                {
                    "role": "assistant",
                    "content": description
                }
            ]
            
            example = {
                'messages': messages,
                'images': [image]  # Must match number of <image> tokens
            }
            transformed_examples.append(example)
            
        except Exception as e:
            print(f"Error processing image {entry.get('image')}: {str(e)}")
            continue
    
    return transformed_examples

def create_dataset():
    """
    Create and push dataset to Hugging Face Hub
    """
    print("Loading datasets from disk...")
    dataset = load_from_disk("yt_dataset_gemni")
    #dataset_mat = load_from_disk("material_dataset_path")
    
    print("Transforming datasets...")
    transformed_dataset = transform_dataset_for_llama(dataset)
    #transformed_material = transform_material_for_llama(dataset_mat)
    
    all_examples = transformed_dataset #+ transformed_material
    
    print(f"Total examples: {len(all_examples)}")
    
    # Define features
    features = Features({
        'messages': [
            {
                'role': Value('string',id = None),
                'content': Value('string',id = None)
            }
        ],
        'images': Sequence(feature=DatasetImage(decode=True, id=None), length=-1, id=None),
    })
    
    print("Creating Hugging Face dataset...")
    hf_dataset = Dataset.from_list(all_examples, features=features)
    
    print("Pushing to Hugging Face Hub...")
    hf_dataset.push_to_hub("Essie0715/arc_gen")
    
    return hf_dataset

if __name__ == "__main__":
    dataset = create_dataset()
    print(f"Dataset created successfully with {len(dataset)} examples")