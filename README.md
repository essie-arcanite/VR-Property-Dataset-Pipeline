# VR Property Showcase Dataset Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg" alt="HuggingFace">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/Essie0715/arc_gen">🤗 <b>View Dataset on HuggingFace Hub</b></a>
</p>

A comprehensive data processing pipeline for creating vision-language model training datasets from **360° VR panoramic images** and **YouTube property tour videos**. This toolkit transforms raw VR imagery and video content into structured datasets suitable for fine-tuning multimodal LLMs (e.g., LLaMA, Qwen-VL) for immersive property description generation.

## 🎯 Overview

This pipeline enables you to:
- Extract perspective views from 360° equirectangular panoramic images
- Process YouTube property tour videos into frame sequences
- Generate rich, immersive property descriptions using Google Gemini API
- Build HuggingFace-compatible datasets for LoRA fine-tuning
- Push datasets directly to HuggingFace Hub

## 📁 Project Structure

```
VR-Property-Dataset-Pipeline/
├── README.md
├── requirements.txt
├── LICENSE
├── scripts/
│   ├── VR_pic_to_fill.py              # 360° panorama → perspective frames
│   ├── resize.py                       # Image resizing utility
│   ├── gemni_to_generate.py           # Gemini API caption generation
│   ├── save_data_as_lora_genmi.py     # Build LoRA dataset from VR images
│   └── make_youtube_dataset_for_hfi_final_version_genmi.py  # Push to HuggingFace
├── data/
│   └── .gitkeep
└── docs/
    └── pipeline_workflow.md
```

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PROCESSING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw VR     │     │  Perspective │     │   Resized    │     │   LoRA      │
│  Panoramas   │────▶│    Frames    │────▶│    Images    │────▶│  Dataset    │
│   (360°)     │     │  (6-72/img)  │     │  (640×360)   │     │             │
└──────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
       │                                                              │
       │  VR_pic_to_fill.py      resize.py       save_data_as_lora_genmi.py
       │                                                              │
       │                                                              ▼
       │                                                     ┌─────────────┐
       │                                                     │  HuggingFace│
       │                                                     │     Hub     │
       │                                                     └─────────────┘
       │                                                              ▲
       ▼                                                              │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│   YouTube    │     │   Extracted  │     │   Gemini     │────────────┘
│   Videos     │────▶│    Frames    │────▶│  Captions    │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                                    gemni_to_generate.py
```

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/VR-Property-Dataset-Pipeline.git
cd VR-Property-Dataset-Pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

```txt
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
pandas>=1.3.0
openpyxl>=3.0.0
datasets>=2.14.0
huggingface_hub>=0.16.0
google-genai>=0.1.0
matplotlib>=3.5.0
```

## 🚀 Usage

### Step 1: Process VR Panoramic Images

Convert 360° equirectangular panoramas into multiple perspective views:

```python
from scripts.VR_pic_to_fill import process_all_images

# Process all panoramic images in the base folder
input_base_folder = "path/to/your/VR_images"
process_all_images(input_base_folder)
```

**Input Structure:**
```
VR_images/
├── Project_A/
│   ├── kitchen.jpg      # 360° panorama
│   ├── bedroom.jpg
│   └── livingroom.jpg
└── Project_B/
    └── ...
```

**Output:** 6 perspective frames per panorama at different heading angles (0°, 60°, 120°, 180°, 240°, 300°).

### Step 2: Resize Images

Resize extracted frames for model training:

```python
from scripts.resize import resize_frames_in_coho_folders

coho_base_path = "path/to/your/VR_images"
resize_frames_in_coho_folders(coho_base_path)
```

### Step 3: Generate Captions with Gemini

Use Google Gemini API to generate immersive property descriptions:

```python
from scripts.gemni_to_generate import create_radiology_style_dataset

# Set your Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey

dataset = create_radiology_style_dataset(
    base_dir="path/to/video_snapshots",
    output_path="room_dataset",
    intro_captions_file="path/to/intro_captions.json"
)
dataset.save_to_disk("yt_dataset_gemni")
```

### Step 4: Build LoRA Dataset

Create a LoRA-compatible dataset with prompts and image paths:

```python
from scripts.save_data_as_lora_genmi import process_coho_folders_for_dataset

coho_base_path = "path/to/your/processed_images"
process_coho_folders_for_dataset(coho_base_path)
```

**Required Excel Files:**
- `dep.xlsx` - Room descriptions (columns: `Place`, `Depscription`)
- `material.xlsx` - Material specifications (columns: `Place`, `Product`, `Type`, `Colour`, `Arc_code`)

### Step 5: Push to HuggingFace Hub

```python
from scripts.make_youtube_dataset_for_hfi_final_version_genmi import create_dataset

# Make sure to set your HuggingFace token
# huggingface_hub.login("your_token_here")

dataset = create_dataset()
# Dataset will be pushed to HuggingFace Hub
```

## 📊 Dataset Format

The output dataset follows the LLaMA Factory / HuggingFace conversation format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>Create a compelling VR narrative for this kitchen..."
    },
    {
      "role": "assistant", 
      "content": "As you step into this modern kitchen, gleaming quartz countertops..."
    }
  ],
  "images": ["path/to/image.jpg"]
}
```

## 🤗 Using the Pre-built Dataset

You can directly use our pre-built dataset from HuggingFace Hub:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Essie0715/arc_gen")

# View dataset info
print(dataset)

# Access an example
example = dataset['train'][0]
print(f"User prompt: {example['messages'][0]['content'][:100]}...")
print(f"Assistant response: {example['messages'][1]['content'][:100]}...")
```

**Dataset Link:** [https://huggingface.co/datasets/Essie0715/arc_gen](https://huggingface.co/datasets/Essie0715/arc_gen)

## 🔧 Configuration

### VR Frame Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 6 | Number of perspective views per panorama |
| `fov` | 90° | Field of view for each perspective |
| `pitch` | -5° | Vertical viewing angle (slight downward tilt) |
| `output_size` | 1920×1680 | Output resolution |

### Resize Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | 640×360 | Maximum output dimensions |
| `quality` | 95 | JPEG compression quality |

## 📝 Prompt Template

The pipeline uses a structured prompt for generating immersive VR narratives:

> Create a compelling, immersive VR narrative (one-short paragraph, under 100 words) using the **second person perspective** to give viewers an immersive experience...
> 
> - Engage your senses (sight, sound, smell, touch)
> - Incorporate dynamic action and emotional depth
> - Focus on room-specific elements and materials

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google Gemini API](https://ai.google.dev/) for multimodal caption generation
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) for dataset management
- [OpenCV](https://opencv.org/) for image processing

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🎯 Use Cases

This dataset and pipeline can be used for:

- **Fine-tuning Vision-Language Models**: Train models like LLaVA, Qwen-VL, or InternVL to generate immersive property descriptions
- **LoRA Training**: Create efficient adapters for multimodal LLMs
- **Real Estate AI Applications**: Build AI-powered virtual tour narration systems
- **VR/AR Content Generation**: Automate descriptive content for immersive experiences

### Training with LLaMA Factory

```yaml
# Example LLaMA Factory configuration
model_name_or_path: Qwen/Qwen-VL-Chat
dataset: Essie0715/arc_gen
template: qwen_vl
finetuning_type: lora
lora_rank: 8
output_dir: ./output
```

---

<p align="center">
  Made with ❤️ for the VR/AR property visualization community
</p>
