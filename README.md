# Falcon N5: AI vs. Original Image Classification

This repository contains tools and models for detecting AI-generated images using the Swin Transformer architecture.

## Overview
The project uses the **Swin Transformer (swin_base_patch4_window7_224)** to classify images as either "Original (Real)" or "AI Generated". It includes a training notebook and inference scripts, along with Grad-CAM visualization to explain the model's focus.

## Project Structure
- `ai-vs-original-image-classification.ipynb`: Jupyter notebook for training, evaluation, and visualization.
- `code/data_load.py`: Data loading utilities.
- `code/load_model.py`: Inference script for testing local images.
- `model/swin_ai_detector.pth`: Pre-trained model weights.

## Installation
1. Install dependencies:
   ```bash
   pip install torch torchvision timm pytorch-grad-cam pillow matplotlib datasets
   ```
2. (Optional) Install Git LFS to handle the model weights:
   ```bash
   git lfs install
   ```

## Usage
To test the model on a single image, update the `image_path` in `code/load_model.py` and run:
```bash
python code/load_model.py
```

## Dataset
This project was trained on the [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset), which includes 96,000 images from both real (MS COCO) and synthetic sources (Stable Diffusion, Midjourney, etc.).
