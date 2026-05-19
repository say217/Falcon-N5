# Falcon N5: AI vs. Authentic

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%23000000.svg?style=for-the-badge&logo=Seaborn&logoColor=white)
![Timm](https://img.shields.io/badge/Timm-red?style=for-the-badge)
![Neural Network](https://img.shields.io/badge/Neural%20Network-blue?style=for-the-badge)

<img width="704" height="359" alt="image" src="https://github.com/user-attachments/assets/2545e899-0d76-4cc4-9692-e636b88213d0" />

## Overview
Project Falcon N5 focuses on building a robust computer vision classifier capable of distinguishing between authentic, human-created images and AI-generated content. Leveraging the powerful Swin Transformer architecture (swin_base_patch4_window7_224), the model is trained to perform binary classification to tackle the growing challenge of digital forgery and artificial content verification. Because the training pipeline encounters an imbalanced distribution of real versus synthetic media, Falcon N5 incorporates a strategically weighted cross-entropy loss function to ensure the model penalizes misclassifications on minority "Real" data heavily, ultimately achieving high validation accuracy and reliable prediction confidence.

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
The model is trained and evaluated using the comprehensive Defactify Image Dataset, a benchmark collection consisting of 96,000 high-quality images and corresponding metadata. The authentic baseline images are sourced directly from the MS COCO dataset, while the synthetic counterparts are meticulously generated using five state-of-the-art AI models: Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3, and Midjourney v6. For the Falcon N5 training pipeline, the data is partitioned into a training set of 42,000 samples, a validation set of 9,000 samples, and a testing set of 45,000 samples, with annotations structuring both binary real-versus-fake detection and multi-class source identification.

[Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
