# Falcon N5: AI vs. Authentic

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%23000000.svg?style=for-the-badge&logo=Seaborn&logoColor=white)
![Timm](https://img.shields.io/badge/Timm-red?style=for-the-badge)
![Neural Network](https://img.shields.io/badge/Neural%20Network-blue?style=for-the-badge)

<img width="504" height="259" alt="image" src="https://github.com/user-attachments/assets/2545e899-0d76-4cc4-9692-e636b88213d0" />

## Overview
Project Falcon N5 focuses on building a robust computer vision classifier capable of distinguishing between authentic, human-created images and AI-generated content. Leveraging the powerful Swin Transformer architecture (swin_base_patch4_window7_224), the model is trained to perform binary classification to tackle the growing challenge of digital forgery and artificial content verification. Because the training pipeline encounters an imbalanced distribution of real versus synthetic media, Falcon N5 incorporates a strategically weighted cross-entropy loss function to ensure the model penalizes misclassifications on minority "Real" data heavily, ultimately achieving high validation accuracy and reliable prediction confidence.

## Project Structure
- `ai-vs-original-image-classification.ipynb`: Jupyter notebook for training, evaluation, and visualization.
- `code/data_load.py`: Data loading utilities.
- `code/load_model.py`: Inference script for testing local images.
- `model/swin_ai_detector.pth`: Pre-trained model weights.


<img width="1089" height="740" alt="image" src="https://github.com/user-attachments/assets/8e12be3e-bd6f-4cf7-87c4-08023bd05491" />

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

**AI-vs-Real Dataset**
A balanced dataset for AI-generated vs Real image classification. 
This dataset is designed to help researchers, developers, and practitioners build and evaluate models that can distinguish between synthetic (AI-generated) and authentic (human-captured) images.

👉 **Link:** [Parveshiiii/AI-vs-Real Dataset](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real)
