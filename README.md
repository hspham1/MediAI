# MediAI
Medical Imaging Models

This repository contains a pix2pix Generative Adversarial Network (GAN) model for medical image-to-image translation, specifically designed to generate MRI images from input CT scan images.

## Overview

MediAI uses a conditional GAN (pix2pix) architecture to perform medical image translation of CT to MRI images by using paired training data.s

## Features

- **Pix2Pix Architecture**: Conditional GAN with encoder-decoder generator and PatchGAN discriminator
- **Medical Image Translation**: CT to MRI image synthesis
- **FastAPI Backend**: RESTful API for model inference
- **Pre-trained Weights**: Includes pre-trained generator and discriminator models
- **Image Normalization**: Automatic normalization pipeline for input/output


## Model Architecture

### Generator
- **Encoder-Decoder Architecture**: Convolutional encoder followed by transposed convolutional decoder
- **Input**: 3-channel CT scan image
- **Output**: 3-channel MRI image (256x256)
- **Features**: Batch normalization, LeakyReLU activation, Tanh output

### Discriminator
- **PatchGAN Architecture**: Classifies image patches as real or fake
- **Input**: Concatenated CT and MRI images (6 channels)
- **Output**: Binary classification (real/fake)
- **Features**: Batch normalization, LeakyReLU activation, Sigmoid output

## Installation

### Requirements
- Python 3.8+
- PyTorch
- FastAPI
- torchvision

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MediAI.git
cd MediAI

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the FastAPI Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Using the Model

```python
import torch
from models.GANs import Generator

# Load pre-trained generator
generator = Generator()
generator.load_state_dict(torch.load('models/pretrained_weights/best_ct-mri_generator.pth'))
generator.eval()

# Inference
with torch.no_grad():
    ct_image = torch.randn(1, 3, 256, 256)  # Example CT scan
    mri_image = generator(ct_image)
```

## API Endpoints

- `GET /`: Health check
- `POST /predict`: Generate MRI from CT scan image
- `GET /docs`: Interactive API documentation (Swagger UI)

## Training

To train the model from scratch:

```bash
python scripts/train.py --epochs 500 --batch-size 16
```

## Pre-trained Models

Pre-trained weights are available in `models/pretrained_weights/`:
- `best_ct-mri_generator.pth`: Best performing generator
- `best_ct-mri_discriminator.pth`: Corresponding discriminators