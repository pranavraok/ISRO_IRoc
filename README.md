# ISRO Image Classification using Supervised Contrastive Learning

This project implements Supervised Contrastive (SupCon) learning for classifying planetary surface texture patterns from ISRO satellite imagery.

## Overview

The project uses Supervised Contrastive Learning to train a robust image encoder that can classify various surface textures and geological features from satellite images. The model learns discriminative representations by contrasting similar and dissimilar image pairs.

## Dataset

The **CV_Final_Dataset** consists of 28 different surface texture classes:
- Banded, Blotchy, Braided, Bright Dune, Bubbly, Bumpy
- Cracked, Crater, Crystalline, Dark Dune, Dotted, Fibrous
- Grid, Impact Ejecta, Meshed, Other, Polka-dotted, Porous
- Slope Streak, Spider, Sprinkled, Stratified, Studded, Swirly
- Swiss Cheese, Veined, Wrinkled, Zigzagged

### Dataset Structure
```
CV_Final_Dataset/
├── split/
│   ├── train/    (Training images)
│   └── test/     (Testing images)
```

### Download Dataset

**Google Drive Link:** [Add your Google Drive dataset link here](https://drive.google.com/drive/folders/1kSMA1vDD-5pkh6zhX-T3TP5TIJxjSFnK?usp=sharing)

Extract the dataset into the `CV_Final_Dataset/` directory.

## Project Files

### Core Training Files
- **`train_supcon.py`** - Main training script using Supervised Contrastive Loss
- **`supcon_loss.py`** - Implementation of Supervised Contrastive Loss
- **`model.py`** - Neural network architecture (encoder + projection head)

### Data Handling
- **`dataset_loader.py`** - Custom dataset loader for image pairs and augmentations
- **`balanced_sampler.py`** - Balanced sampling to ensure equal class representation
- **`split_dataset.py`** - Utility to split dataset into train/test sets

### Inference & Testing
- **`infer_pair.py`** - Inference on image pairs
- **`verify_similarity.py`** - Verify similarity scores between images
- **`test_similarity.py`** - Test similarity computations
- **`loss_test.py`** - Test loss calculations
- **`sampler_test.py`** - Test sampler functionality

### Pre-trained Model
- **`supcon_encoder_final.pth`** - Pre-trained SupCon encoder weights

## Installation

### Requirements
```bash
pip install torch torchvision
pip install numpy pillow
pip install matplotlib scipy
```

### Set Up Environment
```bash
# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train_supcon.py
```

**Key Training Parameters:**
- Batch size: 64
- Learning rate: 0.0005
- Temperature: 0.07
- Epochs: 100
- Optimizer: Adam

### Inference on Image Pairs

```bash
python infer_pair.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

### Verify Similarity

```bash
python verify_similarity.py
```

This script computes and displays similarity scores between image pairs.

### Test Similarity Metrics

```bash
python test_similarity.py
```

## Model Architecture

The model consists of:
1. **Encoder**: ResNet-based or custom CNN backbone
2. **Projection Head**: Non-linear projection to representation space
3. **Loss Function**: Supervised Contrastive Loss (NT-Xent variant)

## Training Details

### Supervised Contrastive Learning
- Positive pairs: Different augmentations of same image
- Negative pairs: Images from different classes
- Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)

### Data Augmentation
- Random rotations
- Random crops
- Color jittering
- Horizontal/Vertical flips

## Results

The trained model `supcon_encoder_final.pth` achieves strong performance on the 28-class classification task by learning robust texture-specific representations.

## Evaluation Metrics

- **Accuracy**: Classification accuracy on test set
- **Similarity Score**: Cosine similarity between same-class images
- **Contrastive Loss**: Training and validation loss curves

## File Structure Summary

```
ISRO IRoc/
├── Training & Loss
│   ├── train_supcon.py              # Main training script using SupCon loss
│   ├── supcon_loss.py               # Supervised Contrastive Loss implementation
│   └── loss_test.py                 # Loss function testing utilities
│
├── Model Architecture
│   └── model.py                     # Neural network model definition (encoder + projection)
│
├── Data Pipeline
│   ├── dataset_loader.py            # Dataset loading and data augmentation
│   ├── balanced_sampler.py          # Balanced batch sampling for classes
│   └── split_dataset.py             # Utility to split dataset into train/test
│
├── Inference & Evaluation
│   ├── infer_pair.py                # Inference on image pairs
│   ├── verify_similarity.py         # Verify and display similarity scores
│   ├── test_similarity.py           # Test similarity computations
│   └── sampler_test.py              # Test sampler functionality
│
├── Model Weights
│   └── supcon_encoder_final.pth     # Pre-trained SupCon encoder weights
│
├── Dataset
│   └── CV_Final_Dataset/            # Image dataset (in .gitignore)
│       └── split/
│           ├── train/               # Training images by class
│           └── test/                # Test images by class
│
├── Configuration
│   ├── README.md                    # This file
│   ├── .gitignore                   # Git ignore configuration
│   └── __pycache__/                 # Python cache (in .gitignore)
```

## How to Use Pre-trained Model

```python
import torch
from model import encoder_model

# Load pre-trained weights
model = encoder_model()
model.load_state_dict(torch.load('supcon_encoder_final.pth'))
model.eval()

# Extract features from images
with torch.no_grad():
    embeddings = model(image_batch)
```

## Troubleshooting

### Dataset not found
- Ensure `CV_Final_Dataset/` exists with `split/train/` and `split/test/` subdirectories
- Check that subdirectories contain class folders with images

### CUDA out of memory
- Reduce batch size in training script
- Use `device = 'cpu'` for CPU-only training (slower)

### Model loading errors
- Ensure `supcon_encoder_final.pth` matches model architecture
- Check PyTorch version compatibility

## References

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. ICML.
- Khosla, P., et al. (2020). Supervised Contrastive Learning. NeurIPS.

**Last Updated:** February 2026
