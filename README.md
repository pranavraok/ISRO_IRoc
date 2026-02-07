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

**Google Drive Link:** [Download CV_Final_Dataset](https://drive.google.com/drive/folders/1kSMA1vDD-5pkh6zhX-T3TP5TIJxjSFnK?usp=sharing)

After downloading, extract the dataset into the project root directory as `CV_Final_Dataset/`.

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

The training script uses the following default parameters:
- **Batch size**: 64
- **Learning rate**: 0.0005
- **Temperature**: 0.07 (for contrastive loss)
- **Epochs**: 10
- **Optimizer**: Adam
- **Data Augmentation**: Random crops, flips, rotations, color jitter

### Inference on Image Pairs

```bash
python infer_pair.py
```

This script loads two images and computes their cosine similarity using the pre-trained encoder.

### Verify Similarity

```bash
python verify_similarity.py
```

Computes and displays similarity scores between multiple image pairs from the test set. This script was used to generate the reported metrics (avg similarity: 0.88, difference: 0.21).

### Test Similarity Metrics

```bash
python test_similarity.py
```

Test utility for validating similarity computation functions.

## Model Architecture

The model follows the Supervised Contrastive Learning framework:

1. **Encoder (Backbone)**: ResNet-based CNN that extracts feature representations from images
2. **Projection Head**: Multi-layer perceptron (MLP) that projects features to a normalized embedding space
3. **Loss Function**: Supervised Contrastive Loss (NT-Xent) that pulls same-class samples together and pushes different-class samples apart

The encoder learns to produce embeddings where cosine similarity correlates with semantic similarity between planetary surface textures.

## Training Details

### Supervised Contrastive Learning Approach

Unlike unsupervised contrastive methods, this implementation leverages class labels:

- **Positive pairs**: Images from the same texture class (any augmentation)
- **Negative pairs**: Images from different texture classes
- **Loss function**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Temperature parameter**: Controls the concentration of the softmax distribution (τ = 0.07)

### Data Augmentation Pipeline

To improve model robustness and generalization:
- Random horizontal and vertical flips
- Random rotations (0°, 90°, 180°, 270°)
- Random resized crops
- Color jittering (brightness, contrast, saturation)
- Normalization using ImageNet statistics

## Results

The trained model `supcon_encoder_final.pth` achieves strong performance on the 28-class classification task by learning robust texture-specific representations.

### Performance Metrics

- **Average Cosine Similarity**: 0.88 (for same-class image pairs)
- **Similarity Difference**: 0.21 (between same-class and different-class pairs)
- **Model**: Supervised Contrastive Learning with ResNet encoder

The high average cosine similarity (0.88) indicates that the model effectively learns to group similar surface textures together in the embedding space, while maintaining a significant margin (0.21) between similar and dissimilar pairs.

## Evaluation Metrics

- **Cosine Similarity**: Measures similarity between embeddings (higher is better for same-class pairs)
- **Contrastive Loss**: NT-Xent loss for training convergence
- **Embedding Quality**: Separation between positive and negative pairs in feature space

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
import torch.nn.functional as F
from model import SupConEncoder  # adjust based on your actual model class
from torchvision import transforms
from PIL import Image

# Load pre-trained weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SupConEncoder()
model.load_state_dict(torch.load('supcon_encoder_final.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and process images
img1 = transform(Image.open('image1.jpg')).unsqueeze(0).to(device)
img2 = transform(Image.open('image2.jpg')).unsqueeze(0).to(device)

# Extract embeddings
with torch.no_grad():
    emb1 = model(img1)
    emb2 = model(img2)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(emb1, emb2)
    print(f'Similarity: {similarity.item():.4f}')
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

- **Supervised Contrastive Learning**: Khosla, P., et al. (2020). "Supervised Contrastive Learning." NeurIPS.
- **SimCLR Framework**: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML.
- **ResNet Architecture**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

## Acknowledgments

This project was developed for classifying planetary surface textures from ISRO satellite imagery using state-of-the-art contrastive learning techniques.

## Repository

**GitHub**: [pranavraok/ISRO_IRoc](https://github.com/pranavraok/ISRO_IRoc)

---

**Last Updated:** February 2026
