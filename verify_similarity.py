import torch
import torch.nn.functional as F
from torchvision import transforms

from dataset_loader import TextureDataset
from model import TextureEncoder

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# LOAD MODEL
# -------------------------------
model = TextureEncoder(embedding_dim=256).to(DEVICE)
model.eval()

# NOTE: using trained weights already in memory
# (If you restart kernel later, we'll add checkpoint loading)

# -------------------------------
# DATASET
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = TextureDataset(DATASET_PATH, transform)

# -------------------------------
# PICK SAMPLES
# -------------------------------
# manually choose indices for clarity
img1, label1 = dataset[0]
img2, label2 = dataset[1]

# find an image with a DIFFERENT label
for i in range(len(dataset)):
    img3, label3 = dataset[i]
    if label3 != label1:
        break

# -------------------------------
# COMPUTE EMBEDDINGS
# -------------------------------
with torch.no_grad():
    z1 = model(img1.unsqueeze(0).to(DEVICE))
    z2 = model(img2.unsqueeze(0).to(DEVICE))
    z3 = model(img3.unsqueeze(0).to(DEVICE))

# -------------------------------
# COSINE SIMILARITY
# -------------------------------
sim_same = F.cosine_similarity(z1, z2).item()
sim_diff = F.cosine_similarity(z1, z3).item()

print(f"Same class similarity: {sim_same:.3f}")
print(f"Different class similarity: {sim_diff:.3f}")
