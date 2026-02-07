from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import TextureDataset
from balanced_sampler import BalancedBatchSampler

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\images"
N_CLASSES = 4
N_SAMPLES = 4

# -------------------------------
# DATASET
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = TextureDataset(
    root_dir=DATASET_PATH,
    transform=transform
)

labels = [label for _, label in dataset.samples]

# -------------------------------
# SAMPLER + LOADER
# -------------------------------
sampler = BalancedBatchSampler(
    labels=labels,
    n_classes=N_CLASSES,
    n_samples=N_SAMPLES
)

loader = DataLoader(dataset, batch_sampler=sampler)

# -------------------------------
# TEST ONE BATCH
# -------------------------------
images, targets = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch labels:", targets)
