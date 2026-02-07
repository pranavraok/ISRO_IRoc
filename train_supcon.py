import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import TextureDataset
from balanced_sampler import BalancedBatchSampler
from supcon_loss import SupConLoss
from model import TextureEncoder


def main():
    # -------------------------------
    # CONFIG (TARGET SETUP)
    # -------------------------------
    DATASET_PATH = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\split\train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 Strong negative pressure
    N_CLASSES = 20
    N_SAMPLES = 4

    # 🔥 Sharper separation
    TEMPERATURE = 0.07

    # 🔥 Final training
    EPOCHS = 10
    LR = 1e-4

    # -------------------------------
    # DATASET + SAMPLER
    # -------------------------------
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor()
    ])

    dataset = TextureDataset(DATASET_PATH, transform)
    labels = [label for _, label in dataset.samples]

    sampler = BalancedBatchSampler(
        labels=labels,
        n_classes=N_CLASSES,
        n_samples=N_SAMPLES
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,      # Windows-safe
        pin_memory=False
    )

    # -------------------------------
    # MODEL, LOSS, OPTIMIZER
    # -------------------------------
    model = TextureEncoder(embedding_dim=256).to(DEVICE)

    # 🔓 UNFREEZE BACKBONE (CRITICAL)
    for param in model.backbone.parameters():
        param.requires_grad = True

    criterion = SupConLoss(temperature=TEMPERATURE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            embeddings = model(images)
            loss = criterion(embeddings, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    torch.save(model.state_dict(), "supcon_encoder_final.pth")
    print("Training finished, Model saved as supcon_encoder_final.pth")


if __name__ == "__main__":
    main()
