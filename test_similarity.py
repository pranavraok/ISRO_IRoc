import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import TextureDataset
from model import TextureEncoder


def main():
    # -------------------------------
    # CONFIG
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_PATH = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\split\train"
    TEST_PATH  = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\split\test"

    MODEL_PATH = "supcon_encoder_final.pth"   # 🔥 FINAL MODEL

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = TextureEncoder(embedding_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # -------------------------------
    # TRANSFORMS (NO AUGMENTATION!)
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # -------------------------------
    # DATASETS
    # -------------------------------
    train_dataset = TextureDataset(TRAIN_PATH, transform)
    test_dataset  = TextureDataset(TEST_PATH, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # -------------------------------
    # EXTRACT TRAIN EMBEDDINGS
    # -------------------------------
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(DEVICE)
            embeddings = model(images)

            train_embeddings.append(embeddings.cpu())
            train_labels.append(labels)

    train_embeddings = torch.cat(train_embeddings)
    train_labels = torch.cat(train_labels)

    # -------------------------------
    # TEST SIMILARITY (UNSEEN DATA)
    # -------------------------------
    same_class_sims = []
    diff_class_sims = []

    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(DEVICE)
            label = label.item()

            test_embedding = model(img)

            similarities = F.cosine_similarity(
                test_embedding.cpu(),
                train_embeddings
            )

            same = similarities[train_labels == label]
            diff = similarities[train_labels != label]

            same_class_sims.append(same.mean().item())
            diff_class_sims.append(diff.mean().item())

    # -------------------------------
    # RESULTS
    # -------------------------------
    avg_same = sum(same_class_sims) / len(same_class_sims)
    avg_diff = sum(diff_class_sims) / len(diff_class_sims)

    print("===== TEST RESULTS (UNSEEN DATA) =====")
    print(f"Average SAME-class similarity : {avg_same:.3f}")
    print(f"Average DIFF-class similarity : {avg_diff:.3f}")
    print("=====================================")


if __name__ == "__main__":
    main()
