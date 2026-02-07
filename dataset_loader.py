import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TextureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------------------------------
# MAIN (testing + label extraction)
# -------------------------------------------------
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = TextureDataset(
        root_dir=r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\images",
        transform=transform
    )

    print("Total samples:", len(dataset))

    img, label = dataset[0]
    print("Image shape:", img.shape)
    print("Label:", label)

    # ✅ ADD THIS LINE HERE
    labels = [label for _, label in dataset.samples]

    print("Total labels:", len(labels))
