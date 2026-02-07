import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import TextureEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "supcon_encoder_final.pth"

IMG1_PATH = r"C:\Users\prana\Downloads\Case1-7.jpeg"
IMG2_PATH = r"C:\Users\prana\Downloads\Case1-8.jpeg"

model = TextureEncoder(256).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    z1 = model(load_image(IMG1_PATH))
    z2 = model(load_image(IMG2_PATH))

    similarity = F.cosine_similarity(z1, z2).item()

print(f"Cosine similarity: {similarity:.3f}")

if similarity >= 0.85:
    print("Very similar textures")
elif similarity >= 0.60:
    print("Moderately similar textures")
else:
    print("Different textures")
