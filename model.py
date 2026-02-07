import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TextureEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        backbone = models.resnet50(pretrained=True)
        backbone.fc = nn.Identity()   
        self.backbone = backbone

        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        z = self.embedding(features)
        z = F.normalize(z, dim=1)     
        return z
