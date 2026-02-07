import torch
from supcon_loss import SupConLoss

loss_fn = SupConLoss()

# fake normalized embeddings
features = torch.randn(16, 256)
features = torch.nn.functional.normalize(features, dim=1)

labels = torch.tensor(
    [19]*4 + [20]*4 + [13]*4 + [0]*4
)

loss = loss_fn(features, labels)
print("SupCon loss:", loss.item())
