
import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from model import model
from dataloader import DynamicImageDataset, DataLoader
from torch.utils.data import DataLoader

# Optimizer and device setup
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

similarity_loss_fn = nn.CosineEmbeddingLoss()

def similarity_loss(aggregated_features, target):
    """
    :param aggregated_features: Aggregated features of the first N-1 images.
    :param target: Ground truth similarity score (1 for similar, -1 for dissimilar).
    """
    batch_size = aggregated_features.shape[0]
    target = torch.ones(batch_size).to(aggregated_features.device)  # Assuming similar inputs
    return similarity_loss_fn(aggregated_features, torch.zeros_like(aggregated_features), target)

def dice_loss(pred, target, smooth=1):
    """
    :param pred: Predicted segmentation map.
    :param target: Ground truth segmentation map.
    :param smooth: Smoothing factor for numerical stability.
    """
    pred = pred.sigmoid()  # Apply sigmoid for binary maps
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

segmentation_loss_fn = nn.BCEWithLogitsLoss()

def segmentation_loss(pred, target):
    """
    :param pred: Predicted segmentation map.
    :param target: Ground truth segmentation map.
    """
    bce_loss = segmentation_loss_fn(pred, target)
    dice = dice_loss(pred, target)
    return bce_loss + dice



# Training loop
def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    epoch_similarity_loss = 0
    epoch_segmentation_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        # Load batch data
        images, seg_target = batch  # images: list of tensors, seg_target: segmentation map
        images = [img.to(device) for img in images]
        seg_target = seg_target.to(device)

        # Forward pass
        optimizer.zero_grad()
        aggregated_features, seg_pred = model(images)

        # Compute losses
        sim_loss = similarity_loss(aggregated_features, target=None)  # Adjust as needed
        seg_loss = segmentation_loss(seg_pred, seg_target)

        # Combine losses
        total_loss = sim_loss + seg_loss

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()

        # Log losses
        epoch_similarity_loss += sim_loss.item()
        epoch_segmentation_loss += seg_loss.item()

    # Average losses
    avg_sim_loss = epoch_similarity_loss / len(dataloader)
    avg_seg_loss = epoch_segmentation_loss / len(dataloader)
    print(f"Epoch {epoch} | Similarity Loss: {avg_sim_loss:.4f} | Segmentation Loss: {avg_seg_loss:.4f}")


dataset = DynamicImageDataset(image_sets, segmentation_masks, transform=None)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Training for multiple epochs
for epoch in range(1, 11):  # Train for 10 epochs
    train_epoch(model, dataloader, optimizer, epoch)
