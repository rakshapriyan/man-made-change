import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class CustomFCN(pl.LightningModule):
    def __init__(self, num_channels=6, num_classes=1, learning_rate=1e-3):
        super(CustomFCN, self).__init__()
        self.save_hyperparameters()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Load pretrained FCN ResNet50 model
        self.model = fcn_resnet50(pretrained=True)

        # Modify the first convolution layer to accept `num_channels` input
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Modify the classifier to output the desired number of classes
        self.model.classifier[4] = nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # Forward pass through the model
        output = self.model(x)
        return output['out']

    def training_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self.forward(images)
        loss = F.binary_cross_entropy_with_logits(predictions, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self.forward(images)
        loss = F.binary_cross_entropy_with_logits(predictions, masks)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Example Dataset Class
class ExampleDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (6, H, W)
        mask = self.masks[idx]    # Shape: (1, H, W)
        return image, mask

# Example usage (assuming you have training and validation datasets)
def main():
    # Example data tensors (replace with your actual data)
    train_images = torch.rand(100, 6, 100, 113)  # 100 samples, 6 channels, 100x113
    train_masks = torch.rand(100, 1, 100, 113)   # Binary masks

    val_images = torch.rand(20, 6, 100, 113)     # 20 samples
    val_masks = torch.rand(20, 1, 100, 113)

    train_dataset = ExampleDataset(train_images, train_masks)
    val_dataset = ExampleDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model initialization
    model = CustomFCN(num_channels=6, num_classes=1, learning_rate=1e-3)

    # Trainer
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1 if torch.cuda.is_available() else None)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
