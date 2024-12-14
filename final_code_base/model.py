import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DynamicImageDataset(Dataset):
    def __init__(self, image_sets, segmentation_masks, transform=None):
        self.image_sets = image_sets
        self.segmentation_masks = segmentation_masks
        self.transform = transform

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, idx):
        images = self.image_sets[idx]
        seg_mask = self.segmentation_masks[idx]

        if self.transform:
            images = [self.transform(img) for img in images]
            seg_mask = self.transform(seg_mask)

        return images, seg_mask


class UNetResNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, learning_rate=1e-3):
        super(UNetResNet, self).__init__()
        self.save_hyperparameters()

        # Load pretrained ResNet backbone
        resnet = resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and avg pooling
        
        # Decoder layers
        self.up1 = self.upconv(512, 256)
        self.up2 = self.upconv(256, 128)
        self.up3 = self.upconv(128, 64)
        self.up4 = self.upconv(64, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)  # Conv1
        enc2 = self.encoder[4](enc1)  # Layer1
        enc3 = self.encoder[5](enc2)  # Layer2
        enc4 = self.encoder[6](enc3)  # Layer3
        enc5 = self.encoder[7](enc4)  # Layer4

        # Decoder
        dec1 = self.up1(enc5)
        dec2 = self.up2(dec1 + enc4)
        dec3 = self.up3(dec2 + enc3)
        dec4 = self.up4(dec3 + enc2)

        return self.outc(dec4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

if __name__ == "__main__":
    model = UNetResNet(n_channels=2, n_classes=2)

    # Trainer
    trainer = pl.Trainer(max_epochs=5, accelerator="auto")
    trainer.fit(model, train_loader)