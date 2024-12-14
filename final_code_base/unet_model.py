import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models.segmentation import fcn_resnet50
import pytorch_lightning as pl
import os
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DynamicImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.len = int(sorted(os.listdir(data_path))[-1].split("_")[0])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sm2020 = np.load(os.path.join(self.data_path, f"{idx}_2020.npy"))
        sm2024 = np.load(os.path.join(self.data_path, f"{idx}_2024.npy"))
        seg_mask = np.load(os.path.join(self.data_path, f"{idx}_mask.npy"))

        if self.transform:
            images = [self.transform(img) for img in images]
            seg_mask = self.transform(seg_mask)

        return torch.Tensor([sm2020, sm2024]), seg_mask


class FlexibleImageUNet(pl.LightningModule):
    def __init__(self, n_channels_per_image, n_classes, learning_rate=1e-3):
        """
        UNet model for variable number of input images.

        Args:
            n_channels_per_image (int): Number of channels per image (e.g., 2 for VV/VH).
            n_classes (int): Number of segmentation classes.
            learning_rate (float): Learning rate for optimizer.
        """
        super(FlexibleImageUNet, self).__init__()
        self.save_hyperparameters()

        # Placeholder for ResNet backbone
        self.resnet = fcn_resnet50(num_classes=2)

        # Replace the first convolution layer to allow dynamic channels
        self.dynamic_conv = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Encoder layers (ResNet without the first conv layer)
        self.encoder_layers = nn.ModuleList(
            [
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.maxpool,
                self.resnet.layer1,
                self.resnet.layer2,
                self.resnet.layer3,
                self.resnet.layer4,
            ]
        )

        # Decoder layers
        self.up1 = self.upconv(512, 256)
        self.up2 = self.upconv(256, 128)
        self.up3 = self.upconv(128, 64)
        self.up4 = self.upconv(64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        self.learning_rate = learning_rate
        self.n_channels_per_image = n_channels_per_image

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, *images):
        """
        Forward pass with a variable number of images.

        Args:
            *images: List of images (B, C, H, W).

        Returns:
            Segmentation output.
        """
        # Combine images along the channel dimension
        x = torch.cat(images, dim=1)  # Shape: [B, total_channels, H, W]

        # Update the first convolution dynamically based on the total input channels
        total_channels = x.shape[1]
        if self.dynamic_conv.in_channels != total_channels:
            self.dynamic_conv = nn.Conv2d(
                in_channels=total_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ).to(device)

        # Encoder forward pass
        enc1 = self.dynamic_conv(x)  # First dynamic convolution
        enc1 = self.encoder_layers[0](enc1)  # BN
        enc1 = self.encoder_layers[1](enc1)  # ReLU
        enc1 = self.encoder_layers[2](enc1)  # MaxPool

        enc2 = self.encoder_layers[3](enc1)  # Layer1
        enc3 = self.encoder_layers[4](enc2)  # Layer2
        enc4 = self.encoder_layers[5](enc3)  # Layer3
        enc5 = self.encoder_layers[6](enc4)  # Layer4

        # Decoder
        dec1 = self.up1(enc5)
        dec1 = F.interpolate(dec1, size=enc4.shape[2:], mode="bilinear", align_corners=False)
        dec2 = self.up2(dec1 + enc4)
        dec2 = F.interpolate(dec2, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = self.up3(dec2 + enc3)
        dec3 = F.interpolate(dec3, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        dec4 = self.up4(dec3 + enc2)
        dec4 = F.interpolate(dec4, size=enc1.shape[2:], mode="bilinear", align_corners=False)

        out = self.outc(dec4)
        out = F.interpolate(
            out,
            size=(images[0].shape[2], images[0].shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return out

    def training_step(self, batch, batch_idx):
        *images, target = batch  # Expect multiple images and a target mask
        logits = self(*images)
        loss = F.cross_entropy(logits, target.long())
        self.log("train_loss", loss)
        print("Train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
