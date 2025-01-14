from unet_model import FlexibleImageUNet
from dataloader import DynamicImageDataset
from res50_model import Res50Seg
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import os
from torch.utils.data import random_split, DataLoader

# split the train set into two
seed = torch.Generator().manual_seed(42)

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        full_dataset = DynamicImageDataset("./dataset")
        train_set_size = int(len(full_dataset) * 0.8)
        valid_set_size = len(full_dataset) - train_set_size

        train_set, valid_set = random_split(full_dataset, [train_set_size, valid_set_size], generator=seed)

        self.train_dataset = train_set
        # You can add validation or test datasets similarly
        self.val_dataset = valid_set

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",  # Monitor the 'train_loss' to decide the best model
    dirpath="checkpoints/",  # Directory to save checkpoints
    filename="model-{epoch:02d}-{train_loss:.4f}",  # Format the checkpoint filename
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=1,  # Save checkpoint every epoch
)

# Check if a checkpoint exists
last_checkpoint = None
if os.path.exists("checkpoints/"):
    checkpoint_files = [f for f in os.listdir("checkpoints/") if f.endswith(".ckpt")]
    if checkpoint_files:
        last_checkpoint = os.path.join("checkpoints/", sorted(checkpoint_files)[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

if last_checkpoint:
    model = Res50Seg.load_from_checkpoint(last_checkpoint)
else:
    # model = FlexibleImageUNet(2,2)
    model = Res50Seg()
    # model = LitSegmentation()

data = SegmentationDataModule(batch_size=5)
trainer = pl.Trainer(
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    max_epochs=20,
    enable_model_summary = True,
)
trainer.fit(model, data)

#  [64, 3, 7, 7], expected input[5, 2, 100, 113]
#  [64, 3, 7, 7], expected input[5, 2, 100, 113]
