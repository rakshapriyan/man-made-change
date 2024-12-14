from res50_vv_vh import CustomFCN
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import MultiChanDataset
import pytorch_lightning as pl
import torch
import os

torch.set_float32_matmul_precision('medium')

# split the train set into two
seed = torch.Generator().manual_seed(42)

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset,batch_size=32, transform=None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        full_dataset = self.dataset
        train_set_size = int(len(full_dataset) * 0.8)
        valid_set_size = len(full_dataset) - train_set_size

        train_set, valid_set = random_split(full_dataset, [train_set_size, valid_set_size], generator=seed)

        self.train_dataset = train_set
        self.val_dataset = valid_set

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15,persistent_workers=True)

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",  # Monitor the 'train_loss' to decide the best model
    dirpath="checkpoints/",  # Directory to save checkpoints
    filename="model-{epoch:02d}-{train_loss:.4f}",  # Format the checkpoint filename
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=1,  # Save checkpoint every epoch
)

# Check if a checkpoint exists
last_checkpoint = None
curr_folder = "res_checkpoints/"
if os.path.exists("res_checkpoints/"):
    checkpoint_files = [f for f in os.listdir(curr_folder) if f.endswith(".ckpt")]
    if checkpoint_files:
        last_checkpoint = os.path.join(curr_folder, sorted(checkpoint_files)[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

if last_checkpoint:
    model = CustomFCN.load_from_checkpoint(last_checkpoint)
else:
    model = CustomFCN(num_channels=6, num_classes=1, learning_rate=1e-3)

data = SegmentationDataModule(dataset=MultiChanDataset("./dataset_1"),batch_size=1)
trainer = pl.Trainer(
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    max_epochs=1,
    enable_model_summary = True,
)
trainer.fit(model, data)