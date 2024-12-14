from model import UNetResNet, DynamicImageDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        # Initialize datasets for train/val/test if needed
        self.train_dataset = DynamicImageDataset(image_sets, segmentation_masks, transform=None)
        # You can add validation or test datasets similarly
        self.val_dataset = DynamicImageDataset(image_sets, segmentation_masks, transform=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
model = UNetResNet()
data = SegmentationDataModule(batch_size=3)
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, data)