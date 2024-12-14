import torch
from torchvision import transforms, datasets, models
import pytorch_lightning as pl


class Res50Seg(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch
        # print(images.shape, targets.shape)
        outputs = self.model(images)["out"]
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        # print("Loss:", loss)
        self.log("train_loss", loss)
        return loss

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)