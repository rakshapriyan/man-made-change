import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSiameseSegmentor(nn.Module):
    def __init__(self, feature_extractor, segmentation_head):
        """
        :param feature_extractor: A CNN backbone for feature extraction (e.g., ResNet).
        :param segmentation_head: A decoder head for segmentation (e.g., U-Net or FPN).
        """
        super(DynamicSiameseSegmentor, self).__init__()
        self.feature_extractor = feature_extractor  # Shared CNN for all images
        self.segmentation_head = segmentation_head  # Segmentation head for the last image

    def forward(self, images):
        """
        :param images: List of aligned images (dynamic number). Last image is for segmentation.
        :return: similarity scores (features aggregated) and segmentation map for the last image.
        """
        # Extract features for all images
        features = [self.feature_extractor(image) for image in images]

        # Aggregate features for comparison (e.g., average pooling)
        aggregated_features = torch.mean(torch.stack(features[:-1]), dim=0)

        # Segment the last image
        last_image_features = features[-1]
        segmentation_map = self.segmentation_head(last_image_features)

        return aggregated_features, segmentation_map

# Example feature extractor (ResNet backbone)
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove FC layers

    def forward(self, x):
        return self.backbone(x)

# Example segmentation head (basic U-Net style)
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)

# Instantiate the model
from torchvision.models import resnet18

backbone = resnet18(pretrained=True)
feature_extractor = FeatureExtractor(backbone)
segmentation_head = SegmentationHead(in_channels=512, out_channels=1)

model = DynamicSiameseSegmentor(feature_extractor, segmentation_head)

# Test the model
dummy_images = [torch.rand(1, 3, 224, 224) for _ in range(5)]
aggregated_features, segmentation_map = model(dummy_images)

print("Aggregated Features Shape:", aggregated_features)
print("Segmentation Map Shape:", segmentation_map)

