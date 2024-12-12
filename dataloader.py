from torch.utils.data import Dataset

class DynamicImageDataset(Dataset):
    def __init__(self, image_sets, segmentation_masks, transform=None):
        """
        :param image_sets: List of image sequences (each sequence is a list of tensors).
        :param segmentation_masks: List of segmentation masks for the last images.
        :param transform: Optional transformations for images and masks.
        """
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


