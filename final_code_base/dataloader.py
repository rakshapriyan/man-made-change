from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np


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

        images = [sm2020, sm2024]

        # TODO fix hack
        fin_images = torch.Tensor(np.array(images))
        last_element = fin_images[-1:, :, :]
        fin_images = torch.cat([fin_images, last_element], dim=0)
        return fin_images, seg_mask



class MultiChanDataset(Dataset):
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

        fin_image = torch.Tensor(np.array((sm2020, sm2024))).permute(0, 3, 1, 2)
        num,chan,w,h = fin_image.shape
        return fin_image.reshape(6, w, h), seg_mask

if __name__ == "__main__":
    dset = MultiChanDataset("./dataset_1")
    print(dset[0][0].shape)
