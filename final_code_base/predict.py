from res50_model import Res50Seg
import geemap.core as geemap
import os
import numpy as np
import ee
from utils import make_grids, ee_to_np
import torch
import matplotlib.pyplot as plt

ee.Authenticate()
ee.Initialize(project="ee-ahilaadithya01")

lt = [76.83528873955892, 28.200573041775723]
rb = [76.90497295345911, 28.167739165270472]
grid_size = 0.01

last_checkpoint = None
if os.path.exists("checkpoints/"):
    checkpoint_files = [f for f in os.listdir("checkpoints/") if f.endswith(".ckpt")]
    if checkpoint_files:
        # checkpoint_files.sort(key=os.path.getctime)
        last_checkpoint = os.path.join("checkpoints/", sorted(checkpoint_files)[-1])

model = Res50Seg.load_from_checkpoint(last_checkpoint).cuda()

roi = ee.Geometry.Rectangle([*lt, *rb])

image1 = ee.Image(
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7"
).select("VV")
image2 = ee.Image(
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3"
).select("VV")

def img_to_inp(images):
    fin_images = torch.Tensor(np.array(images))
    last_element = fin_images[-1:, :, :]
    fin_images = torch.cat([fin_images, last_element], dim=0)
    return fin_images.cuda().unsqueeze(0)

for grid in make_grids(roi):
    sm2020 = ee_to_np(image1, grid)[:113, :100]
    sm2024 = ee_to_np(image2, grid)[:113, :100]
    imgs = img_to_inp([sm2020,sm2024])
    output = model(imgs)["out"]
    mask = torch.argmax(output, dim=1).squeeze(0).cpu()
    plt.imshow(mask, cmap='gray', alpha=0.5)
    plt.show()