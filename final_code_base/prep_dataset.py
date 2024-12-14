import ee
import geemap.core as geemap
from utils import make_grids, ee_to_np, ext_feat_list, polygon_to_mask, latlon_to_pixel
import numpy as np
import matplotlib.pyplot as plt
import cv2

ee.Authenticate()
ee.Initialize(project='loyal-curve-440318-m8')

lt = [76.83528873955892, 28.200573041775723]
rb = [76.90497295345911, 28.167739165270472]
grid_size = 0.015


image1 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7").select("VV")
image2 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3").select("VV")
changeW = ee.FeatureCollection('users/npc203/BWD20_ChangesW')
roi = ee.Geometry.Rectangle([*lt, *rb])

image_sets = []
segmentation_masks = []

grids = make_grids(roi,grid_size)
for c in range(len(grids)):
    print("Grid:", c)
    grid = grids[c]
    sm2020 = ee_to_np(image1, grid)[:100, :113]
    sm2024 = ee_to_np(image2, grid)[:100, :113]

    # GET GRID BOUNDS
    bounds = grid.bounds().coordinates().get(0).getInfo()
    tx,ty = bounds[0]
    bx, by = bounds[2]

    # ACTUAL BOUNDS
    segments = []
    for blob in ext_feat_list(changeW.filterBounds(grid)):
      blob_coords = np.array(blob.geometry().getInfo()["coordinates"]).squeeze()

      trans_blob_coords = []
      for coords in blob_coords:
        trans_blob_coords.append(latlon_to_pixel(*coords,tx,bx,ty,by,*sm2020.shape))

      segments.append(trans_blob_coords)

    if segments:
      print("segments", len(segments))

    c_mask = None
    for s in segments:
        mask = polygon_to_mask(sm2024.shape, s)

        if c_mask is None:
            c_mask = mask

        c_mask = np.maximum(mask,c_mask)

    if c_mask is not None:
      # PLOT
      fig, axes = plt.subplots(1, 4)

      s = sm2020 - sm2024

      curr_img_set = []
      for i, arr in enumerate([sm2020, sm2024, s, s]):
          final_image = cv2.flip(np.rot90(arr,1), 1)
          curr_img_set.append(final_image)
          axes[i].imshow(final_image, cmap='gray')
          axes[i].axis('off')  # Hide axes

      # Remove s,s 
      image_sets.append(curr_img_set[:2])
      segmentation_masks.append(c_mask)

      plt.imshow(c_mask, cmap='jet', alpha=0.5)
      plt.tight_layout()
      plt.show()