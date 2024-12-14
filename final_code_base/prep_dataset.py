import ee
import geemap.core as geemap
from utils import (
    make_grids,
    ee_to_np,
    ext_feat_list,
    polygon_to_mask,
    latlon_to_pixel,
    trim_arrays,
)
import numpy as np
import matplotlib.pyplot as plt
import cv2

ee.Authenticate()
ee.Initialize(project="ee-rakshapriyanraksha")

lt =  76.6773, 28.3604
rb =  76.9972, 28.1123
grid_size = 0.01


image1 = ee.Image(
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7"
)
image2 = ee.Image(
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3"
)
changeW = ee.FeatureCollection("users/npc203/BWD20_Changes")
roi = ee.Geometry.Rectangle([*lt, *rb])

image_sets = []
segmentation_masks = []

a = 0
grids = make_grids(roi, grid_size)
for c in range(len(grids)):
    print("C:", c,"A", a)
    try:
        grid = grids[c]
        sm2020 = ee_to_np(image1, grid)[:113, :100]
        sm2024 = ee_to_np(image2, grid)[:113, :100]

        # GET GRID BOUNDS
        bounds = grid.bounds().coordinates().get(0).getInfo()
        tx, ty = bounds[0]
        bx, by = bounds[2]

        # ACTUAL BOUNDS
        segments = []
        for blob in ext_feat_list(changeW.filterBounds(grid)):
            blob_coords = np.array(blob.geometry().getInfo()["coordinates"]).squeeze()

            trans_blob_coords = []
            for coords in blob_coords:
                trans_blob_coords.append(
                    latlon_to_pixel(*coords, tx, bx, ty, by, sm2020.shape[0], sm2020.shape[1])
                )

            segments.append(trans_blob_coords)

        if segments:
            print("segments", len(segments))

        c_mask = None
        for s in segments:
            mask = polygon_to_mask(sm2024.shape, s)

            if c_mask is None:
                c_mask = mask

            c_mask = np.maximum(mask, c_mask)

        if c_mask is not None:
            # PLOT
            # fig, axes = plt.subplots(1, 4)

            s = sm2020 - sm2024

            curr_img_set = []
            for i, arr in enumerate([sm2020, sm2024, s]):
                final_image = cv2.flip(np.rot90(arr, 1), 1)
                curr_img_set.append(final_image)
                # axes[i].imshow(final_image, cmap="gray")
                # axes[i].axis("off")  # Hide axes

            # Remove s,s
            image_sets.append(curr_img_set[:2])
            segmentation_masks.append(c_mask)

            np.save(f"dataset_2/{a}_2020.npy", curr_img_set[0])
            np.save(f"dataset_2/{a}_2024.npy", curr_img_set[1])
            np.save(f"dataset_2/{a}_mask.npy", c_mask)
            a += 1

        #   plt.imshow(c_mask, cmap='jet', alpha=0.5)
        #   plt.tight_layout()
        #   plt.show()
        #   break
    except Exception as e:
        print(e)