import numpy as np
import matplotlib.pyplot as plt
import geemap
from skimage.draw import polygon
import cv2

def ext_feat_list(feat_col):
  """Extract features from FeatureCollection"""
  num_feats = feat_col.size().getInfo()
  features = []
  f_raw = feat_col.toList(num_feats)
  for i in range(num_feats):
    features.append(ee.Feature(f_raw.get(i)))
  
  return features

def latlon_to_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, img_height, img_width):
    x = (lon - min_lon) / (max_lon - min_lon) * img_width
    y = (max_lat - lat) / (max_lat - min_lat) * img_height
    return int(x), int(y)

def pixel_to_latlon(x, y, min_lat, max_lat, min_lon, max_lon, img_height, img_width):
    lon = x / img_width * (max_lon - min_lon) + min_lon
    lat = max_lat - y / img_height * (max_lat - min_lat)
    return lat, lon

def trim_arrays(array1, array2):
    # Get the shapes of the arrays
    shape1 = array1.shape
    shape2 = array2.shape
    
    # Check which array is larger and trim accordingly
    if shape1[0] > shape2[0]:
        array1 = array1[:shape2[0], :]
    elif shape1[0] < shape2[0]:
        array2 = array2[:shape1[0], :]
    
    if shape1[1] > shape2[1]:
        array1 = array1[:, :shape2[1]]
    elif shape1[1] < shape2[1]:
        array2 = array2[:, :shape1[1]]
    
    return array1, array2

def polygon_to_mask(image_size, polygon_coords):
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)  # Note: height first in OpenCV

    # Clip coordinates to stay within bounds
    clipped_coords = [
        (max(0, min(x, image_size[0] - 1)), max(0, min(y, image_size[1] - 1)))
        for x, y in polygon_coords
    ]

    # Convert to an integer numpy array (required for OpenCV)
    clipped_coords = np.array(clipped_coords, dtype=np.int32)

    # Draw the polygon if it has at least 3 points
    if len(clipped_coords) > 2:
        cv2.fillPoly(mask, [clipped_coords], color=1)  # Fill the polygon with 1s

    return mask


def ee_to_np(image,grid):
  p = geemap.ee_to_numpy(image, region=grid)
  # p = np.array(image.sampleRectangle(region=grid).get("VV").getInfo())
  return p

# image1 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7").select("VV")
# image2 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3").select("VV")

# a1 = [ee_to_np(image1, grid) for grid in grids]
# a2 = [ee_to_np(image2, grid) for grid in grids]

image_sets = []
segmentation_masks = []

grids = make_grids(roi)
for c in range(len(grids)):
    print("C:", c)
    grid = grids[c]
    sm2020, sm2024 = trim_arrays(a1[c].squeeze(), a2[c].squeeze())

    # GET GRID BOUNDS
    bounds = grid.bounds().coordinates().get(0).getInfo()
    tx,ty = bounds[0]
    bx, by = bounds[2]
    
    segments = []

    # ACTUAL BOUNDS
    for blob in ext_feat_list(changeW.filterBounds(grid)):
      blob_coords = np.array(blob.geometry().getInfo()["coordinates"]).squeeze()

      trans_blob_coords = []
      for coords in blob_coords:
          trans_blob_coords.append(latlon_to_pixel(*coords,tx,bx,ty,by,*sm2020.shape[::-1]))
      
      segments.append(trans_blob_coords)

    if segments:
      print("segments", segments[0])

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
      
      image_sets.append(curr_img_set)
      segmentation_masks.append(c_mask)
      
      plt.imshow(c_mask, cmap='jet', alpha=0.5)
      plt.tight_layout()
      plt.show()