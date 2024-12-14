import numpy as np
import matplotlib.pyplot as plt
import geemap
from skimage.draw import polygon
import cv2
import ee


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
        array1 = array1[: shape2[0], :]
    elif shape1[0] < shape2[0]:
        array2 = array2[: shape1[0], :]

    if shape1[1] > shape2[1]:
        array1 = array1[:, : shape2[1]]
    elif shape1[1] < shape2[1]:
        array2 = array2[:, : shape1[1]]

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


def ee_to_np(image, grid):
    p = geemap.ee_to_numpy(image, region=grid).squeeze()
    # p = np.array(image.sampleRectangle(region=grid).get("VV").getInfo())
    return p


def make_grids(roi, grid_size=0.015):
    bounds = roi.bounds().coordinates().get(0).getInfo()
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[2]

    x_steps = np.arange(min_x, max_x, grid_size)
    y_steps = np.arange(min_y, max_y, grid_size)

    grids = []
    for x in x_steps:
        for y in y_steps:
            grid_rect = ee.Geometry.Rectangle([x, y, x + grid_size, y + grid_size])
            grids.append(grid_rect)
    return grids
