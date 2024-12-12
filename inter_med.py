i2020 = image1.clip(roi)
i2024 = image2.clip(roi)

stacked = i2020.addBands(i2024)


x_p = {
    "min": -40,
    "max": 40,
    "pallete": ["yellow", "orange"]
}

inverted_diff_vis_params = {
    'min': 5,  # Minimum value (adjust as needed)
    'max': 10,  # Maximum value (adjust as needed)
    'palette': ['red', 'blue'],  # Inverted color palette
}

diff = stacked.select('VV').subtract(stacked.select('VV_1')).abs()

blue_mask = diff.gte(20)
blue_masked = blue_mask.selfMask()
blue_border = blue_masked.reduceToVectors(
    geometryType='polygon',
    reducer=ee.Reducer.countEvery(),
)

connected_components = blue_mask.connectedComponents(
    connectedness=ee.Kernel.plus(1),  # 8-connected for adjacent pixels
    maxSize=128
)

clusters = connected_components.select('labels')

# Compute the Area of Each Cluster
clustered_area = clusters.multiply(ee.Image.pixelArea()).rename('area')

# Filter Clusters by Area (Remove Smaller Clusters)
min_area = 1000  # Minimum area in square meters, adjust based on your needs

# Mask out clusters smaller than the defined area
large_clusters = clustered_area.gte(min_area).And(clusters.mask())

# Optionally: Reduce the clusters to vector polygons for visualization
vectorized_clusters = large_clusters.reduceToVectors(
    reducer=ee.Reducer.countEvery(),
    scale=10,  # Sentinel-1 resolution, adjust if needed
    maxPixels=1e8
)


