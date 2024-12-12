import ee
import geemap


def mask_edge(image):
  edge = image.lt(-30.0)
  masked_image = image.mask().And(edge.Not())
  return image.updateMask(masked_image)

roi = ee.Geometry.Rectangle([*lt[::-1], *rb[::-1]])

# Load Sentinel-1 SAR collection
sar_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
    .filter(ee.Filter.eq("instrumentMode", "IW")) \
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")
  )

print(sar_collection.size().getInfo())

# Define two time periods
pre = ee.Filter.date("2020-05-16T00:52:24Z","2020-05-16T00:52:49Z")
post = ee.Filter.date("2024-05-19T00:52:34Z","2024-05-19T00:52:59Z")

roi1 = sar_collection.filter(ee.Filter.eq('relatveOrbitNumber_start', 136))
roi2 = 

# Calculate mean backscatter for each time period
mean_pre = sar_collection.filter(pre).mean().clip(roi)
mean_post = sar_collection.filter(post).mean().clip(roi)

# Separate VV and VH bands
vv_pre = mean_pre.select("VV")
vv_post = mean_post.select("VV")
vh_pre = mean_pre.select("VH")
vh_post = mean_post.select("VH")

# Calculate differences
vv_diff = vv_post.subtract(vv_pre).rename("VV_Difference")
vh_diff = vh_post.subtract(vh_pre).rename("VH_Difference")

# Combine VV and VH differences
combined_diff = vv_diff.addBands(vh_diff)

# Visualize results
map = geemap.Map(center=lt, zoom=12)
map.add_basemap('HYBRID')
# map.addLayer(vv_diff, None,"VV Difference")
# map.addLayer(vh_diff, None, "VH Difference")

vis_params = {
    "min": -20,
    "max": 0,
    "palette": ["yellow", "orange","black","blue", "green","red"]
}
map.addLayer(vv_pre, vis_params, "Combined Difference (RGB)")

# edges = ee.Algorithms.CannyEdgeDetector(image=vv_pre, threshold=1, sigma=2)
# map.addLayer(edges, None, "Canny Edges")

map.addLayer(roi, {}, "ROI")
map.addLayerControl()  # Adds a layer control panel
map
