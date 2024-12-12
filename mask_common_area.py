# SHP FILE things
sar_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
    .filter(ee.Filter.eq("instrumentMode", "IW")) \
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")
  )

# Load the uploaded shapefile as a FeatureCollection
shapefile = ee.FeatureCollection('users/npc203/BWD20_ChangesW')

# Print the first feature to verify
print(shapefile.first().getInfo())

# Visualize (if running in an environment like Colab or Jupyter)
import geemap

Map = geemap.Map()
Map.add_basemap('HYBRID')

vis_params = {
    "min": -20,
    "max": 0,
    "palette": ["yellow", "orange","black","blue", "green","red"]
}

image1 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7").select("VV")
image2 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3").select("VV")

# Get the masks of the images
mask1 = image1.mask()
mask2 = image2.mask()

# Compute the intersection of the masks
intersection_mask = mask1.And(mask2)

# Clip the images to the intersection
intersection_image1 = image1.updateMask(intersection_mask)
intersection_image2 = image2.updateMask(intersection_mask)


Map.addLayer(intersection_mask)
Map.addLayer(shapefile, {}, 'Shapefile')
Map.centerObject(shapefile, zoom=10)
Map
