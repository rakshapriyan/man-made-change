# SHP FILE things
sar_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
    .filter(ee.Filter.eq("instrumentMode", "IW")) \
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")
  )
    
lt = 28.3604, 76.6773
rb = 28.1123, 76.9972

roi = ee.Geometry.Rectangle([*lt[::-1], *rb[::-1]])

# Load the uploaded shapefile as a FeatureCollection
shapefile = ee.FeatureCollection('users/npc203/BWD20_ChangesW')

Map = geemap.Map()
Map.add_basemap('HYBRID')

vis_params = {
    "min": -20,
    "max": 0,
    "palette": ["yellow", "orange","black","blue", "green","red"]
}

image1 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7").select("VV")
image2 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3").select("VV")

image_2020 = ee.ImageCollection('COPERNICUS/S2') \
    .filterBounds(roi) \
    .filterDate('2020-01-01', '2020-12-31') \
    .sort('CLOUD_COVER') \
    .first()

image_2022 = ee.ImageCollection('COPERNICUS/S2') \
    .filterBounds(roi) \
    .filterDate('2022-01-01', '2022-12-31') \
    .sort('CLOUD_COVER') \
    .first()

s2_vis_params = {
    'bands': ['B4', 'B3', 'B2'],  # True color (Red, Green, Blue)
    'min': 0,
    'max': 3000,
    'gamma': 1.4,
}

Map.addLayer(image_2020, s2_vis_params, 'Sentinel-2 Image 2020')
Map.addLayer(image_2022, s2_vis_params, 'Sentinel-2 Image 2022')
Map.addLayer(shapefile, {}, 'Shapefile')
Map.centerObject(shapefile, zoom=10)
Map
