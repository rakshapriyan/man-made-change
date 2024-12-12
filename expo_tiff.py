import ee
import geemap.core as geemap

ee.Authenticate()
ee.Initialize(project='loyal-curve-440318-m8')


def export_tiff_file(changeW, file_name):  
  # Convert FeatureCollection to a raster image (use 'area' property for the pixel value, or change to your relevant property)
  rasterized_image = changeW.reduceToImage(
      properties=['area'],  # Property to assign to pixel values
      reducer=ee.Reducer.first()
  )

  # Set the projection and resolution for the export
  rasterized_image = rasterized_image.reproject(
      crs='EPSG:4326',  # Choose your desired CRS (coordinate reference system)
      scale=30  # Set the desired resolution (in meters)
  )

  # Export as GeoTIFF to Google Drive
  export_task = ee.batch.Export.image.toDrive(
      image=rasterized_image,
      description=file_name,
      fileFormat='GeoTIFF',
      region=changeW.geometry(),  # Define the region for export (using the feature collection geometry)
      scale=30,  # Resolution in meters
      crs='EPSG:4326',  # CRS
      folder='EarthEngineExports'  # Optional: Specify a folder in Google Drive
  )

  # Start the export task
  export_task.start()

print("Earth Engine Initialized successfully.")

# SHP FILE things
sar_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
    .filter(ee.Filter.eq("instrumentMode", "IW")) \
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")
  )
    
lt = 28.3604, 76.6773
rb = 28.1123, 76.9972

# lt = [28.1889374044219, 76.85657057751139]
# rb = [28.176643457015388, 76.87137449764242]
roi = ee.Geometry.Rectangle([*lt[::-1], *rb[::-1]])

changeW = ee.FeatureCollection('users/npc203/BWD20_ChangesW')
changes = ee.FeatureCollection('users/npc203/BWD20_Changes')


image1 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200516T005224_20200516T005249_032583_03C61F_9EE7").select("VV").clip(roi)
image2 = ee.Image("COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240519T005234_20240519T005259_053933_068E47_9BD3").select("VV").clip(roi)


vis_params = {
    "min": -20,
    "max": 0,
    "palette": ["yellow", "orange","black","blue", "green","red"]
}

difference = image2.subtract(image1)
pd = difference.convolve(ee.Kernel.gaussian(radius=3, sigma=3, units='pixels')).gt(7)
borders = pd.selfMask() \
    .reduceToVectors(
        geometryType='polygon',
        reducer=ee.Reducer.countEvery(),
        scale=10,
        maxPixels=1e8
    )

Map = geemap.Map()
Map.centerObject(roi, zoom=15)

Map.addLayer(pd.updateMask(pd.gt(0)), {'min': 0, 'max': 1, 'palette': ['white', 'blue']}, "Difference")
Map.addLayer(changeW, {}, 'changeW')
Map.addLayer(borders, {'color':'red'}, "Difference2")

# Map.addLayer(image1, vis_params, '2020')
# Map.addLayer(image2, vis_params, '2024')



#export_tiff_file(borders, "curr_file")
Map