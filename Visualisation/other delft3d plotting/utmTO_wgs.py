import pyproj
from pyproj import Transformer

# Define the UTM zone 48N (for South Korea) to WGS84 transformer
transformer = Transformer.from_crs(
    "EPSG:32648",  # UTM Zone 48N
    "EPSG:4326",   # WGS84
    always_xy=True
)

# Read coordinates from wave.loc file
utm_coordinates = []
with open("../wave.loc", 'r') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            easting, northing = map(float, line.split())
            utm_coordinates.append((easting, northing))

# Convert each coordinate pair
print("Converting UTM to WGS84 (Longitude, Latitude):")
print("-" * 50)
for easting, northing in utm_coordinates:
    lon, lat = transformer.transform(easting, northing)
    print(f"UTM: ({easting:.7f}, {northing:.7f}) -> WGS84: ({lon:.7f}°, {lat:.7f}°)")