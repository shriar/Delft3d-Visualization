import webbrowser
import folium
import pyproj
import xarray as xr
import glob
import numpy as np

file_list = glob.glob("data*.nc")
ds = xr.open_dataset(file_list[2])
lat = ds["latitude"].values
lon = ds["longitude"].values

num_lat = len(lat)
num_lon = len(lon)
print(f"num_lon: {num_lon} num_lat: {num_lat}")
coord_array = np.zeros((num_lon, num_lat, 2))

for i in range(num_lon):
    for j in range(num_lat):
        coord_array[i,j,0] = lon[i]
        coord_array[i,j,1] = lat[j]

# coord_array = coord_array[-1, :, :][:, None, :]
print(coord_array.shape)

# convert wgs84 to utm46n
wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 latitude/longitude
utm46N = pyproj.CRS('EPSG:32646')  # UTM zone 46N

transformer = pyproj.Transformer.from_crs(wgs84, utm46N, always_xy=True)

utm_coords = np.zeros_like(coord_array)
for i in range(coord_array.shape[0]):
    for j in range(coord_array.shape[1]):
        utm_coords[i,j,0], utm_coords[i,j,1] = transformer.transform(
            coord_array[i,j,0],  # lon
            coord_array[i,j,1]   # lat
        )

# Fix: Use lat/lon in correct order for initial map location
map_obj = folium.Map(location=[coord_array[0,0,1], coord_array[0,0,0]], zoom_start=8)

for i in range(coord_array.shape[0]):
    for j in range(coord_array.shape[1]):
        # print(f"i: {i} j: {j}")
        folium.Marker(
            [coord_array[i,j,1], coord_array[i,j,0]],  # Folium expects [lat, lon]
            popup=f"UTM46N: ({utm_coords[i,j,0]:.1f}, {utm_coords[i,j,1]:.1f})\n\n lon: {coord_array[i,j,0]:.4f}, lat: {coord_array[i,j,1]:.4f}"
        ).add_to(map_obj)

map_obj.save("map.html")
webbrowser.open("map.html")
