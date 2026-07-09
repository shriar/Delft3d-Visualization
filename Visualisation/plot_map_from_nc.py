import webbrowser
import folium
import xarray as xr
import glob
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

folder_path = "Input"
file_list = glob.glob(f"{folder_path}\\wave_2025.nc")
ds = xr.open_dataset(f"{folder_path}/wave_2025.nc")
# print(ds["shww"].values.shape)
lat = ds["latitude"].values
lon = ds["longitude"].values

# Create a 3D matrix: (num_lon, num_lat, 2)
num_lat = len(lat)
num_lon = len(lon)
coord_array = np.zeros((num_lon, num_lat, 2))
for i in range(num_lon):
    for j in range(num_lat):
        coord_array[i, j, 0] = lon[i]
        coord_array[i, j, 1] = lat[j]
        print(f"{lon[i]:.4f} {lat[j]:.4f}")

# Flatten the 3D matrix for plotting
coords = coord_array.reshape(-1, 2)

# Create GeoDataFrame for shapefile
gdf = gpd.GeoDataFrame(
    {'latitude': coords[:,1], 'longitude': coords[:,0]},
    geometry=[Point(xy) for xy in coords],
    crs="EPSG:4326"
)
# gdf.to_file(f"{folder_path}/points_nc.gpkg", driver='GPKG')

# Set up the map at the mean location
map_obj = folium.Map(location=[coords[:,1].mean(), coords[:,0].mean()], zoom_start=8)

for lon_val, lat_val in coords:
    folium.CircleMarker(
        location=[lat_val, lon_val],
        radius=1,
        popup=f"(lat, lon): ({lon_val:.4f}, {lat_val:.4f})",
        color='red',
        fill=True
    ).add_to(map_obj)

map_obj.save(f"{folder_path}/map_ECMWF_dry.html")
webbrowser.open(f"{folder_path}/map_ECMWF_dry.html")
