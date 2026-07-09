import xarray as xr
import glob
import pandas as pd
from pyproj import Transformer
from geopy.distance import distance
import numpy as np

def write_header(output_file, grid_unit="degree", x_llcenter=None, y_llcenter=None, dx=0.25, dy=0.25, quantity1=None, unit1=None, n_cols=None, n_rows=None):
    header = f"""FileVersion      = 1.03
filetype         =  meteo_on_equidistant_grid
NODATA_value     = -32767
n_cols           = {n_cols}
n_rows           = {n_rows}
grid_unit        = {grid_unit}
x_llcenter       = {x_llcenter}
y_llcenter       = {y_llcenter}
dx               = {dx}
dy               = {dy}
n_quantity       = 1
quantity1        = {quantity1}
unit1            = {unit1}
"""

    output_file.write(header)

def create_space_varying_data(var_name, coordinate_system="wgs", file_path=None, output_folder=None):
    file_list = glob.glob(file_path)
    target_file = file_list[0]
    
    if target_file.lower().endswith(('.grib', '.grb', '.grib2', '.grb2')):
        ds = xr.open_dataset(target_file, engine='cfgrib')
    else:
        ds = xr.open_dataset(target_file)
    
    start_time = pd.Timestamp("2025-01-01")
    end_time = pd.Timestamp("2025-01-31")

    # start_time = pd.Timestamp(ds.valid_time.min().values)
    # end_time = pd.Timestamp(ds.valid_time.max().values)

    mask = (ds.valid_time >= start_time) & (ds.valid_time < end_time)
    ds = ds.sel(valid_time=mask)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32646", always_xy=True)
    lon, lat = ds.longitude[0].values, ds.latitude[-1].values
    x, y = transformer.transform(lon, lat) if coordinate_system == "utm" else (lon, lat)

    dx_dy = 0.25
    if coordinate_system == "utm":
        point1 = (ds.latitude[-1].values, ds.longitude[0].values)
        point2 = (ds.latitude[-1].values, ds.longitude[1].values)
        dx_dy = distance(point1, point2).meters

    var_data = ds[var_name].values
    var_data[np.isnan(var_data)] = -999.0

    num_times = len(ds.valid_time)
    num_lat = len(ds.latitude)
    num_lon = len(ds.longitude)
    
    filename = f"{output_folder}/{var_name}_{coordinate_system}_{start_time.year}{start_time.month:02d}{start_time.day:02d}"
    filename += ".amu" if var_name == "u10" else ".amv" if var_name == "v10" else ".amp" if var_name == "msl" else ".unknown"

    with open(filename, 'w') as outfile:
        grid_unit = "m" if coordinate_system == "utm" else "degree" if coordinate_system == "wgs" else "unknown"
        header_params = {
            'grid_unit': grid_unit,
            'x_llcenter': x,
            'y_llcenter': y,
            'dx': dx_dy,
            'dy': dx_dy,
            'quantity1': "x_wind" if var_name == "u10" else "y_wind" if var_name == "v10" else "air_pressure" if var_name == "msl" else None,
            'unit1': "m s-1" if var_name in ["u10", "v10"] else "Pa" if var_name == "msl" else None,
            'n_cols': num_lon,
            'n_rows': num_lat
        }
        write_header(outfile, **header_params)

        for i in range(num_times):
            target_time = pd.Timestamp(ds.valid_time[i].values)
            hour = (target_time - start_time).total_seconds() / 3600
            outfile.write(f"TIME = {hour} hours since {start_time.strftime('%Y-%m-%d %H:%M:%S')} +00:00\n")
            
            np.savetxt(outfile, var_data[i][::-1], fmt="%.2f", delimiter=" ", newline=" \n")
        
        print(f"Created file: {filename}")


def main():
    create_space_varying_data("u10", file_path="Input/wind_pressure_2025.grib", output_folder='Output', coordinate_system="wgs")
    create_space_varying_data("v10", file_path="Input/wind_pressure_2025.grib", output_folder='Output', coordinate_system="wgs")
    create_space_varying_data("msl", file_path="Input/wind_pressure_2025.grib", output_folder='Output', coordinate_system="wgs")


if __name__ == "__main__":
    main()
