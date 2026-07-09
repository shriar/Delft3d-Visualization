import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

# From era5
# shww = Significant height of wind waves --> m
# pp1d = Peak wave period --> s
# mwp = Mean wave period --> s
# mwd = Mean wave direction  --> degrees
# wdw = Wave spectral directional width --> Radians
# mdww = Mean direction of wind waves --> degrees
# mpww = Mean period of wind waves --> s

# # converted unit to
# # shww --> meter
# # pp1d --> second
# # mwp --> second
# # mwd --> degree
# # wdw --> degree

class FileCreation:
    def __init__(self, start_datetime=None, end_datetime=None, orientation=None):
        self.start_datetime = start_datetime if start_datetime is not None else '2019-01-01T00:00:00.000000000'
        self.end_datetime = end_datetime if end_datetime is not None else '2019-01-16T00:00:00.000000000'
        self.orientation = orientation if orientation is not None else "south"
        self.start_date = self.start_datetime.split("T")[0]

        self.year = self.start_datetime.split("-")[0]

    def clip_data_between_datetime(self, ds):
        mask = (ds["valid_time"] >= np.datetime64(self.start_datetime)) & (ds["valid_time"] <= np.datetime64(self.end_datetime))
        # return ds.sel(valid_time=mask)
        return ds.where(mask, drop=True)

    def create_spectrum_file(self):
        file_path = glob.glob(f"data_stream-wave_{self.year}*.nc")[0]
        ds = xr.open_dataset(file_path)
        clipped_ds = self.clip_data_between_datetime(ds)

        # longitude = clipped_ds["longitude"].values
        # latitude = clipped_ds["latitude"].values
        # print(clipped_ds["shww"].values.shape)
        # print(f"latitude: {latitude.shape}  longitude: {longitude.shape}")
        if self.orientation == "south":
            shww = clipped_ds["shww"].values[:, -1, :]
            shww = np.nanmean(shww, axis=-1, keepdims=False) # meter

            mpww = clipped_ds["pp1d"].values[:, -1, :]
            mpww = np.nanmean(mpww, axis=-1, keepdims=False) # second

            mdww = clipped_ds["mwd"].values[:, -1, :]
            mdww = 90 - np.nanmean(mdww, axis=-1, keepdims=False) # degree

            wdw = clipped_ds["wdw"].values[:, -1, :]
            wdw = np.degrees(np.nanmean(wdw, axis=-1, keepdims=False)) # convert to degrees
        elif self.orientation == "north":
            shww = clipped_ds["shww"].values[:, 0, :]
            shww = np.nanmean(shww, axis=-1, keepdims=False) # meter

            mpww = clipped_ds["pp1d"].values[:, 0, :]
            mpww = np.nanmean(mpww, axis=-1, keepdims=False) # second

            mdww = clipped_ds["mwd"].values[:, 0, :]
            mdww = 90 - np.nanmean(mdww, axis=-1, keepdims=False) # degree

            wdw = clipped_ds["wdw"].values[:, 0, :]
            wdw = np.degrees(np.nanmean(wdw, axis=-1, keepdims=False)) # convert to degrees
        elif self.orientation == "east":
            shww = clipped_ds["shww"].values[:, :, -1]
            shww = np.nanmean(shww, axis=-1, keepdims=False) # meter

            mpww = clipped_ds["pp1d"].values[:, :, -1]
            mpww = np.nanmean(mpww, axis=-1, keepdims=False) # second

            mdww = clipped_ds["mwd"].values[:, :, -1]
            mdww = 90 - np.nanmean(mdww, axis=-1, keepdims=False) # degree

            wdw = clipped_ds["wdw"].values[:, :, -1]
            wdw = np.degrees(np.nanmean(wdw, axis=-1, keepdims=False)) # convert to degrees
        elif self.orientation == "west":
            shww = clipped_ds["shww"].values[:, :, 0]
            shww = np.nanmean(shww, axis=-1, keepdims=False) # meter

            mpww = clipped_ds["pp1d"].values[:, :, 0]
            mpww = np.nanmean(mpww, axis=-1, keepdims=False) # second

            mdww = clipped_ds["mwd"].values[:, :, 0]
            mdww = 90 - np.nanmean(mdww, axis=-1, keepdims=False) # degree

            wdw = clipped_ds["wdw"].values[:, :, 0]
            wdw = np.degrees(np.nanmean(wdw, axis=-1, keepdims=False)) # convert to degrees

        # print(shww.shape)
        # print(mpww.shape)
        # print(mdww.shape)
        # print(wdw.shape)

        # save data to text file
        df = pd.DataFrame(clipped_ds["valid_time"].dt.strftime("%Y%m%d.%H%M").values, columns=["valid_time"])
        df["shww"] = shww
        df["mpww"] = mpww
        df["mdww"] = mdww
        df["wdw"] = wdw

        file_name = f"wave_{self.start_date}_{self.orientation}.bnd"
        df.to_csv(f'../{file_name}', header=False, index=False, sep=' ')

        with open(f'../{file_name}', 'r') as file:
            data = file.readlines()
        data.insert(0, 'TPAR\n')

        with open(f'../{file_name}', 'w') as file:
            file.writelines(data)

        print(f"file {file_name} created")

    def plot_wave_parameters(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Wave Parameters Over Time', fontsize=16)

        # Convert time string back to datetime for plotting
        time = pd.to_datetime(df['valid_time'], format='%Y%m%d.%H%M')

        # Plot each parameter
        axs[0,0].plot(time, df['shww'])
        axs[0,0].set_title('Significant Wave Height')
        axs[0,0].set_ylabel('Height (m)')
        axs[0,0].tick_params(axis='x', rotation=45)

        axs[0,1].plot(time, df['mpww']) 
        axs[0,1].set_title('Mean Wave Period')
        axs[0,1].set_ylabel('Period (s)')
        axs[0,1].tick_params(axis='x', rotation=45)

        axs[1,0].plot(time, df['mdww'])
        axs[1,0].set_title('Mean Wave Direction') 
        axs[1,0].set_ylabel('Direction (degrees)')
        axs[1,0].tick_params(axis='x', rotation=45)

        axs[1,1].plot(time, df['wdw'])
        axs[1,1].set_title('Wave Directional Width')
        axs[1,1].set_ylabel('Width (degrees)') 
        axs[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def create_wind_time_varying_file(self):
        file_path = glob.glob(f"data_stream-oper_{self.year}*.nc")[0]
        ds = xr.open_dataset(file_path)
        clipped_ds = self.clip_data_between_datetime(ds)

        reference_time = pd.to_datetime(self.start_datetime)
        # reference_time = pd.to_datetime('20190101.0000', format='%Y%m%d.%H%M')
        valid_time = pd.to_datetime(clipped_ds["valid_time"].values)
        time_diff = (valid_time - reference_time).total_seconds() / 60
        u10 = clipped_ds["u10"].values.mean(axis=-1).mean(axis=-1)
        v10 = clipped_ds["v10"].values.mean(axis=-1).mean(axis=-1)
        uv = np.sqrt(u10**2 + v10**2)
        direction = np.degrees(np.arctan2(u10, v10))

        df = pd.DataFrame(time_diff, columns=["time_diff"])
        df["uv"] = uv
        df["direction"] = direction

        file_name = f"Wind_{self.start_date}.wnd"
        df.to_csv(f'../{file_name}', header=False, index=False, sep=' ')
        print(f"file {file_name} created")

    def create_wind_time_space_varying_file(self):
        ds = xr.open_dataset(file_list[0])
        clipped_ds = self.clip_data_between_datetime(ds)

        # reference_time = pd.to_datetime('20190101.0000', format='%Y%m%d.%H%M')
        valid_time = pd.to_datetime(clipped_ds["valid_time"].values)
        time_diff = (valid_time - valid_time[0]).total_seconds() / 60
        u10 = clipped_ds["u10"].values
        v10 = clipped_ds["v10"].values
        print(f"u10.shape: {u10.shape} - v10.shape: {v10.shape}")
        from scipy.interpolate import RegularGridInterpolator

        # Define the original grid
        original_x = np.arange(u10.shape[1])
        original_y = np.arange(u10.shape[2])
        original_t = np.arange(u10.shape[0])

        # Define the new grid
        new_x = np.linspace(0, u10.shape[1] - 1, 474)
        new_y = np.linspace(0, v10.shape[2] - 1, 401)
        new_t = original_t  # time dimension remains the same

        # Create interpolators for u10 and v10
        interpolator_u10 = RegularGridInterpolator((original_t, original_x, original_y), u10)
        interpolator_v10 = RegularGridInterpolator((original_t, original_x, original_y), v10)

        # Create a meshgrid for the new grid
        new_grid = np.meshgrid(new_t, new_x, new_y, indexing='ij')
        new_points = np.array([new_grid[0].flatten(), new_grid[1].flatten(), new_grid[2].flatten()]).T

        # Interpolate the data
        u10_interpolated = interpolator_u10(new_points).reshape((73, 401, 474))
        v10_interpolated = interpolator_v10(new_points).reshape((73, 401, 474))

        print(f"u10_interpolated.shape: {u10_interpolated.shape} - v10_interpolated.shape: {v10_interpolated.shape}")
        with open('../wind_boundary_1.wnd', 'w') as f:
            f.write("FileVersion = 1.03\n")
            f.write("FileType = meteo_on_computational_grid\n")
            f.write("n_quantity = 3\n")
            f.write("quantity1 = x_wind\n")
            f.write("quantity2 = y_wind\n")
            f.write("quantity3 = air_pressure\n")
            f.write("unit1 = m s-1\n")
            f.write("unit2 = m s-1\n")
            f.write("unit3 = Pa\n")
            # f.write("grid_unit = m\n")
            # f.write("x_llcorner = 0.0\n")
            # f.write("y_llcorner = 0.0\n")
            
            for i in tqdm(range(u10_interpolated.shape[0])):
                Time = time_diff[i]
                u10 = u10_interpolated[i, :, :]
                v10 = v10_interpolated[i, :, :]
                # f.write(f"Time = {Time} minutes since {valid_time[i].strftime('%Y-%m-%d %H:%M:%S')} +00:00 # Time definition\n")
                # f.write(" ".join(f"{val:.3f}" for val in u10) + " # Wind component west to east\n")
                # f.write(" ".join(f"{val:.3f}" for val in v10) + " # Wind component south to north\n")
                # f.write(" ".join(f"{val:.3f}" for val in np.zeros_like(u10)) + " # Atmospheric pressure\n")
                f.write(f"Time = {Time} minutes since {valid_time[i].strftime('%Y-%m-%d %H:%M:%S')} +00:00\n")
                np.savetxt(f, u10, fmt='%.3f')
                f.write("\n")
                np.savetxt(f, v10, fmt='%.3f')
                f.write("\n")
                np.savetxt(f, np.zeros_like(u10), fmt='%.3f')

        print(f"file created")

list_startDateTime = ['2018-01-01T00:00:00.000000000', '2018-07-01T00:00:00.000000000', '2019-01-01T00:00:00.000000000', '2019-07-01T00:00:00.000000000']
list_endDateTime =   ['2018-01-16T00:00:00.000000000', '2018-07-16T00:00:00.000000000', '2019-01-16T00:00:00.000000000', '2019-07-16T00:00:00.000000000']
# list_endDateTime = [(datetime.fromisoformat(start) + timedelta(days=15)).isoformat() for start in list_startDateTime]
orientation = 'south'

for start_datetime, end_datetime in zip(list_startDateTime, list_endDateTime):
    # print(f"start_datetime: {start_datetime} - end_datetime: {end_datetime}")
    file_creation = FileCreation(start_datetime, end_datetime, orientation)

    file_creation.create_spectrum_file()
    file_creation.create_wind_time_varying_file()

# file_creation = FileCreation()

# file_creation.create_spectrum_file()
# file_creation.create_wind_time_varying_file()
