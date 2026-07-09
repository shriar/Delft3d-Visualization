import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
# from sklearn.metrics import r2_score
# import json
# import os
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# From era5
# shww = Significant height of wind waves --> m
# pp1d = Peak wave period --> s
# mwp = Mean wave period --> s
# mwd = Mean wave direction  --> degrees
# wdw = Wave spectral directional width --> Radians

# model
# RTpeak = peak wave period [s]
# Dspr = directional spreading of the waves [◦]
# Tm01 = mean wave period (Tm01) [s]
# Hsig = significant wave height [m]

class WaveAnalysis:
    def __init__(self):
        self.wave_file_paths = glob.glob("../wavh-wave_*.nc")
        # print(self.wave_file_paths)
        self.wave_filename = self.wave_file_paths[2]
        self.year = self.wave_filename.split("_")[-2][:4]
        self.fig_name = self.wave_filename.split("_")[-1].split(".")[0]
        # print(self.wave_filename)
        self.file_path = glob.glob(f"data_stream-wave_{self.year}*.nc")[0]
        
        self.model_ds = xr.open_dataset(self.wave_filename)
        self.obs_ds = xr.open_dataset(self.file_path)

        self.setup_dataframes()

    def setup_dataframes(self):
        # Set up model dataframe
        self.model_df = pd.DataFrame(self.model_ds["time"].values, columns=["Time"])
        self.model_df["Hsig_1"] = self.model_ds["Hsig"].values[:, 0]
        self.model_df["Hsig_2"] = self.model_ds["Hsig"].values[:, 1]
        self.model_df["Hsig_3"] = self.model_ds["Hsig"].values[:, 2]
        
        print(len(self.model_df))

        # Set up observation dataframe
        lat_index_1 = np.where(self.obs_ds["latitude"].values == 22.23)[0][0]
        lon_index_1 = np.where(self.obs_ds["longitude"].values == 91.57)[0][0]

        lat_index_2 = np.where(self.obs_ds["latitude"].values == 21.73)[0][0] 
        lon_index_2 = np.where(self.obs_ds["longitude"].values == 90.57)[0][0]

        lat_index_3 = np.where(self.obs_ds["latitude"].values == 21.73)[0][0]
        lon_index_3 = np.where(self.obs_ds["longitude"].values == 91.07)[0][0]

        self.obs_df = pd.DataFrame(self.obs_ds["valid_time"].values, columns=["Time"])
        self.obs_df["shww_1"] = self.obs_ds["shww"].values[:, lat_index_1, lon_index_1]
        self.obs_df["shww_2"] = self.obs_ds["shww"].values[:, lat_index_2, lon_index_2]
        self.obs_df["shww_3"] = self.obs_ds["shww"].values[:, lat_index_3, lon_index_3]

        # Process dataframes
        self.model_df, self.obs_df = self.clip_df_common_time(self.model_df, self.obs_df)
        self.model_df, self.obs_df = self.interpolate(self.obs_df, self.model_df)

    def interpolate(self, obs_df, model_df):
        min_interval = min(obs_df['Time'].diff().min(), model_df['Time'].diff().min())

        new_time_index = pd.date_range(start=min(obs_df['Time'].min(), model_df['Time'].min()),
                                    end=max(obs_df['Time'].max(), model_df['Time'].max()),
                                    freq=min_interval)

        model_interpolated = model_df.set_index('Time').reindex(new_time_index).interpolate()
        obs_interpolated = obs_df.set_index('Time').reindex(new_time_index).interpolate()
        
        model_interpolated.index.name = 'Time'
        obs_interpolated.index.name = 'Time'
        
        model_interpolated = model_interpolated.reset_index()
        obs_interpolated = obs_interpolated.reset_index()
        return model_interpolated, obs_interpolated

    def clip_df_common_time(self, model_df, obs_df):
        start_time = max(model_df['Time'][0], obs_df['Time'][0])
        end_date = min(model_df['Time'].iloc[-1], obs_df['Time'].iloc[-1])

        model_df = model_df[(model_df['Time'] >= start_time) & (model_df['Time'] <= end_date)]
        obs_df = obs_df[(obs_df['Time'] >= start_time) & (obs_df['Time'] <= end_date)]

        return model_df, obs_df

    def create_plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        num_plots = 3
        
        for plot_idx in range(num_plots):
            row = plot_idx // 2
            col = plot_idx % 2
            axs[row, col].plot(self.model_df["Time"], self.model_df[f"Hsig_{plot_idx+1}"], label=f"Model")
            axs[row, col].plot(self.obs_df["Time"], self.obs_df[f"shww_{plot_idx+1}"], label=f"Observed")
            axs[row, col].legend()
            
            axs[row, col].tick_params(axis='x', rotation=45, labelsize=8)
            
            # Custom date formatter that only shows year on first tick
            def date_format_func(x, pos=None):
                date = mdates.num2date(x)
                if pos == 0:  # First tick
                    return date.strftime('%Y-%m-%d %H:%M')
                return date.strftime('%m-%d %H:%M')
                
            axs[row, col].xaxis.set_major_formatter(plt.FuncFormatter(date_format_func))

        plt.tight_layout()
        return fig
    
    def show_plot(self):
        fig = self.create_plot()
        plt.show()

    def save_fig(self, filename):
        fig = self.create_plot()
        fig.savefig(f'{filename}.png')
        plt.close(fig)

    def process_all_files(self):
        for wave_file in self.wave_file_paths:
            self.wave_filename = wave_file
            self.year = wave_file.split("_")[-2][:4]
            self.fig_name = wave_file[8:-3]
            self.obs_file_path = glob.glob(f"data_stream-wave_{self.year}*.nc")[0]
            
            self.model_ds = xr.open_dataset(wave_file)
            self.obs_ds = xr.open_dataset(self.obs_file_path)
            
            self.setup_dataframes()
            self.save_fig(self.fig_name)

# Create instance and process all files
wave_analysis = WaveAnalysis()
wave_analysis.process_all_files()
# wave_analysis.show_plot()

def plot_tidal_water_level():
    def read_nc(path):
        data_coarser = xr.open_dataset(path)
        df_coarser = pd.DataFrame({'time': pd.to_datetime(data_coarser['time'].values)})
        df_coarser["ZWL"] = data_coarser["ZWL"].values
        return df_coarser

    def read_csv(path):
        data_finer = pd.read_csv(path)
        df_finer = pd.DataFrame({"time": pd.to_datetime(data_finer["date and time"])})
        df_finer["ZWL"] = data_finer["water level (m)"]
        return df_finer

    def interpolate(small_df, large_df):
        # min_interval = '100T'
        min_interval = min(small_df['time'].diff().min(), large_df['time'].diff().min())
        # print(min_interval)
        new_time_index_small = pd.date_range(start=small_df['time'].min(), end=small_df['time'].max(), freq=min_interval)
        new_time_index_large = pd.date_range(start=large_df['time'].min(), end=large_df['time'].max(), freq=min_interval)

        small_df_interpolated = small_df.set_index('time').reindex(new_time_index_small).interpolate()
        small_df_interpolated.index.name = 'time'
        small_df_interpolated.reset_index(inplace=True)

        large_df_interpolated = large_df.set_index('time').reindex(new_time_index_large).interpolate()
        large_df_interpolated.index.name = 'time'
        large_df_interpolated.reset_index(inplace=True)

        return small_df_interpolated, large_df_interpolated

    coarser_df = read_nc("../nc coarser his for point/trih-coarser20190101.nc")
    finer_df = read_csv("../water level.csv")

    finer_df, coarser_df = interpolate(finer_df, coarser_df)

    plt.figure(figsize=(10, 6))
    plt.plot(coarser_df["time"], coarser_df["ZWL"], label="Coarser", color='blue')
    plt.scatter(finer_df["time"], finer_df["ZWL"], label="Finer", s=1, color='red')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    plt.show()

# plot wave variable
def plot_wave_variables():
    pass
    # def clip_data_between_datetime(ds, start_datetime, end_datetime):
    #     mask = (ds["valid_time"] >= np.datetime64(start_datetime)) & (ds["valid_time"] <= np.datetime64(end_datetime))
    #     return ds.where(mask, drop=True)

    # clip_ds = clip_data_between_datetime(combined_ds, '2019-01-01T00:00:00.000000000', '2019-05-01T00:00:00.000000000')
    # clip_time = clip_ds["valid_time"].values
    # mwd_2 = clip_ds["mwd"][:, 0, 0].values
    # pp1d_2 = clip_ds["pp1d"][:, 0, 0].values
    # shww_2 = clip_ds["shww"][:, 0, 0].values
    # wdw_2 = np.rad2deg(clip_ds["wdw"][:, 0, 0].values)
    # mwp_2 = clip_ds["mwp"][:, 4, 10].values
    # # shww = Significant height of wind waves --> m
    # # pp1d = Peak wave period --> s
    # # mwd = Mean wave direction  --> degrees
    # # wdw = Wave spectral directional width --> Radians
    # # mwp = Mean wave period --> s
    # def plot_all_vars(time, vars_dict):
    #     fig = make_subplots(rows=3, cols=2, 
    #                         subplot_titles=list(vars_dict.keys()),
    #                         vertical_spacing=0.1, horizontal_spacing=0.09,
    #                         )
        
    #     for i, (var_name, var_data) in enumerate(vars_dict.items()):
    #         row = i // 2 + 1
    #         col = i % 2 + 1
            
    #         fig.add_trace(
    #             go.Scatter(x=time, y=var_data, mode='lines', name=var_name, showlegend=False),
    #             row=row, col=col
    #         )
            
    #         fig.update_xaxes(title_text="Time", row=row, col=col, gridcolor='white')
    #         fig.update_yaxes(title_text=var_name, row=row, col=col, gridcolor='white')

    #     fig.update_layout(
    #         height=1200,
    #         width=1300,
    #         plot_bgcolor='lightblue',
    #         paper_bgcolor='lightyellow',
    #         title_text="Ocean Wave Variables Over Time"
    #     )

    #     fig.show()

    # vars_dict = {
    #     "Mean Wave Direction (degrees)": mwd_2,
    #     "Peak wave period (s)": pp1d_2,
    #     "Significant Height of Wind Waves (m)": shww_2,
    #     "Wave spectral directional width (degrees)": wdw_2,
    #     "Mean Wave Period (s)": mwp_2
    # }

    # plot_all_vars(clip_time, vars_dict)