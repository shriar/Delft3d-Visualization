import pandas as pd
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import re
import io
from typing import List, Union, Dict, Any, Optional

def read_nc(path: str) -> pd.DataFrame:
    """Read netCDF file and convert to DataFrame with time and water level data."""
    ds = xr.open_dataset(path)
    df = pd.DataFrame({'Time': pd.to_datetime(ds['time'].values)})
    df["ZWL"] = ds["ZWL"].values[:, 0]
    return df

def read_csv(path: str) -> pd.DataFrame:
    """Read CSV file and convert to DataFrame with time and water level data."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    df['Time'] = pd.to_datetime(df['date and time'])
    df['ZWL'] = df['water level (m)'].astype(float)
    return df[['Time', 'ZWL']]


def Read_tek_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content_without_header = '\n'.join(content.split('\n')[5:])

    df = pd.read_csv(io.StringIO(content_without_header),
                    delim_whitespace=True,
                    names=['Date', 'Time', 'WL'])

    def pad_time(time_str):
        return time_str.zfill(6)

    df['Time'] = df['Time'].astype(str).apply(pad_time)
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='%Y%m%d %H%M%S')
    df = df.drop(columns=['Date', 'Time'])
    df = df[['DateTime', 'WL']]

    return df

def clip_dataframe_by_time(df: pd.DataFrame,
                          start_datetime: Union[str, pd.Timestamp],
                          end_datetime: Union[str, pd.Timestamp]) -> pd.DataFrame:
    """Filter DataFrame to include only rows within the specified time range."""
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)

    return df[(df['Time'] >= start_datetime) & (df['Time'] <= end_datetime)]

def interpolate_dataframe(df: pd.DataFrame, time_interval: str) -> pd.DataFrame:
    """Interpolate DataFrame to have regular time intervals."""
    new_time_index = pd.date_range(
        start=df['Time'].min(),
        end=df['Time'].max(),
        freq=time_interval
    )

    df_interpolated = df.set_index('Time').reindex(new_time_index).interpolate()
    df_interpolated.index.name = 'Time'

    return df_interpolated.reset_index()

def plot_tidal_water_level(df: pd.DataFrame, fig_path: Optional[str] = None,
                          specified_idx: Optional[np.ndarray] = None, title: Optional[str] = None) -> None:
    """Plot tidal water level data with annotations for specified indices."""
    plt.figure(figsize=(12, 8))
    plt.plot(df['Time'], df['ZWL'], 'b-', linewidth=2.5, label='Water Level')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))

    for i, row in df.iterrows():
        if specified_idx is not None and i in specified_idx:
            print(i)
            plt.plot(row['Time'], row['ZWL'], 'ro', markersize=12)
            plt.annotate(f"Time: {row['Time'].strftime('%Y-%m-%d %H:%M')}\nIndex: {i + 1}",
                        xy=(row['Time'], row['ZWL']),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.9)
        else:
            continue
            plt.plot(row['Time'], row['ZWL'], 'ko', markersize=4)
            plt.annotate(f"Time: {row['Time'].strftime('%Y-%m-%d %H:%M')}\nIndex: {i + 1}",
                        xy=(row['Time'], row['ZWL']),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.4)

    plt.xlabel('Time', fontsize=12, fontweight='bold')
    plt.ylabel('Water Level (m)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks([])
    # plt.xticks(rotation=30, fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
    plt.close()

def process_tidal_data(file_path: str, start_time: Optional[str] = None,
                      end_time: Optional[str] = None, interval: str = "60min") -> pd.DataFrame:
    """Process tidal data by reading, interpolating and optionally clipping by time."""
    if file_path.endswith('.csv'):
        full_df = read_csv(file_path)
    else:
        full_df = read_nc(file_path)
    full_df = interpolate_dataframe(full_df, interval)

    if start_time and end_time:
        return clip_dataframe_by_time(full_df, start_time, end_time)
    return full_df

def main() -> None:
    scenarios = {
        "wet": {
            "trih_path": "Input/20260726_WL_Sandwip.csv",
            "spring": {
                "start_time": "2025-07-26 00:00:00",
                "end_time": "2025-07-28 00:00:00",
                "flood": {
                    "middle": np.array([18]),   # Index 17 (0-based) -> 2025-07-26 17:00:00
                    "top": np.array([21]),      # Index 20 (0-based) -> 2025-07-26 20:00:00
                },
                "ebb": {
                    "middle": np.array([13]),   # Index 12 (0-based) -> 2025-07-26 12:00:00
                    "down": np.array([16]),     # Index 15 (0-based) -> 2025-07-26 15:00:00
                }
            },
            "neap": {
                "start_time": "2025-08-03 00:00:00",
                "end_time": "2025-08-05 00:00:00",
                "flood": {
                    "middle": np.array([216]),  # Index 215 (0-based) -> 2025-08-03 23:00:00
                    "top": np.array([220])      # Index 219 (0-based) -> 2025-08-04 03:00:00
                },
                "ebb": {
                    "middle": np.array([209]),  # Index 208 (0-based) -> 2025-08-03 16:00:00
                    "down": np.array([213])     # Index 212 (0-based) -> 2025-08-03 20:00:00
                }
            },
        },
        "dry": {
            "trih_path": "Input/20260726_WL_Sandwip.csv",
            "spring": {
                "start_time": "2025-07-26 00:00:00",
                "end_time": "2025-07-28 00:00:00",
                "flood": {
                    "middle": np.array([18]),
                    "top": np.array([21])
                },
                "ebb": {
                    "middle": np.array([13]),
                    "down": np.array([16])
                }
            },
            "neap": {
                "start_time": "2025-08-03 00:00:00",
                "end_time": "2025-08-05 00:00:00",
                "flood": {
                    "middle": np.array([216]),
                    "top": np.array([220])
                },
                "ebb": {
                    "middle": np.array([209]),
                    "down": np.array([213])
                }
            },
        },
    }

    for session_name, session_config in scenarios.items():
        for tide_type, tide_config in session_config.items():
            if tide_type in ["spring", "neap"]:
                for phase, phase_config in tide_config.items():
                    if phase in ["flood", "ebb"]:
                        for level_type, index in phase_config.items():
                            date_match = re.search(r'(\d{8})', session_config["trih_path"])
                            fig_name = date_match.group(1) if date_match else "output"

                            processed_df = process_tidal_data(
                                session_config["trih_path"],
                                start_time=tide_config["start_time"],
                                end_time=tide_config["end_time"],
                                interval="60min"
                            )

                            output_path = f"Output/figure/scenario_4/tidalWL_{session_name}_{tide_type}_{phase}_{level_type}.png"
                            title = f"{fig_name}_{session_name}_{tide_type}_{phase}_{level_type}"
                            plot_tidal_water_level(processed_df, output_path, specified_idx=index - 1, title=title)

if __name__ == "__main__":
    main()
