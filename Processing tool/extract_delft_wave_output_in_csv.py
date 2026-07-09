import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ds = xr.open_dataset("../wavh-wave_20170726.nc")
# print(ds)
# print("Data variables:", list(ds.data_vars))

# Hsig   --- Significant wave height
# RTpeak --- Peak period
# Tm01   --- Mean absolute wave period
# Dir    --- Mean wave direction


variables = ["Hsig", "Dir", "Tm01", "RTpeak"]
station_name = ["Kuakata", "Bhola", "Hatiya", "Sandwip", "Chittagong", "Kutubdia", "Laboni", "Himchori", "Inani"]
station_id = [3, 17, 30, 54, 65, 94, 143, 148, 152]

dict_df = {}
for var in variables:
    data_dict = {"time": pd.to_datetime(ds['time'].values)}
    
    for i, name in zip(station_id, station_name):
        data = ds[var].values[:, i-1]
        data[data == -999] = np.nan
        data_dict[name] = data
    
    df = pd.DataFrame(data_dict)
    dict_df[var] = df


var_labels = {
    "Hsig": "Significant Wave Height (m)",
    "Dir": "Mean Wave Direction (degree)",
    "Tm01": "Mean Absolute Wave Period (s)",
    "RTpeak": "Peak Period (s)"
}

for var in variables:
    plt.figure(figsize=(14, 6))
    
    for i, name in zip(station_id, station_name):
        time_data = pd.to_datetime(dict_df[var]['time'].values)
        var_data = dict_df[var][name].values
        plt.plot(time_data, var_data, label=name, linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(var_labels.get(var, var), fontsize=12)
    plt.title(f'{var_labels.get(var, var)} - All Stations', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=3)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{var}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")
    plt.close()

print("\nAll plots created successfully!")
