# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# %%
excel_file = pd.ExcelFile("Input/data_performance.xlsx")
sheet_names = excel_file.sheet_names

# %%
rmse_data = []
for sheet in sheet_names:
    df = excel_file.parse(sheet, header=None)
    # Drop columns that are completely NaN/empty first
    df = df.dropna(axis=1, how='all')

    # Check if the sheet has sufficient data
    if df.shape[0] < 2 or df.shape[1] < 4:
        print(f"  [Warning] Skipping '{sheet}': insufficient rows/columns (shape: {df.shape}, need >= 2 rows and >= 4 columns).")
        continue

    # Get dataset name from first row
    data_name = df.iloc[0, 1]
    if pd.isna(data_name) or str(data_name).strip() == "":
        data_name = df.iloc[0, 0]
    if pd.isna(data_name) or str(data_name).strip() == "":
        data_name = sheet
    data_name = str(data_name).strip()

    df_1 = df.iloc[1:, 0:2]
    df_2 = df.iloc[1:, 2:4]

    df_1 = df_1.rename(columns={df_1.columns[0]: 'DateTime', df_1.columns[1]: 'Value'})
    df_2 = df_2.rename(columns={df_2.columns[0]: 'DateTime', df_2.columns[1]: 'Value'})
    
    df_1 = df_1.dropna()
    df_2 = df_2.dropna()
    
    # Convert non-DateTime columns to numeric
    for col in df_1.columns:
        if col != 'DateTime':
            df_1[col] = pd.to_numeric(df_1[col], errors='coerce')
            
    for col in df_2.columns:
        if col != 'DateTime':
            df_2[col] = pd.to_numeric(df_2[col], errors='coerce')

    # Convert DateTime columns to proper datetime types
    df_1['DateTime'] = pd.to_datetime(df_1['DateTime'], errors='coerce', format='mixed')
    df_2['DateTime'] = pd.to_datetime(df_2['DateTime'], errors='coerce', format='mixed')
    df_1 = df_1.dropna(subset=['DateTime'])
    df_2 = df_2.dropna(subset=['DateTime'])

    df_1 = df_1.drop_duplicates(subset=['DateTime'])
    df_2 = df_2.drop_duplicates(subset=['DateTime'])

    # --- Ensure both DataFrames share the same range, start, and interval ---
    # 1. Find overlapping date range
    min_date = max(df_1['DateTime'].min(), df_2['DateTime'].min())
    max_date = min(df_1['DateTime'].max(), df_2['DateTime'].max())

    if pd.isna(min_date) or pd.isna(max_date) or min_date >= max_date:
        print(f"  [Warning] Skipping '{data_name}': no valid overlapping date range.")
        continue

    # 2. Determine common interval (use the finer/smaller interval of the two)
    dt_1 = df_1['DateTime'].sort_values().diff().dropna().median()
    dt_2 = df_2['DateTime'].sort_values().diff().dropna().median()

    # Skip if either interval is NaT (e.g. only one data point)
    if pd.isna(dt_1) and pd.isna(dt_2):
        print(f"  [Warning] Skipping '{data_name}': cannot determine time interval.")
        continue
    elif pd.isna(dt_1):
        common_interval = dt_2
    elif pd.isna(dt_2):
        common_interval = dt_1
    else:
        common_interval = min(dt_1, dt_2)

    # Convert to a seconds-based freq string for pd.date_range
    interval_seconds = int(common_interval.total_seconds())
    if interval_seconds <= 0:
        print(f"  [Warning] Skipping '{data_name}': invalid time interval ({common_interval}).")
        continue

    # 3. Build a single uniform time grid
    common_index = pd.date_range(start=min_date, end=max_date, freq=f'{interval_seconds}s')

    # 4. Clip both DataFrames to the overlapping range first
    df_1 = df_1[(df_1['DateTime'] >= min_date) & (df_1['DateTime'] <= max_date)]
    df_2 = df_2[(df_2['DateTime'] >= min_date) & (df_2['DateTime'] <= max_date)]

    # 5. Set DateTime as index and reindex each df onto the common grid,
    #    then linearly interpolate to fill gaps
    s1 = (df_1.set_index('DateTime')['Value']
              .reindex(df_1.set_index('DateTime').index.union(common_index))
              .interpolate(method='index')
              .reindex(common_index))

    s2 = (df_2.set_index('DateTime')['Value']
              .reindex(df_2.set_index('DateTime').index.union(common_index))
              .interpolate(method='index')
              .reindex(common_index))

    df_merged = pd.DataFrame({'DateTime': common_index, 'Value_1': s1.values, 'Value_2': s2.values})
    df_merged = df_merged.dropna()
    
    # if sheet == 'Sheet14':
    #     print(df_merged)

    observed = df_merged['Value_1']
    predicted = df_merged['Value_2']

    # Calculate R-squared (R²)
    correlation_matrix = np.corrcoef(observed, predicted)
    r_squared = correlation_matrix[0, 1] ** 2

    # Calculate NSE (Nash-Sutcliffe Efficiency)
    nse = 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

    # Calculate PBIAS (Percent Bias)
    pbias = 100 * np.sum(observed - predicted) / np.sum(observed)

    # Calculate RSR (RMSE-observations Standard deviation Ratio)
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    rsr = rmse / np.std(observed)

    rmse_data.append({'Data Name': data_name, 'R²': float(r_squared), 'NSE': float(nse), 'PBIAS': float(pbias), 'RSR': float(rsr)})


    # break

rmse_data = pd.DataFrame(rmse_data)
rmse_data.to_csv('Output/Perf.csv', index=False)
