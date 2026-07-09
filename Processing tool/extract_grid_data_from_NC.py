"""Extract time series data for a specific grid point from a NetCDF file and export to CSV."""

from pathlib import Path
import xarray as xr
import pandas as pd


def extract_grid_data(
    nc_path: str,
    target_lat: float,
    target_lon: float,
    output_csv: str = None,
    method: str = 'nearest'
) -> pd.DataFrame:
    """Extract time series data from NetCDF at a specific lat/lon.

    Args:
        nc_path: Path to the NetCDF file.
        target_lat: Latitude of interest.
        target_lon: Longitude of interest.
        output_csv: Path to save the output CSV. If None, won't save.
        method: Selection method ('nearest' or None for exact match).

    Returns:
        Pandas DataFrame containing the extracted time series.
    """
    print(f"Opening NetCDF file: '{nc_path}'...")
    with xr.open_dataset(nc_path) as ds:
        # Check coordinates in dataset
        available_lats = ds.latitude.values
        available_lons = ds.longitude.values
        print(f"Available latitudes in NetCDF: {available_lats}")
        print(f"Available longitudes in NetCDF: {available_lons}")

        # Select data at the target latitude and longitude
        print(f"Extracting grid data for Target Lat: {target_lat}, Target Lon: {target_lon} (method: {method})...")
        if method == 'nearest':
            # Select nearest point
            point_ds = ds.sel(latitude=target_lat, longitude=target_lon, method='nearest')
            actual_lat = float(point_ds.latitude.values)
            actual_lon = float(point_ds.longitude.values)
            print(f"Nearest grid point found: Lat {actual_lat:.4f}, Lon {actual_lon:.4f}")
        else:
            # Exact selection
            point_ds = ds.sel(latitude=target_lat, longitude=target_lon)
            actual_lat = target_lat
            actual_lon = target_lon

        # Drop non-coordinate single dimensions/variables that might clutter the dataframe if they are not timeseries
        # standard dims/coords to keep: 'valid_time' (which translates to 'time')
        # We convert to a pandas DataFrame
        df = point_ds.to_dataframe()

        # Reset index so 'valid_time' (and others) become normal columns
        df = df.reset_index()

        # If latitude/longitude are coords, they might be in the index or columns.
        # Let's ensure they are explicitly present as columns
        if 'latitude' not in df.columns:
            df['latitude'] = actual_lat
        if 'longitude' not in df.columns:
            df['longitude'] = actual_lon

        # Reorder columns to put time and location first
        cols = list(df.columns)
        primary_cols = ['valid_time', 'latitude', 'longitude']
        # Filter primary columns to only those that exist
        primary_cols = [c for c in primary_cols if c in cols]
        other_cols = [c for c in cols if c not in primary_cols]
        df = df[primary_cols + other_cols]

        if output_csv:
            # Ensure output directory exists
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Successfully saved extracted data to '{output_csv}' ({len(df)} rows)")

        return df


def main():
    # --- CONFIGURATION ---
    nc_file = 'Input/wave_2025.nc'
    
    # Specify the target coordinates to extract data for
    # Example: point 7 (91.5000, 22.5000)
    target_latitude = 22.500
    target_longitude = 91.000
    
    # Method to find the grid point:
    # 'nearest' - finds the closest grid point in the NetCDF
    # None - requires an exact coordinate match
    match_method = 'None'
    
    # Output CSV path
    output_file = f"Output/extracted_grid_lat{target_latitude}_lon{target_longitude}.csv"
    # ---------------------

    try:
        df = extract_grid_data(
            nc_path=nc_file,
            target_lat=target_latitude,
            target_lon=target_longitude,
            output_csv=output_file,
            method=match_method
        )
        
        # Display preview of extracted data
        print("\n--- Data Preview (First 5 rows) ---")
        print(df.head().to_string(index=False))
        
        print("\n--- Available Variables Summary ---")
        for col in df.columns:
            if col not in ['valid_time', 'latitude', 'longitude', 'expver', 'number']:
                print(f"Variable '{col}': Mean={df[col].mean():.4f}, Min={df[col].min():.4f}, Max={df[col].max():.4f}")

    except Exception as e:
        print(f"Error during extraction: {e}")


if __name__ == '__main__':
    main()
