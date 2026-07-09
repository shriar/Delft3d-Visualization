import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

class WaveBoundaryFileCreator:
    """
    Creates wave boundary files from ERA5 NetCDF data.
    """
    def __init__(self, start_datetime, end_datetime, orientation, input_folder="RA", input_file_path=None, output_folder=None):
        self.start_datetime = np.datetime64(start_datetime)
        self.end_datetime = np.datetime64(end_datetime)
        self.orientation = orientation
        self.input_folder = Path(input_folder)
        self.input_file_path = input_file_path
        self.output_folder = Path(output_folder) if output_folder else self.input_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.start_date = self.start_datetime.astype(datetime)

    def _clip_dataset(self, ds):
        """Clips the dataset to the specified time range."""
        time_mask = (ds["valid_time"] >= self.start_datetime) & (ds["valid_time"] <= self.end_datetime)
        return ds.where(time_mask, drop=True)

    def _get_boundary_data(self, clipped_ds):
        """Extracts and processes data for a given orientation."""
        slicing_map = {
            "north": (slice(None), 0, slice(None)),
            "south": (slice(None), -1, slice(None)),
            "west": (slice(None), slice(None), 0),
            "east": (slice(None), slice(None), -1),
        }

        if self.orientation not in slicing_map:
            raise ValueError(f"Invalid orientation: {self.orientation}. Must be one of {list(slicing_map.keys())}")

        slices = slicing_map[self.orientation]
        
        data = {}
        variables = ["shww", "pp1d", "mdww", "dwww"]
        for var in variables:
            # Slice data, then compute mean over the spatial boundary axis
            boundary_data = clipped_ds[var].values[slices]
            processed_data = np.nanmean(boundary_data, axis=-1)
            
            if var == "dwww":
                processed_data = np.degrees(processed_data)
            
            data[var] = processed_data
            
        return data

    def create_boundary_file(self):
        """
        Loads data, processes it, and writes the boundary file.
        """
        if self.input_file_path:
            input_file = Path(self.input_file_path)
        else:
            input_file = self.input_folder / f"wave_dry_{self.start_date.year}.nc"
            
        if not input_file.exists():
            print(f"Error: Input file not found at {input_file}")
            return

        if input_file.suffix.lower() in ['.grib', '.grb', '.grib2', '.grb2']:
            ds = xr.open_dataset(input_file, engine='cfgrib')
        else:
            ds = xr.open_dataset(input_file)
            
        clipped_ds = self._clip_dataset(ds)

        if clipped_ds.valid_time.size == 0:
            print("No data available for the specified time range.")
            return

        boundary_data = self._get_boundary_data(clipped_ds)

        df = pd.DataFrame({
            "valid_time": clipped_ds["valid_time"].dt.strftime("%Y%m%d.%H%M").values,
            "shww": boundary_data["shww"],
            "pp1d": boundary_data["pp1d"],
            "mdww": boundary_data["mdww"],
            "dwww": boundary_data["dwww"],
        })

        output_filename = self.output_folder / f"wave_{self.orientation}_{self.start_date:%Y%m%d}.bnd"
        
        # Write file with header in one go
        with open(output_filename, 'w') as f:
            f.write('TPAR\n')
            df.to_csv(f, header=False, index=False, sep=' ', lineterminator='\n', float_format='%.2f')

        print(f"File {output_filename} created successfully.")


if __name__ == "__main__":
    # list_startDateTime = ['2025-07-26T00:00:00']
    # list_endDateTime = ['2025-08-12T00:00:00']

    list_startDateTime = ['2025-01-01T00:00:00']
    list_endDateTime = ['2025-01-17T00:00:00']


    orientations = ["south", "west"]

    for orientation in orientations:
        for start_dt, end_dt in zip(list_startDateTime, list_endDateTime):
            creator = WaveBoundaryFileCreator(
                start_datetime=start_dt,
                end_datetime=end_dt,
                orientation=orientation,
                input_file_path="Input/wave_2025.nc",
                output_folder="Output"
            )
            creator.create_boundary_file()
