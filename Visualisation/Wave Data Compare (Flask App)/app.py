"""
Flask app to compare ECMWF wave data with Delft3D wave data.
Run with: uv run --with flask --with xarray --with netcdf4 --with numpy --with pandas python app.py
"""

import io
import os
import numpy as np
import pandas as pd
import xarray as xr
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# --- Global dataset state ---
ds_ecmwf = None
ds_delft = None
loaded_ecmwf_path = None
loaded_delft_path = None

def find_nc_files():
    """Scan 'Input' and current directory for NetCDF files."""
    directories = ["Input", "."]
    results = []
    seen_paths = set()
    for d in directories:
        if os.path.exists(d):
            for name in os.listdir(d):
                if name.endswith(".nc"):
                    rel_path = os.path.join(d, name)
                    abs_path = os.path.abspath(rel_path)
                    if abs_path not in seen_paths:
                        seen_paths.add(abs_path)
                        results.append({
                            "name": rel_path.replace("./", ""),
                            "path": abs_path
                        })
    return results

def auto_detect_files(files_list):
    """Auto-detect which file is ECMWF and which is Delft3D."""
    ecmwf_file = None
    delft_file = None
    
    # 1. Try matching by keyword in filename
    for f in files_list:
        name_lower = f["name"].lower()
        if "ecmwf" in name_lower and not ecmwf_file:
            ecmwf_file = f["path"]
        elif "delft3d" in name_lower and not delft_file:
            delft_file = f["path"]

    # 2. Try looking inside file coordinates/dims if not found
    for f in files_list:
        if f["path"] == ecmwf_file or f["path"] == delft_file:
            continue
        try:
            with xr.open_dataset(f["path"]) as ds:
                if 'stations' in ds.dims or 'station_name' in ds.data_vars:
                    if not delft_file:
                        delft_file = f["path"]
                elif 'latitude' in ds.dims or 'lat' in ds.dims:
                    if not ecmwf_file:
                        ecmwf_file = f["path"]
        except Exception:
            pass

    # 3. Fallbacks if one or both are still missing
    for f in files_list:
        if not ecmwf_file and f["path"] != delft_file:
            ecmwf_file = f["path"]
        elif not delft_file and f["path"] != ecmwf_file:
            delft_file = f["path"]

    # 4. Ultimate fallback to local hardcoded names if no files scanned
    if not ecmwf_file and os.path.exists("wave_2025.nc"):
        ecmwf_file = os.path.abspath("wave_2025.nc")
    if not delft_file and os.path.exists("wavh-wave_20250726.nc"):
        delft_file = os.path.abspath("wavh-wave_20250726.nc")

    return ecmwf_file, delft_file

def load_data(ec_path, dl_path):
    """Load NetCDF files into global datasets, renaming dimensions if necessary."""
    global ds_ecmwf, ds_delft, loaded_ecmwf_path, loaded_delft_path
    
    if not ec_path or not dl_path:
        raise ValueError("Both ECMWF and Delft3D file paths must be specified.")

    if ds_ecmwf is None or loaded_ecmwf_path != ec_path:
        print(f"Loading ECMWF dataset from: {ec_path}")
        ds = xr.open_dataset(ec_path)
        if "valid_time" in ds.coords or "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        ds_ecmwf = ds
        loaded_ecmwf_path = ec_path
        
    if ds_delft is None or loaded_delft_path != dl_path:
        print(f"Loading Delft3D dataset from: {dl_path}")
        ds_delft = xr.open_dataset(dl_path)
        loaded_delft_path = dl_path

def ensure_datasets_loaded():
    """Ensure datasets are loaded, auto-detecting files if not already specified."""
    if ds_ecmwf is None or ds_delft is None:
        files = find_nc_files()
        ec_path, dl_path = auto_detect_files(files)
        load_data(ec_path, dl_path)


def get_ecmwf_meta():
    """Extract ECMWF metadata: variables, grid points, time range."""
    variables = {}
    for v in ds_ecmwf.data_vars:
        attrs = ds_ecmwf[v].attrs
        variables[v] = {
            "long_name": attrs.get("long_name", v),
            "units": attrs.get("units", ""),
        }

    lats = ds_ecmwf.latitude.values.tolist()
    lons = ds_ecmwf.longitude.values.tolist()
    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append({"lat": lat, "lon": lon, "label": f"{lat}°N, {lon}°E"})

    time_min = str(ds_ecmwf.time.values[0])[:19]
    time_max = str(ds_ecmwf.time.values[-1])[:19]

    return {
        "variables": variables,
        "grid_points": grid_points,
        "time_range": {"min": time_min, "max": time_max},
    }


def get_delft3d_meta():
    """Extract Delft3D metadata: variables, stations, time range."""
    skip = {"station_x_coordinate", "station_y_coordinate", "station_name", "station_id"}
    variables = {}
    for v in ds_delft.data_vars:
        if v in skip:
            continue
        attrs = ds_delft[v].attrs
        variables[v] = {
            "long_name": attrs.get("long_name", v),
            "units": attrs.get("units", ""),
        }

    stations = []
    for i in range(ds_delft.dims["stations"]):
        raw_name = ds_delft["station_name"].values[i]
        name = raw_name.decode().strip() if isinstance(raw_name, bytes) else str(raw_name).strip()
        x = float(ds_delft["station_x_coordinate"].values[i])
        y = float(ds_delft["station_y_coordinate"].values[i])
        label = f"{name} ({y:.3f}°N, {x:.3f}°E)" if name and name != "Station" else f"Stn {i+1} ({y:.3f}°N, {x:.3f}°E)"
        stations.append({"index": i, "x": x, "y": y, "name": name, "label": label})

    time_min = str(ds_delft.time.values[0])[:19]
    time_max = str(ds_delft.time.values[-1])[:19]

    return {
        "variables": variables,
        "stations": stations,
        "time_range": {"min": time_min, "max": time_max},
    }


def compute_overlap():
    """Compute the overlapping time range between the two datasets."""
    ec_min = np.datetime64(ds_ecmwf.time.values[0])
    ec_max = np.datetime64(ds_ecmwf.time.values[-1])
    dl_min = np.datetime64(ds_delft.time.values[0])
    dl_max = np.datetime64(ds_delft.time.values[-1])
    overlap_min = max(ec_min, dl_min)
    overlap_max = min(ec_max, dl_max)
    return str(overlap_min)[:19], str(overlap_max)[:19]


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/files")
def get_files():
    """API endpoint to get available NetCDF files and currently active loaded paths."""
    files = find_nc_files()
    ec_detected, dl_detected = auto_detect_files(files)
    
    # Resolve loaded paths to user-friendly names if possible
    active_ec = None
    active_dl = None
    for f in files:
        if f["path"] == loaded_ecmwf_path:
            active_ec = f["path"]
        if f["path"] == loaded_delft_path:
            active_dl = f["path"]

    return jsonify({
        "files": files,
        "active": {
            "ecmwf": active_ec or loaded_ecmwf_path or ec_detected,
            "delft3d": active_dl or loaded_delft_path or dl_detected
        }
    })

@app.route("/api/load_files", methods=["POST"])
def api_load_files():
    """API endpoint to load specific ECMWF and Delft3D files."""
    data = request.json or {}
    ec_path = data.get("ecmwf_path")
    dl_path = data.get("delft3d_path")
    
    if not ec_path or not dl_path:
        return jsonify({"error": "Both ecmwf_path and delft3d_path are required."}), 400
        
    try:
        load_data(ec_path, dl_path)
        return jsonify({"status": "success", "message": "Datasets loaded successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metadata")
def metadata():
    try:
        ensure_datasets_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to load datasets: {str(e)}"}), 500

    ecmwf_meta = get_ecmwf_meta()
    delft3d_meta = get_delft3d_meta()
    overlap_min, overlap_max = compute_overlap()
    return jsonify({
        "ecmwf": ecmwf_meta,
        "delft3d": delft3d_meta,
        "overlap": {"min": overlap_min, "max": overlap_max},
    })


@app.route("/api/timeseries")
def timeseries():
    try:
        ensure_datasets_loaded()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ECMWF params
    ec_var = request.args.get("ec_var")
    ec_lat = request.args.get("ec_lat", type=float)
    ec_lon = request.args.get("ec_lon", type=float)

    # Delft3D params
    dl_var = request.args.get("dl_var")
    dl_station = request.args.get("dl_station", type=int)

    # Time range
    t_min = request.args.get("t_min")
    t_max = request.args.get("t_max")

    result = {}

    # Extract ECMWF series
    if ec_var and ec_lat is not None and ec_lon is not None:
        try:
            sel = ds_ecmwf[ec_var].sel(latitude=ec_lat, longitude=ec_lon, method="nearest")
            if t_min and t_max:
                sel = sel.sel(time=slice(t_min, t_max))
            times = [str(t)[:19] for t in sel.time.values]
            values = sel.values.tolist()
            # Replace NaN with None for JSON
            values = [None if (v != v) else v for v in values]
            attrs = ds_ecmwf[ec_var].attrs
            result["ecmwf"] = {
                "times": times,
                "values": values,
                "var": ec_var,
                "long_name": attrs.get("long_name", ec_var),
                "units": attrs.get("units", ""),
                "location": f"{ec_lat}°N, {ec_lon}°E",
            }
        except Exception as e:
            print(f"Error extracting ECMWF series: {e}")

    # Extract Delft3D series
    if dl_var and dl_station is not None:
        try:
            sel = ds_delft[dl_var].isel(stations=dl_station)
            if t_min and t_max:
                sel = sel.sel(time=slice(t_min, t_max))
            times = [str(t)[:19] for t in sel.time.values]
            values = sel.values.tolist()
            values = [None if (v != v) else v for v in values]
            attrs = ds_delft[dl_var].attrs

            # Station label
            raw = ds_delft["station_name"].values[dl_station]
            sname = raw.decode().strip() if isinstance(raw, bytes) else str(raw).strip()
            sx = float(ds_delft["station_x_coordinate"].values[dl_station])
            sy = float(ds_delft["station_y_coordinate"].values[dl_station])
            loc_label = f"{sname} ({sy:.3f}°N, {sx:.3f}°E)" if sname and sname != "Station" else f"Stn {dl_station+1} ({sy:.3f}°N, {sx:.3f}°E)"

            result["delft3d"] = {
                "times": times,
                "values": values,
                "var": dl_var,
                "long_name": attrs.get("long_name", dl_var),
                "units": attrs.get("units", ""),
                "location": loc_label,
            }
        except Exception as e:
            print(f"Error extracting Delft3D series: {e}")

    return jsonify(result)


@app.route("/api/statistics")
def statistics():
    """Compute comparison statistics for overlapping time steps."""
    try:
        ensure_datasets_loaded()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ec_var = request.args.get("ec_var")
    ec_lat = request.args.get("ec_lat", type=float)
    ec_lon = request.args.get("ec_lon", type=float)
    dl_var = request.args.get("dl_var")
    dl_station = request.args.get("dl_station", type=int)
    t_min = request.args.get("t_min")
    t_max = request.args.get("t_max")

    if not all([ec_var, ec_lat is not None, ec_lon is not None, dl_var, dl_station is not None]):
        return jsonify({"error": "Missing parameters"}), 400

    try:
        # Get both series
        ec_sel = ds_ecmwf[ec_var].sel(latitude=ec_lat, longitude=ec_lon, method="nearest")
        dl_sel = ds_delft[dl_var].isel(stations=dl_station)

        if t_min and t_max:
            ec_sel = ec_sel.sel(time=slice(t_min, t_max))
            dl_sel = dl_sel.sel(time=slice(t_min, t_max))

        # Align on common time steps
        common_times = np.intersect1d(ec_sel.time.values, dl_sel.time.values)
        if len(common_times) == 0:
            return jsonify({"error": "No overlapping time steps found"}), 400

        ec_vals = ec_sel.sel(time=common_times).values
        dl_vals = dl_sel.sel(time=common_times).values

        # Remove NaN pairs
        mask = ~(np.isnan(ec_vals) | np.isnan(dl_vals))
        ec_clean = ec_vals[mask]
        dl_clean = dl_vals[mask]

        if len(ec_clean) == 0:
            return jsonify({"error": "No valid data points after removing NaNs"}), 400

        # Statistics
        diff = ec_clean - dl_clean
        bias = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        corr = float(np.corrcoef(ec_clean, dl_clean)[0, 1]) if len(ec_clean) > 1 else None
        n = int(len(ec_clean))

        # Scatter data (subsample if large)
        step = max(1, len(ec_clean) // 500)
        scatter_ec = ec_clean[::step].tolist()
        scatter_dl = dl_clean[::step].tolist()

        return jsonify({
            "bias": round(bias, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "correlation": round(corr, 4) if corr is not None else None,
            "n_points": n,
            "scatter": {"ecmwf": scatter_ec, "delft3d": scatter_dl},
        })
    except Exception as e:
        return jsonify({"error": f"Failed to compute statistics: {str(e)}"}), 500


@app.route("/api/loc_map")
def loc_map():
    """Endpoint to check and return locations from all found .loc files."""
    search_dirs = ["Input", "."]
    loc_files = {}
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        try:
            for f in os.listdir(d):
                if f.endswith(".loc"):
                    full_path = os.path.normpath(os.path.join(d, f))
                    # Avoid duplicate filenames
                    if f not in loc_files:
                        loc_files[f] = full_path
        except Exception as e:
            print(f"Error scanning directory {d}: {e}")

    if not loc_files:
        return jsonify({"exists": False, "files": {}})

    results = {}
    for name, path in loc_files.items():
        points = []
        try:
            with open(path, "r") as f:
                idx = 1
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                points.append({
                                    "index": idx,
                                    "lon": lon,
                                    "lat": lat
                                })
                                idx += 1
                            except ValueError:
                                continue
            results[name] = {
                "path": path,
                "points": points
            }
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return jsonify({
        "exists": len(results) > 0,
        "files": results
    })


@app.route("/api/export")
def export_csv():
    """Export both time series interpolated to a common regular time interval."""
    try:
        ensure_datasets_loaded()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ec_var = request.args.get("ec_var")
    ec_lat = request.args.get("ec_lat", type=float)
    ec_lon = request.args.get("ec_lon", type=float)
    dl_var = request.args.get("dl_var")
    dl_station = request.args.get("dl_station", type=int)
    t_min = request.args.get("t_min")
    t_max = request.args.get("t_max")
    interval_min = request.args.get("interval_minutes", default=60, type=int)

    try:
        # --- Extract ECMWF series ---
        ec_times_raw, ec_vals_raw = None, None
        ec_label = "ECMWF"
        if ec_var and ec_lat is not None and ec_lon is not None:
            sel = ds_ecmwf[ec_var].sel(latitude=ec_lat, longitude=ec_lon, method="nearest")
            if t_min and t_max:
                sel = sel.sel(time=slice(t_min, t_max))
            ec_times_raw = sel.time.values.astype("datetime64[ns]")
            ec_vals_raw = sel.values.astype(float)
            ec_label = f"ECMWF_{ec_var}"

        # --- Extract Delft3D series ---
        dl_times_raw, dl_vals_raw = None, None
        dl_label = "Delft3D"
        if dl_var and dl_station is not None:
            sel = ds_delft[dl_var].isel(stations=dl_station)
            if t_min and t_max:
                sel = sel.sel(time=slice(t_min, t_max))
            dl_times_raw = sel.time.values.astype("datetime64[ns]")
            dl_vals_raw = sel.values.astype(float)
            dl_label = f"Delft3D_{dl_var}"

        if ec_times_raw is None and dl_times_raw is None:
            return jsonify({"error": "No data to export"}), 400

        # --- Determine the common time range ---
        t_start_candidates = []
        t_end_candidates = []
        if ec_times_raw is not None and len(ec_times_raw) > 0:
            t_start_candidates.append(ec_times_raw[0])
            t_end_candidates.append(ec_times_raw[-1])
        if dl_times_raw is not None and len(dl_times_raw) > 0:
            t_start_candidates.append(dl_times_raw[0])
            t_end_candidates.append(dl_times_raw[-1])

        common_start = max(t_start_candidates)
        common_end = min(t_end_candidates)

        if common_start >= common_end:
            return jsonify({"error": "No overlapping time range between the two datasets"}), 400

        # --- Build the regular time grid ---
        common_grid = np.arange(
            common_start,
            common_end + np.timedelta64(1, "s"),  # inclusive end
            np.timedelta64(interval_min, "m"),
        )

        # Helper: epoch seconds for interpolation
        def to_epoch(dt_arr):
            return (dt_arr - np.datetime64("1970-01-01T00:00:00", "ns")) / np.timedelta64(1, "s")

        grid_epoch = to_epoch(common_grid)

        # --- Interpolate each series onto the common grid ---
        def interpolate_series(times, values, grid_ep):
            """Linear interpolation; NaN outside original range."""
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                return np.full(len(grid_ep), np.nan)
            ep = to_epoch(times[mask])
            vals = values[mask]
            # Sort by time just in case
            order = np.argsort(ep)
            ep = ep[order]
            vals = vals[order]
            interpolated = np.interp(grid_ep, ep, vals, left=np.nan, right=np.nan)
            return interpolated

        ec_interp = None
        dl_interp = None

        if ec_times_raw is not None:
            ec_interp = interpolate_series(ec_times_raw, ec_vals_raw, grid_epoch)
        if dl_times_raw is not None:
            dl_interp = interpolate_series(dl_times_raw, dl_vals_raw, grid_epoch)

        # --- Build CSV ---
        time_strings = [str(t)[:19].replace("T", " ") for t in common_grid]

        df_dict = {"Time": time_strings}
        if ec_interp is not None:
            df_dict[ec_label] = ec_interp
        if dl_interp is not None:
            df_dict[dl_label] = dl_interp

        df = pd.DataFrame(df_dict)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_content = buf.getvalue()

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=timeseries_export_{interval_min}min.csv"},
        )

    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Wave Comparison App...")
    # Initial scan on startup
    try:
        ensure_datasets_loaded()
        print(f"Auto-selected ECMWF: {loaded_ecmwf_path}")
        print(f"Auto-selected Delft3D: {loaded_delft_path}")
    except Exception as e:
        print(f"Startup warning: {e}")
    app.run(debug=True, port=5000)
