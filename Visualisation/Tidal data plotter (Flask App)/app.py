"""
Flask Plotter – upload CSV/Excel, pick X vs Y columns, set axis range, and plot.
"""

import io
import os
import base64
import json

import numpy as np

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: str) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)


def _try_parse_dates(series: pd.Series) -> pd.Series:
    """Try converting a Series to datetime; return original if it fails."""
    try:
        return pd.to_datetime(series)
    except Exception:
        return series


def _interpolate_dataframe(df: pd.DataFrame, x_col: str, y_col: str, interval: str) -> pd.DataFrame:
    """Interpolate DataFrame to regular time intervals."""
    new_time_index = pd.date_range(
        start=df[x_col].min(),
        end=df[x_col].max(),
        freq=interval,
    )
    df_interp = df.set_index(x_col).reindex(new_time_index)
    df_interp[y_col] = df_interp[y_col].interpolate()
    df_interp.index.name = x_col
    return df_interp.reset_index()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a file, save it, and return column names + preview."""
    print("\n>>> [POST] /upload - Uploading file...")
    f = request.files.get("file")
    if f is None:
        print("!!! [POST] /upload - Error: No file provided")
        return jsonify(error="No file provided"), 400

    filename = f.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)
    print(f"--- [POST] /upload - Saved file to {filepath}")

    try:
        df = _read_file(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"!!! [POST] /upload - Error reading file: {e}")
        return jsonify(error=f"Error reading file: {e}"), 400

    # Store path in session so we can reuse it later
    session["filepath"] = filepath

    # Try to detect column types for the frontend
    col_info = []
    for col in df.columns:
        sample = df[col].dropna().head(5).tolist()
        sample_str = [str(s) for s in sample]
        col_info.append({"name": col, "sample": sample_str})

    print(f"<<< [POST] /upload - Success: {filename} uploaded. Rows: {len(df)}")
    return jsonify(columns=col_info, rows=len(df), filename=filename)


@app.route("/plot", methods=["POST"])
def plot():
    """Generate an interactive matplotlib plot via mpld3."""
    data = request.get_json()
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        print("!!! [POST] /plot - Error: No file uploaded yet")
        return jsonify(error="No file uploaded yet."), 400

    x_col = data.get("x_col")
    y_col = data.get("y_col")
    x_min = data.get("x_min")
    x_max = data.get("x_max")
    x_label = data.get("x_label", "")
    y_label = data.get("y_label", "")
    title = data.get("title", "")
    line_color = data.get("line_color", "#2196F3")
    line_width = float(data.get("line_width", 2))
    show_grid = data.get("show_grid", True)
    show_legend = data.get("show_legend", True)
    show_markers = data.get("show_markers", False)
    marker_size = float(data.get("marker_size", 5))
    show_annotations = data.get("show_annotations", False)
    specified_indices_str = data.get("specified_indices", "").strip()
    interpolate = data.get("interpolate", False)
    interp_interval = data.get("interp_interval", "60min")
    tick_interval = data.get("tick_interval", "auto")
    ann_fontsize = float(data.get("ann_fontsize", 10))
    ann_offset = float(data.get("ann_offset", 3))
    fig_width = float(data.get("fig_width", 14))
    fig_height = float(data.get("fig_height", 6))

    print("\n>>> [POST] /plot - Generating plot...")
    print(f"    - filepath: {filepath}")
    print(f"    - x_col: '{x_col}', y_col: '{y_col}'")
    print(f"    - x_min: {x_min}, x_max: {x_max}")
    print(f"    - title: '{title}', x_label: '{x_label}', y_label: '{y_label}'")
    print(f"    - line_color: {line_color}, line_width: {line_width}, show_grid: {show_grid}, show_legend: {show_legend}")
    print(f"    - show_markers: {show_markers}, show_annotations: {show_annotations}")

    try:
        df = _read_file(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"!!! [POST] /plot - Error reading file: {e}")
        return jsonify(error=f"Error reading file: {e}"), 400

    if x_col not in df.columns or y_col not in df.columns:
        print(f"!!! [POST] /plot - Error: selected columns not found in {df.columns.tolist()}")
        return jsonify(error="Selected columns not found in data."), 400

    # Parse X as dates if possible
    df[x_col] = _try_parse_dates(df[x_col])
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])

    # ---- Interpolation (Always done on original data before time clipping) ----
    if interpolate and is_datetime and interp_interval:
        try:
            df = _interpolate_dataframe(df, x_col, y_col, interp_interval)
        except Exception as e:
            return jsonify(error=f"Interpolation failed: {e}"), 400

    # ---- Filter by range ----
    if x_min and x_max:
        if is_datetime:
            try:
                # Normalize "yyyy-mm-dd, HH:MM" to "yyyy-mm-dd HH:MM"
                norm_min = str(x_min).replace(",", " ").replace("  ", " ")
                norm_max = str(x_max).replace(",", " ").replace("  ", " ")
                x_min_val = pd.to_datetime(norm_min)
                x_max_val = pd.to_datetime(norm_max)
            except Exception:
                return jsonify(error="Invalid datetime range. Use format: yyyy-mm-dd, HH:MM"), 400
        else:
            try:
                x_min_val = float(x_min)
                x_max_val = float(x_max)
            except Exception:
                return jsonify(error="Invalid numeric range."), 400
        mask = (df[x_col] >= x_min_val) & (df[x_col] <= x_max_val)
        df = df.loc[mask]

    if df.empty:
        return jsonify(error="No data in the selected range."), 400

    # ---- Parse specified indices (1-based from user) ----
    specified_idx = None
    if specified_indices_str:
        try:
            specified_idx = set(
                int(x.strip()) - 1 for x in specified_indices_str.split(",") if x.strip()
            )
        except ValueError:
            return jsonify(error="Invalid indices. Use comma-separated numbers like: 1, 5, 10"), 400

    # Calculate Y-axis range for scaling the text offset
    y_min_val = df[y_col].min()
    y_max_val = df[y_col].max()
    y_range = y_max_val - y_min_val if y_max_val != y_min_val else 1.0
    y_data_offset = (ann_offset / 100.0) * y_range

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.plot(df[x_col], df[y_col], '-', color=line_color, linewidth=line_width, label=y_col)

    # Annotated markers (like reference script: red dots + Time/Index labels)
    if show_markers or show_annotations:
        for i, row in df.iterrows():
            # If specific indices given, skip all others
            if specified_idx is not None and i not in specified_idx:
                continue
            ax.plot(row[x_col], row[y_col], 'ro', markersize=marker_size)
            if show_annotations:
                if is_datetime:
                    label_text = f"Time: {row[x_col].strftime('%Y-%m-%d %H:%M')}\nIndex: {i + 1}"
                else:
                    label_text = f"X: {row[x_col]}\nIndex: {i + 1}"
                # Use ax.text in data coordinates for flawless mpld3 rendering.
                # Convert Timestamp to float representation (date2num) to avoid mpld3 json serializing error.
                x_pos = mdates.date2num(row[x_col]) if is_datetime else row[x_col]
                ax.text(
                    x_pos,
                    row[y_col] + y_data_offset,
                    label_text,
                    fontsize=ann_fontsize,
                    fontweight="bold",
                    color="darkred",
                    alpha=0.9,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

    if is_datetime:
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y, %H:%M"))
        # ---- X-axis tick interval ----
        if tick_interval and tick_interval != "auto":
            try:
                interval_hours = float(tick_interval)
                if interval_hours >= 24:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=int(interval_hours // 24)))
                else:
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(interval_hours)))
            except (ValueError, TypeError):
                pass  # fall back to auto

    ax.set_xlabel(x_label if x_label else x_col, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label if y_label else y_col, fontsize=13, fontweight="bold")
    ax.set_title(title or f"{y_col} vs {x_col}", fontsize=16, fontweight="bold")

    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    if show_legend:
        ax.legend(fontsize=11)
    fig.tight_layout()

    # Convert to interactive HTML via mpld3
    html_str = mpld3.fig_to_html(fig)
    plt.close(fig)

    print(f"<<< [POST] /plot - Success: Plot rendered successfully (HTML size: {len(html_str)} chars)")
    return jsonify(plot_html=html_str)


@app.route("/export", methods=["POST"])
def export():
    """Export the processed dataset or current plot in PNG/PDF/CSV format."""
    data = request.get_json()
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        print("!!! [POST] /export - Error: No file uploaded yet")
        return jsonify(error="No file uploaded yet."), 400

    export_format = data.get("format", "png") # "png", "pdf", "csv"

    x_col = data.get("x_col")
    y_col = data.get("y_col")
    x_min = data.get("x_min")
    x_max = data.get("x_max")
    x_label = data.get("x_label", "")
    y_label = data.get("y_label", "")
    title = data.get("title", "")
    line_color = data.get("line_color", "#2196F3")
    line_width = float(data.get("line_width", 2))
    show_grid = data.get("show_grid", True)
    show_legend = data.get("show_legend", True)
    show_markers = data.get("show_markers", False)
    marker_size = float(data.get("marker_size", 5))
    show_annotations = data.get("show_annotations", False)
    specified_indices_str = data.get("specified_indices", "").strip()
    interpolate = data.get("interpolate", False)
    interp_interval = data.get("interp_interval", "60min")
    tick_interval = data.get("tick_interval", "auto")
    ann_fontsize = float(data.get("ann_fontsize", 10))
    ann_offset = float(data.get("ann_offset", 3))
    fig_width = float(data.get("fig_width", 14))
    fig_height = float(data.get("fig_height", 6))

    print(f"\n>>> [POST] /export - Exporting as {export_format.upper()}...")
    print(f"    - filepath: {filepath}")
    print(f"    - x_col: '{x_col}', y_col: '{y_col}'")
    print(f"    - title: '{title}', x_label: '{x_label}', y_label: '{y_label}'")

    try:
        df = _read_file(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"!!! [POST] /export - Error reading file: {e}")
        return jsonify(error=f"Error reading file: {e}"), 400

    if x_col not in df.columns or y_col not in df.columns:
        print(f"!!! [POST] /export - Error: selected columns not found in {df.columns.tolist()}")
        return jsonify(error="Selected columns not found in data."), 400

    # Parse X as dates if possible
    df[x_col] = _try_parse_dates(df[x_col])
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])

    # ---- Interpolation (Always done on original data before time clipping) ----
    if interpolate and is_datetime and interp_interval:
        try:
            df = _interpolate_dataframe(df, x_col, y_col, interp_interval)
        except Exception as e:
            return jsonify(error=f"Interpolation failed: {e}"), 400

    # ---- Filter by range ----
    if x_min and x_max:
        if is_datetime:
            try:
                norm_min = str(x_min).replace(",", " ").replace("  ", " ")
                norm_max = str(x_max).replace(",", " ").replace("  ", " ")
                x_min_val = pd.to_datetime(norm_min)
                x_max_val = pd.to_datetime(norm_max)
            except Exception:
                return jsonify(error="Invalid datetime range. Use format: yyyy-mm-dd, HH:MM"), 400
        else:
            try:
                x_min_val = float(x_min)
                x_max_val = float(x_max)
            except Exception:
                return jsonify(error="Invalid numeric range."), 400
        mask = (df[x_col] >= x_min_val) & (df[x_col] <= x_max_val)
        df = df.loc[mask]

    if df.empty:
        return jsonify(error="No data in the selected range."), 400

    # If exporting data
    if export_format == "csv":
        csv_data = df.to_csv(index=False)
        b64_data = base64.b64encode(csv_data.encode()).decode()
        return jsonify(data=b64_data, filename="processed_data.csv", mimetype="text/csv")

    # ---- Parse specified indices (1-based from user) ----
    specified_idx = None
    if specified_indices_str:
        try:
            specified_idx = set(
                int(x.strip()) - 1 for x in specified_indices_str.split(",") if x.strip()
            )
        except ValueError:
            return jsonify(error="Invalid indices. Use comma-separated numbers like: 1, 5, 10"), 400

    # Calculate Y-axis range for scaling the text offset
    y_min_val = df[y_col].min()
    y_max_val = df[y_col].max()
    y_range = y_max_val - y_min_val if y_max_val != y_min_val else 1.0
    y_data_offset = (ann_offset / 100.0) * y_range

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.plot(df[x_col], df[y_col], '-', color=line_color, linewidth=line_width, label=y_col)

    # Annotated markers
    if show_markers or show_annotations:
        for i, row in df.iterrows():
            if specified_idx is not None and i not in specified_idx:
                continue
            ax.plot(row[x_col], row[y_col], 'ro', markersize=marker_size)
            if show_annotations:
                if is_datetime:
                    label_text = f"Time: {row[x_col].strftime('%Y-%m-%d %H:%M')}\nIndex: {i + 1}"
                else:
                    label_text = f"X: {row[x_col]}\nIndex: {i + 1}"
                # Use ax.text in data coordinates. Convert Timestamp to float representation.
                x_pos = mdates.date2num(row[x_col]) if is_datetime else row[x_col]
                ax.text(
                    x_pos,
                    row[y_col] + y_data_offset,
                    label_text,
                    fontsize=ann_fontsize,
                    fontweight="bold",
                    color="darkred",
                    alpha=0.9,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

    if is_datetime:
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y, %H:%M"))
        if tick_interval and tick_interval != "auto":
            try:
                interval_hours = float(tick_interval)
                if interval_hours >= 24:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=int(interval_hours // 24)))
                else:
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(interval_hours)))
            except (ValueError, TypeError):
                pass

    ax.set_xlabel(x_label if x_label else x_col, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label if y_label else y_col, fontsize=13, fontweight="bold")
    ax.set_title(title or f"{y_col} vs {x_col}", fontsize=16, fontweight="bold")

    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    if show_legend:
        ax.legend(fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    if export_format == "pdf":
        fig.savefig(buf, format="pdf", bbox_inches="tight")
        filename = "plot.pdf"
        mimetype = "application/pdf"
    else:
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        filename = "plot.png"
        mimetype = "image/png"
    plt.close(fig)
    buf.seek(0)
    b64_data = base64.b64encode(buf.read()).decode()

    print(f"<<< [POST] /export - Success: Exported {filename} ({len(b64_data)} bytes base64)")
    return jsonify(data=b64_data, filename=filename, mimetype=mimetype)


@app.route("/column_range", methods=["POST"])
def column_range():
    """Return min/max of a column (for setting range sliders)."""
    data = request.get_json()
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        print("!!! [POST] /column_range - Error: No file uploaded yet")
        return jsonify(error="No file uploaded yet."), 400

    col = data.get("column")
    print(f"\n>>> [POST] /column_range - Fetching range for column: '{col}'")
    
    try:
        df = _read_file(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"!!! [POST] /column_range - Error reading file: {e}")
        return jsonify(error=f"Error reading file: {e}"), 400

    if col not in df.columns:
        print(f"!!! [POST] /column_range - Error: Column '{col}' not found")
        return jsonify(error="Column not found."), 400

    series = _try_parse_dates(df[col])
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    if is_datetime:
        print(f"<<< [POST] /column_range - Success (Datetime): min={series.min()}, max={series.max()}")
        return jsonify(
            min=str(series.min()),
            max=str(series.max()),
            is_datetime=True,
        )
    else:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            print("<<< [POST] /column_range - Success: Column is non-numeric/empty")
            return jsonify(min="", max="", is_datetime=False)
        print(f"<<< [POST] /column_range - Success (Numeric): min={numeric.min()}, max={numeric.max()}")
        return jsonify(
            min=str(numeric.min()),
            max=str(numeric.max()),
            is_datetime=False,
        )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
