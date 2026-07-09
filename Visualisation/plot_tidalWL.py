import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def main() -> None:
    # 1. Read CSV and parse columns
    print("Reading and parsing Sandwip water level CSV data...")
    df = pd.read_csv('Input/20260726_WL_Sandwip.csv')
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['date and time'])
    df['ZWL'] = df['water level (m)'].astype(float)

    # 2. Initialize the plot with an extra large canvas to prevent label overlapping
    plt.figure(figsize=(36, 18))
    plt.plot(df['Time'], df['ZWL'], 'b-', linewidth=3.0, label='Water Level')

    # 3. Mark and annotate every single hourly measurement point with large bold dark red text
    for idx, row in df.iterrows():
        plt.plot(row['Time'], row['ZWL'], 'ro', markersize=10)
        plt.annotate(f"Time: {row['Time'].strftime('%Y-%m-%d %H:%M')}\nIndex: {idx + 1}",
                    xy=(row['Time'], row['ZWL']),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight='bold',
                    color='darkred',
                    alpha=1.0)

    # 4. Format labels, grid, and dates on x-axis with large fonts
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 12h interval displays dates beautifully on wide canvas
    plt.xlabel('Time', fontsize=18, fontweight='bold')
    plt.ylabel('Water Level (m)', fontsize=18, fontweight='bold')
    plt.title("20260726_WL_Sandwip - Whole Series with Every Hourly Point Annotated", fontsize=24, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 5. Save figure and display
    output_path = "Output/figure/tidalWL_whole_marked.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
