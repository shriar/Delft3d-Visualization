"""Plot wave location points on an interactive map and optionally export to geopackage."""

import webbrowser
from pathlib import Path
from typing import List, Tuple

import folium
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def read_coordinates(file_path: str) -> List[Tuple[float, float]]:
    """Read coordinates from a location file.
    
    Args:
        file_path: Path to the location file with lon/lat pairs
        
    Returns:
        List of (lon, lat) tuples
    """
    coordinates = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) == 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coordinates.append((lon, lat))
    return coordinates

def create_geodataframe(coordinates: List[Tuple[float, float]]) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from coordinates.
    
    Args:
        coordinates: List of (lon, lat) tuples
        
    Returns:
        GeoDataFrame with point geometries
    """
    geometry = [Point(lon, lat) for lon, lat in coordinates]
    gdf = gpd.GeoDataFrame(
        {
            'point_id': range(1, len(coordinates) + 1),
            'longitude': [coord[0] for coord in coordinates],
            'latitude': [coord[1] for coord in coordinates]
        },
        geometry=geometry,
        crs='EPSG:4326'  # WGS84
    )
    return gdf

def save_to_geopackage(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Save GeoDataFrame to geopackage format.
    
    Args:
        gdf: GeoDataFrame to save
        output_path: Output file path for the geopackage
    """
    gdf.to_file(output_path, driver='GPKG', layer=output_path.split('.')[0])
    print(f"Points saved to geopackage: '{output_path}'")

def create_folium_map(
    coordinates: List[Tuple[float, float]],
    add_polyline: bool = False,
    add_labels: bool = False
    ) -> folium.Map:
    """Create an interactive Folium map with the coordinates.
    
    Args:
        coordinates: List of (lon, lat) tuples
        add_polyline: Whether to connect points with lines
        add_labels: Whether to add numeric labels to points
        
    Returns:
        Folium Map object
    """
    coords = np.array(coordinates)
    lons = coords[:, 0]
    lats = coords[:, 1]
    
    # Calculate center point for map
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    print(f"Center coordinates: Latitude {center_lat:.4f}, Longitude {center_lon:.4f}")
    print(f"Number of points: {len(coordinates)}")
    
    # Create the map centered on the mean location
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    wave_considered_loc = {
        str(i): "ECMWF" for i in range(156, 156)
    }
    wave_considered_loc.update({
        '1': "Monpora-Hatiya channel",
        '2': "Hatiya",
        '3': "Jahangir_char-Noakhali",
        '4': "Sandwip-Jahangir char channel",
        '5': "Bashkhali",
        '6': "Hatiya",
        '7': "Sandwip"
    })

    for i, (lon, lat) in enumerate(coordinates):
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=folium.Popup(f"Point {i+1}<br>{lon:.3f} {lat:.3f}", max_width=200),
            # tooltip=f"Point {i+1} ({lat:.3f}, {lon:.3f})",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.7
        ).add_to(map_obj)
        
        # Add text label above the point
        if add_labels:
            label_text = wave_considered_loc.get(str(i+1), "")
            display_text = f"{i+1} {label_text}" if label_text else f"{i+1}"
            folium.Marker(
                location=[lat + 0.03, lon+0.03],  # Offset slightly north to avoid blocking clicks
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: 12pt; color: black; font-weight: bold; 
                                text-shadow: 1px 1px 2px white, -1px -1px 2px white, 
                                1px -1px 2px white, -1px 1px 2px white;
                                pointer-events: none;">
                        {display_text}
                    </div>
                ''')
            ).add_to(map_obj)
    
    # Connect points with lines if requested
    if add_polyline:
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat in coordinates],
            color='blue',
            weight=2,
            opacity=0.5
        ).add_to(map_obj)
    
    return map_obj


def main():
    """Main function to orchestrate the workflow."""
    # Configuration variables - modify these as needed
    input_file = 'Input/wave_location.loc'
    output_html = "Output/wave_location_considered_points.html"  # None = auto-generate from input_file
    save_gpkg = False  # Set to True to save geopackage
    output_gpkg = None  # None = auto-generate from input_file
    add_polyline = False  # Set to True to connect points with lines
    add_labels = True  # Set to True to add numeric labels to points
    open_browser = True  # Set to False to skip opening browser
    
    # Read coordinates
    print(f"Reading coordinates from '{input_file}'...")
    coordinates = read_coordinates(input_file)
    
    if not coordinates:
        print("Error: No coordinates found in the input file.")
        return
    
    # Save to geopackage if requested
    if save_gpkg:
        gdf = create_geodataframe(coordinates)
        gpkg_path = output_gpkg or str(Path(input_file).with_suffix('.gpkg'))
        save_to_geopackage(gdf, gpkg_path)
    
    # Create and save the map
    map_obj = create_folium_map(
        coordinates,
        add_polyline=add_polyline,
        add_labels=add_labels
    )
    
    html_path = output_html
    map_obj.save(html_path)
    print(f"Map saved as '{html_path}'")
    
    # Open in browser unless disabled
    if open_browser:
        print("Opening map in browser...")
        webbrowser.open(html_path)


if __name__ == '__main__':
    main()
