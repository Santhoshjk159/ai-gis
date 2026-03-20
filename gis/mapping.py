from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import HeatMap

from gis.clustering import cluster_species_locations


def create_geodataframe(
    frame: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame[lon_col], frame[lat_col]),
        crs="EPSG:4326",
    )
    return gdf


def _base_map(center_lat: float, center_lon: float, zoom_start: int = 7) -> folium.Map:
    return folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)


def create_species_map(
    frame: pd.DataFrame,
    output_html: str | Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    species_col: str = "species",
) -> str:
    if frame.empty:
        raise ValueError("Cannot create map from empty dataframe.")

    center_lat = float(frame[lat_col].mean())
    center_lon = float(frame[lon_col].mean())
    map_obj = _base_map(center_lat, center_lon)

    for _, row in frame.iterrows():
        popup = f"Species: {row.get(species_col, 'Unknown')}"
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup,
            tooltip=row.get(species_col, "Plant"),
        ).add_to(map_obj)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_html))
    return str(output_html)


def create_heatmap(
    frame: pd.DataFrame,
    output_html: str | Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> str:
    if frame.empty:
        raise ValueError("Cannot create heatmap from empty dataframe.")

    center_lat = float(frame[lat_col].mean())
    center_lon = float(frame[lon_col].mean())
    map_obj = _base_map(center_lat, center_lon)
    heat_points = frame[[lat_col, lon_col]].values.tolist()
    HeatMap(heat_points, radius=20, blur=18).add_to(map_obj)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_html))
    return str(output_html)


def create_cluster_map(
    frame: pd.DataFrame,
    output_html: str | Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> str:
    clustered = cluster_species_locations(frame, lat_col=lat_col, lon_col=lon_col)
    center_lat = float(clustered[lat_col].mean())
    center_lon = float(clustered[lon_col].mean())
    map_obj = _base_map(center_lat, center_lon)

    color_palette = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "black"]

    for _, row in clustered.iterrows():
        cluster_id = int(row["cluster_id"])
        color = "gray" if cluster_id == -1 else color_palette[cluster_id % len(color_palette)]
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster: {cluster_id}",
        ).add_to(map_obj)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_html))
    return str(output_html)
