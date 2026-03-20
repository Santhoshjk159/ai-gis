import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def cluster_species_locations(
    frame: pd.DataFrame,
    eps_km: float = 5.0,
    min_samples: int = 3,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        result["cluster_id"] = -1
        return result

    coords = frame[[lat_col, lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords)

    earth_radius_km = 6371.0088
    eps_radians = eps_km / earth_radius_km

    model = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords_rad)

    clustered = frame.copy()
    clustered["cluster_id"] = labels
    return clustered
