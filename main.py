import argparse
from pathlib import Path

import pandas as pd

from gis.mapping import create_cluster_map, create_heatmap, create_species_map
from models.train_model import train
from src.utils import load_config


def run_training(config_path: str) -> None:
    config = load_config(config_path)
    training = config.get("training", {})
    paths = config.get("paths", {})

    artifacts = train(
        data_dir=paths.get("raw_data_dir", "data/raw"),
        output_dir=paths.get("results_dir", "outputs/results"),
        image_size=tuple(training.get("image_size", [224, 224])),
        batch_size=int(training.get("batch_size", 16)),
        epochs=int(training.get("epochs", 8)),
        val_split=float(training.get("validation_split", 0.2)),
        base_model_name=str(training.get("backbone", "efficientnetb0")),
        seed=int(training.get("seed", 42)),
    )
    print("Training artifacts:")
    for key, value in artifacts.items():
        print(f"  - {key}: {value}")


def run_gis(config_path: str) -> None:
    config = load_config(config_path)
    paths = config.get("paths", {})
    maps_dir = Path(paths.get("maps_dir", "outputs/maps"))
    maps_dir.mkdir(parents=True, exist_ok=True)

    sample_path = Path(paths.get("processed_locations", "data/processed/sample_locations.csv"))
    if sample_path.exists():
        frame = pd.read_csv(sample_path)
    else:
        frame = pd.DataFrame(
            [
                {"species": "Aloe Vera", "latitude": 11.12, "longitude": 78.65},
                {"species": "Tulsi", "latitude": 11.15, "longitude": 78.60},
                {"species": "Neem", "latitude": 11.18, "longitude": 78.62},
                {"species": "Ashwagandha", "latitude": 11.11, "longitude": 78.70},
            ]
        )

    species_map = create_species_map(frame, maps_dir / "map.html")
    heatmap = create_heatmap(frame, maps_dir / "heatmap.html")
    cluster_map = create_cluster_map(frame, maps_dir / "clusters.html")

    print("GIS artifacts:")
    print(f"  - Species Map: {species_map}")
    print(f"  - Heatmap: {heatmap}")
    print(f"  - Cluster Map: {cluster_map}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI + GIS Medicinal Plants Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "gis", "all"],
        default="all",
        help="Execution mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"train", "all"}:
        run_training(args.config)
    if args.mode in {"gis", "all"}:
        run_gis(args.config)


if __name__ == "__main__":
    main()
