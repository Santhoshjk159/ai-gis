# AI-Assisted Identification & GIS Mapping of Endangered Medicinal Plants

A production-ready project that combines computer vision and geospatial intelligence to identify endangered medicinal plants from images and map their ecological distribution.

## Project Overview

This repository demonstrates an end-to-end workflow for:
- Plant species identification using a CNN backbone (EfficientNetB0 or ResNet50)
- Data preprocessing and augmentation for robust training
- Ecological Authenticity Index scoring based on model confidence + geospatial context
- GIS visualization with species markers, heatmaps, and DBSCAN clusters
- A Streamlit app for interactive demo and recruiter-friendly presentation

## Tech Stack

- **AI/ML**: TensorFlow, scikit-learn, NumPy
- **Image Processing**: OpenCV, Pillow
- **Data & Analytics**: pandas, matplotlib
- **GIS**: GeoPandas, Folium
- **App Layer**: Streamlit
- **Config/Orchestration**: YAML + Python CLI

## Repository Structure

```text
ai-gis-medicinal-plants/
│── data/
│   ├── raw/
│   ├── processed/
│
│── models/
│   ├── train_model.py
│   ├── predict.py
│
│── src/
│   ├── preprocessing.py
│   ├── utils.py
│
│── gis/
│   ├── mapping.py
│   ├── clustering.py
│
│── notebooks/
│   ├── exploration.ipynb
│
│── outputs/
│   ├── maps/
│   ├── results/
│
│── app/
│   ├── app.py
│
│── requirements.txt
│── README.md
│── config.yaml
│── main.py
```

## Architecture Diagram (Text-Based)

```text
[Plant Images + GPS Metadata]
           |
           v
 [Preprocessing & Augmentation]
           |
           v
 [CNN Backbone: EfficientNet/ResNet]
           |
           +----> [Metrics: Accuracy/Precision/Recall]
           +----> [Confusion Matrix]
           +----> [Predicted Class + Confidence]
                          |
                          v
         [Ecological Authenticity Index]
                          |
                          v
        [GeoPandas + Folium GIS Engine]
                          |
                          v
          [Interactive HTML Maps + App UI]
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd ai-gis-medicinal-plants
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate environment:
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Format (for Training)

Place class-wise plant images in:

```text
data/raw/
  AloeVera/
    image1.jpg
    image2.jpg
  Neem/
    image1.jpg
  Tulsi/
    image1.jpg
```

## How to Run

### 1) Run full CLI pipeline (training + GIS)

```bash
python main.py --mode all --config config.yaml
```

### 2) Run only model training

```bash
python main.py --mode train --config config.yaml
```

### 3) Run only GIS map generation

```bash
python main.py --mode gis --config config.yaml
```

### 4) Run Streamlit demo app

```bash
streamlit run app/app.py
```

## Outputs Generated

After execution, the project generates:
- `outputs/results/model.keras` (trained model)
- `outputs/results/metrics.json` (accuracy, precision, recall, training history)
- `outputs/results/confusion_matrix.png`
- `outputs/maps/map.html` (species map)
- `outputs/maps/heatmap.html`
- `outputs/maps/clusters.html`

## Sample Output Preview

- **Prediction**: `Neem` (confidence: `0.91`)
- **Ecological Authenticity Index**: `0.83`
- **GIS Layer**: Marker + Heatmap + Cluster overlays

## Future Improvements

- Add real ecological raster layers (soil, rainfall, temperature)
- Replace mock suitability with remote-sensing derived indices
- Add model explainability (Grad-CAM)
- Integrate geospatial database (PostGIS)
- Add CI/CD workflow and Docker deployment

## Notes for Interview/Demo

- Clean modular design for AI + GIS integration
- Explicit model metrics and confusion-matrix-based evaluation
- Reproducible configuration using `config.yaml`
- Streamlit UI enables non-technical stakeholder demo
