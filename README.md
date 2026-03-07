# SANCHARi 🛰️🛣️

**Satellite Road Extraction Pipeline — V4**

> **Mission:** Democratizing high-quality satellite analytics. SANCHARi provides a robust, open-source pipeline for extracting road networks from satellite imagery (NAIP, Sentinel-2, or local GeoTIFFs), enabling accessible mapping for disaster relief, urban planning, and developing regions where vector maps are outdated or missing.

![SANCHARi V4 Overlay](predicted/embed/v4_sat_overlay_1.jpg)

---

## 🚀 Project Evolution: V1 → V4

| Feature | V1 (Baseline) | V2 (Transfer Learning) | V3 (Refinement) | **V4 (Current)** |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Custom U-Net | ResNet34-UNet | ResNet34-UNet + Attention | **U-Net++ w/ EfficientNet-B4** |
| **Input Patching** | 256×256 | 256×256 | 1024×1024 Sliding | **512×512 (50% Overlap)** |
| **Loss Function** | BCE | Dice | Dice | **Combo (Dice+Focal) → Lovász** |
| **Augmentation** | Basic flips | Standard | Standard | **GridDistortion + ElasticTransform** |
| **Inference** | Direct patch | Direct patch | 4-Way TTA | **Sliding Window + 4-Way TTA** |
| **Post-Processing** | Threshold only | Threshold only | Basic morphology | **Graph-theoretic (sknw + NetworkX)** |
| **Output** | PNG mask | PNG mask | PNG + basic GeoJSON | **Skeleton + GeoJSON FeatureCollection** |
| **IoU Score** | ~55% | ~68% | ~75% | **~80%** |

### Visual Progression

| V1 (Noisy) | V3 (Improved) | **V4 (Clean & Connected)** |
| :---: | :---: | :---: |
| ![V1](predicted/embed/v1_pred_mask.png) | ![V3](predicted/embed/v3_pred_mask.png) | ![V4](predicted/embed/v4_pred_mask_1.png) |

---

## 🧠 Technical Deep Dive

### 1. Architecture: U-Net++ & EfficientNet-B4 (`model.py`)

V4 uses a **U-Net++** decoder with an **EfficientNet-B4** encoder via `segmentation_models_pytorch`.

- **EfficientNet-B4**: Compound-scaled backbone (depth × width × resolution) pretrained on ImageNet. Far richer feature representations than ResNet34 with comparable parameter efficiency (~19M params).
- **U-Net++**: Replaces standard skip connections with *dense, nested* skip pathways between every encoder and decoder level. Reduces the semantic gap, preserving sub-pixel spatial detail critical for thin road extraction.

### 2. Training Strategy (`train.py` + `train_lovasz.py`)

**Phase 1 — Main Training (50 epochs):**
- **ComboLoss**: `0.5 × DiceLoss + 0.5 × FocalLoss`
  - Dice optimizes overlap; Focal focuses the model on hard pixels (road edges, shadows, intersections).
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4) with Cosine Annealing scheduler.
- **Gradient Accumulation**: Effective batch size of 16 (8 × 2 steps) to fit within GPU VRAM.

**Phase 2 — Hard Negative Mining (optional, `--hard-mining` flag, +10 epochs):**
- Computes per-image IoU on the full training set.
- Isolates the bottom 20% hardest samples (lowest IoU).
- Fine-tunes exclusively on hard samples at LR=1e-5.

**Phase 3 — Lovász Fine-Tuning (`train_lovasz.py`, 20 epochs):**
- Switches loss to `LovaszLoss` — directly optimizes the Jaccard index via convex surrogation.
- Very conservative LR (1e-5) applied to the best ComboLoss checkpoint.

### 3. Advanced Post-Processing Pipeline (`postprocess.py`)

Raw segmentation masks are noisy. V4 applies a full **graph-theoretic refining pipeline**:

1. **Threshold & Hole Fill** — `prob > 0.45`; fill holes ≤ 400px to prevent false skeleton loops inside wide roads.
2. **Noise Removal** — remove speckle objects < 100px; morphological `closing(disk=3)` to smooth boundaries.
3. **EDT Skeletonize** — convert binary mask to a 1-pixel-wide centerline skeleton.
4. **sknw Graph** — convert skeleton to a NetworkX MultiGraph (junctions = nodes, road segments = weighted edges with pixel-coordinate paths).
5. **Prune & Clean** — iteratively remove short spurs (< 20px), collapse false self-loops (< 100px perimeter), remove redundant parallel edges, clean isolated nodes.
6. **Vectorize** — apply affine transform (pixel coords → projected coords), simplify geometry, export as EPSG:4326 GeoJSON.

> Why a graph instead of morphological gap-closing? A graph lets us reason about *road connectivity* precisely — connecting true endpoints without bloating blobs or destroying topology.

### 4. Inference: Sliding Window + 4-Way TTA

All four inference scripts share the same strategy:

- **Sliding Window**: 512×512 patches, 256-stride (50% overlap), reflect-padded to cover edges.
- **4-Way TTA**: Each patch is predicted in 4 orientations (original, H-flip, V-flip, 90° rotate) → averaged → accumulated into full-image probability map.
- **Normalization**: ImageNet stats (mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`).

---

## 🛠️ Usage Guide

### 1. Local Setup (No Docker)

```bash
git clone https://github.com/lokrim/sanchari-model.git
cd sanchari-model
pip install -r requirements.txt
```

### 2. Docker Setup (Recommended)

Docker packages everything — CUDA, Python, all dependencies — into one image (~6 GB). No local Python setup needed.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with **WSL 2** backend (Windows) or Docker Engine (Mac/Linux)
- For GPU: NVIDIA drivers installed + Docker Desktop WSL 2 integration enabled (Windows) or `nvidia-container-toolkit` (Linux)

#### Step 1 — Configure credential paths

Edit `docker-compose.yml` volume mounts for your OS:

**Windows** (default — replace `AHAMED` with your Windows username):
```yaml
- C:/Users/AHAMED/.kaggle:/root/.kaggle:ro
- C:/Users/AHAMED/.config/earthengine:/root/.config/earthengine
```

**Mac/Linux** (uncomment these, comment out the Windows lines):
```yaml
- ~/.kaggle:/root/.kaggle:ro
- ~/.config/earthengine:/root/.config/earthengine
```

#### Step 2 — Kaggle credentials (one-time)

Download `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token", then place it:

| OS | Path |
|:---|:---|
| Windows | `C:\Users\<USERNAME>\.kaggle\kaggle.json` |
| Mac/Linux | `~/.kaggle/kaggle.json` |

#### Step 3 — Build the image

```bash
docker compose build sanchari-gee-api
```

This builds once and tags the image as `sanchari-model:v4`. All three services share the same image (~6 GB).

#### Step 4 — Authenticate Google Earth Engine (one-time)

```
docker run --rm -it -v C:/Users/AHAMED/.config/earthengine:/root/.config/earthengine sanchari-model:v4 earthengine authenticate --auth_mode=notebook
```

> **Mac/Linux:** Replace the `-v` path with `~/.config/earthengine:/root/.config/earthengine`

This opens a browser-based OAuth flow:
1. Click the URL printed in the terminal
2. Sign in with your Google account
3. Copy the verification code back into the terminal

Credentials are saved to your host machine and automatically mounted into all containers.

#### Step 5 — Verify everything works

```bash
# Verify GEE credentials
docker compose run --rm cli python check_gee.py

# Verify GPU is detected
docker compose run --rm cli python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Running APIs

```bash
# GEE API → port 8001 (fetches satellite imagery live from Google Earth Engine)
docker compose up sanchari-gee-api

# Local GeoTIFF API → port 8000 (uses .tif files in ./geotiffs/)
docker compose up sanchari-local-api

# Both APIs simultaneously
docker compose up sanchari-gee-api sanchari-local-api
```

#### Running Scripts

```bash
# Download dataset and preprocess into 512×512 tiles
docker compose run --rm cli python src/preprocess.py --download

# Training (Phase 1 — ComboLoss, 50 epochs)
docker compose run --rm cli python src/train.py

# Training with hard negative mining
docker compose run --rm cli python src/train.py --hard-mining

# Lovász fine-tuning (run after main training)
docker compose run --rm cli python src/train_lovasz.py

# Batch inference via GEE (10 random US city coordinates)
docker compose run --rm cli python src/predict_gee.py

# Batch inference on local test images
docker compose run --rm cli python src/predict.py --input test-images --output predicted/predicted

# Optimize binarisation threshold
docker compose run --rm cli python src/optimize_threshold.py

# Run unit tests
docker compose run --rm cli python -m pytest src/test_scripts.py -v

# Check coordinate → GeoTIFF coverage
docker compose run --rm cli python check_coords.py 30.2672 -97.7431

# Interactive shell inside the container
docker compose run --rm cli bash
```

#### Docker Disk Usage & Cleanup

The image is ~6 GB. To reclaim space:
```bash
docker builder prune -a -f    # Clear build cache
docker system prune -a         # Remove all unused images + containers
docker system df               # Check current disk usage
```

### 2. Preprocessing

Download the DeepGlobe dataset from Kaggle and tile into 512×512 patches with 50% overlap:

```bash
python src/preprocess.py --download
```

### 3. Training

```bash
# Phase 1 — Main Training
python src/train.py

# Phase 1 + Hard Negative Mining
python src/train.py --hard-mining

# Phase 3 — Lovász Fine-Tuning (run after main training)
python src/train_lovasz.py
```

> Requires a GPU (RTX 3060+ recommended; tested on RTX 4090).

### 4. Inference — Two Modes

#### A. Google Earth Engine (GEE) 🌍

Fetches NAIP or Sentinel-2 imagery live — no local files needed.

```bash
# Batch inference (10 random US city coordinates)
python src/predict_gee.py

# Real-time API server
python src/main_gee.py --debug
```

**Endpoint:** `POST /predict` on port **8001**
```json
{ "latitude": 30.2672, "longitude": -97.7431 }
```

Optional fields: `"collection"` (default: `USDA/NAIP/DOQQ`), `"zoom"` (default: `1.0` m/px scale).

Outputs: images to `predicted/predicted/`, GeoJSON to `predicted/output-geojson/`.

#### B. Local GeoTIFFs 🗺️

Place `.tif` files in `./geotiffs/`. The API auto-detects the file covering the given coordinate.

```bash
# Batch inference on test images
python src/predict.py --input test-images --output predicted

# Local API server
python src/main.py --debug
```

**Endpoint:** `POST /predict` on port **8000**
```json
{ "latitude": 30.2672, "longitude": -97.7431 }
```

Process: Scan GeoTIFFs → find matching file → crop 1024×1024 window → sliding window inference → return GeoJSON.

---

## 🌐 API Output — GeoJSON

Both APIs return a **GeoJSON FeatureCollection** of polyline road segments (EPSG:4326), compatible with QGIS, Mapbox, and Leaflet.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "MultiLineString",
        "coordinates": [
          [[-97.743, 30.267], [-97.744, 30.268]]
        ]
      },
      "properties": { "name": "road_network" }
    }
  ]
}
```

---

## 📂 Project Structure

```
sanchari-model/
├── src/model.py              # U-Net++ + EfficientNet-B4 architecture
├── src/preprocess.py         # Dataset download + 512×512 tiling
├── src/dataset.py            # PyTorch Dataset + Albumentations augmentation
├── src/train.py              # Main training loop (ComboLoss + Hard Mining)
├── src/train_lovasz.py       # Lovász fine-tuning
├── src/postprocess.py        # Graph-theoretic post-processing (shared)
├── src/predict.py            # Local batch inference
├── src/predict_gee.py        # GEE batch inference
├── src/main.py               # Local GeoTIFF FastAPI server (port 8000)
├── src/main_gee.py           # GEE FastAPI server (port 8001)
├── src/optimize_threshold.py # Threshold optimization on validation set
├── check_api.py              # Debug: verify api works with requests
├── check_coords.py           # Debug: verify coordinate → GeoTIFF mapping
├── check_gee.py              # Debug: verify GEE connectivity
├── src/test_scripts.py       # Automated pipeline validation tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── geotiffs/                # Local GeoTIFF files (user-supplied)
├── weights/                 # Trained model weights
│   └── best_model_v4.pth
├── predicted/               # Inference outputs
└── old-versions/            # V1, V2, V3 archived scripts
    ├── V1/
    ├── V2/
    └── V3/
```

---

## 🔑 Credentials Reference

| Credential | Location (Windows) | Location (Mac/Linux) | Purpose |
|:---|:---|:---|:---|
| Kaggle API key | `C:\Users\<USERNAME>\.kaggle\kaggle.json` | `~/.kaggle/kaggle.json` | Dataset download |
| GEE OAuth token | `C:\Users\<USERNAME>\.config\earthengine\credentials` | `~/.config/earthengine/credentials` | Satellite imagery access |

**Re-authenticate GEE (if token expires):**
```bash
docker run --rm -it \
  -v C:/Users/AHAMED/.config/earthengine:/root/.config/earthengine \
  sanchari-model:v4 \
  earthengine authenticate --auth_mode=notebook
```

> **Mac/Linux:** Replace the `-v` path with `~/.config/earthengine:/root/.config/earthengine`

---

## 📦 Key Dependencies

| Library | Purpose |
|:---|:---|
| `torch`, `segmentation_models_pytorch`, `timm` | Model architecture & training |
| `albumentations` | Image augmentation |
| `rasterio`, `pyproj` | GeoTIFF I/O & CRS projection |
| `earthengine-api` | Google Earth Engine access |
| `sknw`, `networkx` | Skeleton → graph construction & pruning |
| `shapely`, `geopandas` | Geometry & GeoJSON export |
| `fastapi`, `uvicorn` | REST API servers |
| `scikit-image` | Skeletonization & morphological ops |

---

## 🧪 Example API Requests

**Local GeoTIFF API** (`main.py` — port 8000):
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"latitude\":30.224949915094008,\"longitude\":-97.78460932372762}"
```

**GEE API** (`main_gee.py` — port 8001):
```bash
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d "{\"latitude\":34.09452,\"longitude\":-118.27286}"
```

Both return a GeoJSON FeatureCollection of the road network.

---

**License:** MIT
