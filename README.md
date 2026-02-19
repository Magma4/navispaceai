# NavispaceAI

NavispaceAI is an end-to-end indoor navigation platform:
- FastAPI backend for blueprint processing, occupancy generation, and pathfinding
- React + Three.js frontend for interactive 3D navigation and autonomous agent movement
- Multi-floor, room-indexed, meter-based indoor path workflows

## Architecture

```text
Blueprint(s) -> CV preprocessing -> Wall/Room extraction -> Occupancy grid(s)
              -> Multi-floor building model -> 3D A* pathfinding -> FastAPI APIs
              -> React + Three.js scene -> Autonomous agent + camera follow
```

## Production Deployment (Docker)

### 1) Prerequisites

- Docker 24+
- Docker Compose v2

### 2) Environment Setup

Copy env templates and adjust values as needed.

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Notes:
- `API_PORT` controls backend exposed port (default `8000`)
- `REACT_APP_API_URL` / `VITE_API_URL` should point to backend API URL
- `DATABASE_URL` is optional unless persistence is enabled

### 3) Run Full Stack

```bash
docker compose up --build
```

Services:
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000](http://localhost:8000)

Optional Postgres profile:

```bash
docker compose --profile db up --build
```

### 4) Volume Mounts

Configured volumes:
- `./assets -> /app/assets` for sample and uploaded blueprint assets
- `./backend/generated -> /app/backend/generated` for generated grids/models

## Local Development (without Docker)

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Optional ML segmentation stack (for high-accuracy wall/door detection):

```bash
pip install -r backend/ml/requirements-ml.txt
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Set API URL (if needed):

```bash
export VITE_API_URL=http://localhost:8000
export REACT_APP_API_URL=http://localhost:8000
```

## Live 3D Demo Checklist

After services start:

1. Open [http://localhost:3000](http://localhost:3000)
2. Upload sample blueprint from `assets/`
3. Process building/floor data
4. Load 3D model in scene
5. Select start/goal (coordinates or room selector)
6. Compute path and enable autonomous mode
7. Toggle camera follow for chase mode
8. Switch floors to verify multi-floor rendering and path transitions

## Trainable Wall/Door Model

The backend now supports a trainable segmentation model for better detection on complex blueprints.

### Dataset format

```text
<data-root>/
  train/
    images/
      *.png|jpg
    masks/
      walls/
        <same_stem>.png
      doors/
        <same_stem>.png
  val/
    images/
    masks/
      walls/
      doors/
```

Masks should be binary (white = class, black = background).

### Prepare CubiCasa5K raw download

If you downloaded the official CubiCasa bundle (`train.txt`, `val.txt`, `model.svg` files), convert it first:

```bash
python -m backend.ml.prepare_cubicasa \
  --cubicasa-root data/raw/cubicasa5k \
  --output-root data/floorplans
```

This creates `data/floorplans/train` and `data/floorplans/val` in the format required by the trainer.

### Train

```bash
python -m backend.ml.train \
  --data-dir data/floorplans \
  --epochs 40 \
  --batch-size 4 \
  --image-size 768 \
  --output backend/ml/checkpoints/wall_door_unet.pt
```

### Enable inference in API

```bash
export NAVISPACE_ENABLE_ML=true
export NAVISPACE_SEG_MODEL=backend/ml/checkpoints/wall_door_unet.pt
export NAVISPACE_SEG_DEVICE=cpu
export NAVISPACE_SEG_INPUT_SIZE=768
export NAVISPACE_SEG_THRESHOLD=0.45
```

Then restart backend. If model is unavailable, backend automatically falls back to classical CV.

## API Endpoints (Current + Extended)

- `POST /process-blueprint`
- `POST /find-path`
- `GET /floors` (extended multi-floor metadata)
- `GET /rooms` (extended room index metadata)
- `POST /find-path-3d` (extended 3D path in meters)
- `POST /process-building` (extended multi-floor ingestion)

## Testing

### Backend

```bash
pytest -q
```

### Frontend

```bash
cd frontend
node tests/navigation.test.mjs
npm run test --if-present
```

## CI/CD Overview

GitHub Actions workflows:

- `.github/workflows/backend.yml`
  - Runs on push/PR
  - Installs backend deps
  - Runs `black --check`, `flake8`, `pytest`
  - Builds backend Docker image
  - Includes placeholder image push/deploy step

- `.github/workflows/frontend.yml`
  - Runs on push/PR
  - Installs frontend deps
  - Runs lint script (if present)
  - Runs frontend tests
  - Builds frontend static assets and Docker image
  - Includes placeholder static deploy step

Both workflows are production CI-ready with explicit placeholder blocks for:
- registry auth + image push
- staging/preview deployment targets

## Roadmap

- Full persistence layer for building/floor/room metadata (Postgres + migrations)
- Auth and per-tenant building workspaces
- Streaming updates for dynamic obstacles and re-planning
- Enhanced connector inference (stairs/elevators) from CAD annotations
- Kubernetes Helm deployment and autoscaling profile
- Observability stack (OpenTelemetry, Prometheus, Grafana)
