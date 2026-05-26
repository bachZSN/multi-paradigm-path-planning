# Multi-Paradigm Path Planning

A comparative study of classic search and diffusion-guided planning on shared grid-world environments with elevation terrain.

This repo contains:

- Baselines: BFS, Dijkstra, A*
- Grid diffusion: predict a 2D path confidence heatmap, then refine with a terrain-aware shortest-path search
- Coordinate diffusion: predict a continuous waypoint trajectory, then refine to a valid grid path

## Project Structure

```
.
├── main.py                          # Entry point — launch the app
├── app.py                           # App class wiring UI actions to algorithms
├── requirements.txt                 # Dependencies (torch, pygame, numpy, ...)
│
├── environments/
│   ├── grid_world.py                # GridWorld terrain model (float64, grid[y,x])
│   └── __init__.py
│
├── algorithms/
│   ├── astar.py                     # A*, Dijkstra, BFS, cost calculation
│   ├── diffusion.py                 # Grid-based PathUNet + DDPM + DDIM/CFG infer
│   ├── diffusion_coord.py           # 1D trajectory TrajectoryUNet1D + DDPM + in-painting
│   └── __init__.py                  # Lazy exports (keeps torch import out of app startup)
│
├── experiments/
│   ├── make_dataset.py              # Dataset generator (grid-based: elevation + path maps)
│   ├── train_diffusion.py           # Training script for grid-based diffusion model
│   ├── train_diffusion_coord.py     # Training script for 1D trajectory diffusion model
│   └── __init__.py
│
├── visualization/
│   ├── renderer.py                  # Pygame world renderer
│   ├── UIManager.py                 # Button layout, keyboard shortcuts, event handling
│   └── __init__.py
│
└── data/                            # Checkpoints & datasets (gitignored)
    └── .gitkeep
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch.

If you are running the UI, you also need `pygame` (installed via `requirements.txt`).

## Running the App

```bash
python main.py
```

### Controls

| Key | Action      | Description                              |
| --- | ----------- | ---------------------------------------- |
| `1` | A\*         | Classic A\* search                       |
| `2` | Diffusion   | Grid-based diffusion model               |
| `3` | Coord-Diff  | 1D trajectory coordinate diffusion model |
| `r` | Reset       | Generate new random world                |
| `t` | Toggle Path | Show/hide the path overlay               |
| `q` | Quit        | Exit the app                             |

Click any button with the mouse to activate it.

When you run `Diffusion` or `Coord-Diff`, the app prints timing and cost metrics comparing diffusion vs baseline A*.

## Training the Models

### Grid-Based Diffusion (`PathUNet`)

Predicts a 2D path confidence map (heatmap) and then extracts a grid-valid path with a terrain-aware shortest-path search.

```bash
# Generate dataset + train in one step (caches to data/training_dataset.pt)
python -m experiments.train_diffusion --num-worlds 200 --epochs 50

# Train from an already-cached dataset
python -m experiments.train_diffusion --epochs 100

# Custom parameters
python -m experiments.train_diffusion \
    --num-worlds 500 \
    --samples-per-world 5 \
    --batch-size 32 \
    --epochs 80 \
    --lr 1e-3 \
    --num-timesteps 200 \
    --checkpoint data/diffusion_model.pt
```

The model runs at reduced resolution (`--world-size`, default `32×32`). In the app, the world is larger (see `create_default_world()`), so inference downsamples elevation + start/goal heatmaps to model resolution, runs diffusion, then upsamples the predicted confidence map back to full resolution before refinement.

Refinement uses a Dijkstra-style search with a neuro-symbolic cost:

- `step_cost = 1`
- `climb_penalty = max(0, Δelevation) * 20`
- `model_penalty = (1 - sigmoid(confidence)) * 50`

**Loss guide:**

- Epoch 1–10: ~1.0 → ~0.6 (learns map structure)
- Epoch 10–30: ~0.6 → ~0.3 (learns terrain-aware routing)
- Epoch 30–50+: ~0.3 → ~0.15 (refinement)
- Below 0.1: likely overfitting (add more worlds or augment)

### 1D Trajectory Diffusion (`TrajectoryUNet1D`)

Predicts a continuous `[T, 2]` waypoint sequence, then converts it into a grid-valid path with a terrain-aware refinement search.

```bash
# Generate + train (caches to data/trajectory_dataset.pt)
python -m experiments.train_diffusion_coord --num-worlds 200 --epochs 80

# Custom parameters
python -m experiments.train_diffusion_coord \
    --num-worlds 300 \
    --samples-per-world 5 \
    --T 64 \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --num-timesteps 200 \
    --checkpoint data/diffusion_coord_model.pt
```

The model uses 1D temporal convolutions over the trajectory timeline and conditions on:

- elevation map (queried via `F.grid_sample`)
- start and goal coordinates (provided as inputs and also enforced during sampling)

Sampling performs endpoint in-painting: start (index `0`) and goal (index `T-1`) are clamped at every denoising step.

Checkpoint note: changes to the coordinate model architecture require retraining; if you see a size-mismatch when loading `data/diffusion_coord_model.pt`, retrain with `python -m experiments.train_diffusion_coord`.

### Dataset Contents

**Grid-based** (`PathfindingDataset`, generated by `make_dataset.py`):
Each sample returns 4 tensors:

- `elevation` `[1, H, W]` — normalised terrain
- `start_map` `[1, H, W]` — gaussian heatmap at start
- `goal_map` `[1, H, W]` — gaussian heatmap at goal
- `path_map` `[1, H, W]` — binary A\* path occupancy

**Trajectory-based** (`TrajectoryDataset`, generated by `train_diffusion_coord.py`):
Each sample returns:

- `trajectory` `[T, 2]` — interpolated waypoints normalised to `[-1, 1]`
- `elevation` `[1, H, W]` — normalised terrain
- `start` `[2]` — normalised to `[-1, 1]`
- `goal` `[2]` — normalised to `[-1, 1]`

## How the Diffusion Models Work

### Grid-Based Pipeline

1. Build start/goal heatmaps and downsample elevation to model resolution
2. Condition the model on 4 channels: `(noisy_path, elevation, start_map, goal_map)`
3. Run CFG-guided DDIM sampling to produce a 2D confidence map
4. Upsample confidence to full world resolution
5. Run terrain-aware shortest-path search guided by confidence to produce a connected, grid-valid path

### Coordinate-Based Pipeline

1. Normalise start/goal to `[-1, 1]` and initialise a trajectory as noise
2. DDIM-style sampling predicts and removes noise over `T` timesteps
3. Clamp the endpoints (in-painting) at every denoising step
4. De-normalise to world pixel coordinates
5. Refine to a grid-valid path using a terrain-aware search biased toward the predicted trajectory corridor

## Configuration

All constants used by the app (checkpoint paths, model sizes) are in `app.py`. The two model checkpoints are:

- `data/diffusion_model.pt` — grid PathUNet (117 MB with fp32 weights)
- `data/diffusion_coord_model.pt` — 1D TrajectoryUNet1D (smaller, ~5–15 MB)

## Metrics (Printed By The App)

The UI actions print:

- Baseline A*: `A* time`, `A* cost`
- Grid diffusion: `Diffuse sample`, `Refine search`, `Total`, `Diffusion cost`, `Cost delta`
- Coord diffusion: `Diffuse sample`, `Refine search`, `Total`, `Coord cost`, `Cost delta`

The diffusion methods include a refinement stage, so “diffusion time” is reported as separate sampling and refinement components.

## Why “Corridor Dijkstra” Can Be Faster Than A*

The refinement search is guided by a strong corridor term (derived from the model’s confidence or trajectory distance). This can drastically reduce the number of states that are competitive in the priority queue, so the search often expands far fewer nodes than baseline A*.

In contrast, the baseline A* heuristic is Manhattan distance while the true path cost includes elevation, so the heuristic can be weak and A* may expand close to Dijkstra-level nodes.

## Performance Notes

- App startup is kept fast by lazily importing torch-heavy modules only when diffusion buttons are used.
- Terrain generation is vectorized; `create_default_world()` should be fast even for larger maps.

## Citation & Paper

See `paper/` directory for related write-up.
