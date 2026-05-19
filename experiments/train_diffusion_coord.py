"""
Train the 1D trajectory coordinate diffusion model (TrajectoryUNet1D).

Reuses the same A*-on-GridWorld data pipeline as make_dataset.py but
stores paths as fixed-length waypoint sequences [T, 2] instead of 2D
occupancy grids.

Usage (from project root):
  # Generate + train (caches dataset to data/trajectory_dataset.pt)
  python -m experiments.train_diffusion_coord

  # Override training parameters
  python -m experiments.train_diffusion_coord --epochs 100 --T 64 --num-worlds 200

Output: data/diffusion_coord_model.pt  (loadable by the Coord-Diff button)
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from environments.grid_world import GridWorld
from algorithms.astar import astar
from algorithms.diffusion_coord import TrajectoryUNet1D, TrajectoryDDPM


# ---------------------------------------------------------------------------
# Dataset: path coordinates as [T, 2] waypoint sequences
# ---------------------------------------------------------------------------

def interpolate_path(path: np.ndarray, T: int) -> np.ndarray:
    """Resample a variable-length path to exactly T evenly-spaced waypoints."""
    n = len(path)
    if n <= 1:
        return np.tile(path[0] if n == 1 else [0.0, 0.0], (T, 1))
    lengths = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total == 0:
        return np.tile(path[0], (T, 1))
    target = np.linspace(0, total, T)
    result = []
    for d in target:
        idx = np.searchsorted(cumulative, d, side="right") - 1
        idx = max(0, min(idx, n - 2))
        t = (d - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx] + 1e-8)
        result.append(path[idx] + t * (path[idx + 1] - path[idx]))
    return np.array(result, dtype=np.float32)


class TrajectoryDataset(Dataset):
    """Each sample: (trajectory [T, 2], elevation_map [1, H, W], start [2], goal [2])."""

    def __init__(self, trajectories: list[np.ndarray],
                 elevation_grids: list[np.ndarray],
                 starts: list[np.ndarray], goals: list[np.ndarray]):
        self.trajectories = trajectories
        self.elevations = elevation_grids
        self.starts = starts
        self.goals = goals

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        traj = torch.from_numpy(self.trajectories[idx])         # [T, 2]
        elev = torch.from_numpy(self.elevations[idx]).unsqueeze(0)  # [1, H, W]
        return traj, elev


def generate_trajectory_dataset(
    num_worlds: int = 200,
    samples_per_world: int = 5,
    world_size: int = 32,
    obstacles_per_world: int = 8,
    T: int = 64,
    seed: int = 42,
) -> TrajectoryDataset:
    """Generate random GridWorlds, run A*, and store [T, 2] waypoint paths."""
    rng = np.random.RandomState(seed)
    trajectories, elevations, starts, goals = [], [], [], []

    for w in range(num_worlds):
        world = GridWorld(world_size)
        for _ in range(obstacles_per_world):
            x = int(rng.randint(0, world_size))
            y = int(rng.randint(0, world_size))
            world.add_mountain(x, y, height=float(rng.uniform(3.0, 15.0)),
                               radius=int(rng.randint(5, 12)))

        for _ in range(samples_per_world):
            sx = int(rng.randint(0, world_size))
            sy = int(rng.randint(0, world_size))
            gx = int(rng.randint(0, world_size))
            gy = int(rng.randint(0, world_size))
            if (sx, sy) == (gx, gy):
                continue

            _, path_coords = astar((sx, sy), (gx, gy), world)
            if not path_coords:
                continue

            path_arr = np.array(path_coords, dtype=np.float32)

            # normalise coordinates to [-1, 1] for the model
            def norm_coords(px, py):
                return np.array([-1.0 + 2.0 * px / (world_size - 1),
                                 -1.0 + 2.0 * py / (world_size - 1)])

            path_norm = np.array([norm_coords(p[0], p[1]) for p in path_arr])
            traj = interpolate_path(path_norm, T)               # [T, 2]

            elev = world.grid.astype(np.float32)
            # normalise elevation
            if elev.max() > 0:
                elev = elev / elev.max()

            trajectories.append(traj)
            elevations.append(elev)
            starts.append(norm_coords(sx, sy))
            goals.append(norm_coords(gx, gy))

        if (w + 1) % 50 == 0:
            print(f"  generated {w + 1}/{num_worlds} worlds ...")

    return TrajectoryDataset(trajectories, elevations, starts, goals)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=200)
    parser.add_argument("--samples-per-world", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--obstacles-per-world", type=int, default=8)
    parser.add_argument("--T", type=int, default=64, help="fixed path length (waypoints)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--data-path", type=str, default="data/trajectory_dataset.pt")
    parser.add_argument("--checkpoint", type=str, default="data/diffusion_coord_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- dataset ---
    if os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        dataset = torch.load(args.data_path, weights_only=False)
    else:
        print(f"Generating {args.num_worlds} worlds × {args.samples_per_world} pairs ...")
        dataset = generate_trajectory_dataset(
            num_worlds=args.num_worlds,
            samples_per_world=args.samples_per_world,
            world_size=args.world_size,
            obstacles_per_world=args.obstacles_per_world,
            T=args.T,
        )
        os.makedirs("data", exist_ok=True)
        torch.save(dataset, args.data_path)
        print(f"Saved {len(dataset)} samples to {args.data_path}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- model ---
    model = TrajectoryUNet1D(time_dim=128, T=args.T).to(device)
    ddpm = TrajectoryDDPM(model, num_timesteps=args.num_timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Training for {args.epochs} epochs ({len(dataset)} samples, {len(loader)} batches/epoch) ...")

    for epoch in range(args.epochs):
        total_loss = 0.0
        model.train()
        for batch in loader:
            loss = ddpm.train_step(batch, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  |  Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), args.checkpoint)

    torch.save(model.state_dict(), args.checkpoint)
    print(f"\nFinal model saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
