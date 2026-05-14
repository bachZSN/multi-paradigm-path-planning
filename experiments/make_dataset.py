"""
Generate path planning datasets using A* on random GridWorlds.

Each sample contains:
  - elevation [1, H, W] : terrain height map (normalized 0-1)
  - start_map [1, H, W] : gaussian heatmap at the start position
  - goal_map  [1, H, W] : gaussian heatmap at the goal position
  - path_map  [1, H, W] : binary occupancy grid of the optimal A* path

Usage:
  python -m experiments.make_dataset                    # generate & save
  python -m experiments.make_dataset --num-samples 1000 # override count

Output is saved to data/dataset.pt as a torch.utils.data.Dataset.
"""

from __future__ import annotations
import os
import argparse
import math
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from environments.grid_world import GridWorld
from algorithms.astar import astar


class PathfindingDataset(Dataset):
    """
    Each item returns (elevation, start_map, goal_map, path_map),
    all as [1, H, W] float32 tensors.

    Start and goal are encoded as gaussian heatmaps so the model can
    learn to route between arbitrary positions.
    """

    def __init__(self, elevation_grids: Sequence[np.ndarray],
                 start_positions: Sequence[tuple[int, int]],
                 goal_positions: Sequence[tuple[int, int]],
                 path_grids: Sequence[np.ndarray]):
        self.elevation = [g.astype(np.float32) for g in elevation_grids]
        self.starts = start_positions
        self.goals = goal_positions
        self.paths = [p.astype(np.float32) for p in path_grids]
        self.H, self.W = self.elevation[0].shape

    def __len__(self) -> int:
        return len(self.elevation)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        elev = torch.from_numpy(self.elevation[idx]).unsqueeze(0)
        path = torch.from_numpy(self.paths[idx]).unsqueeze(0)

        # normalize elevation to [0,1] per map
        if elev.max() > 0:
            elev = elev / elev.max()

        start_map = self._heatmap(self.starts[idx])
        goal_map = self._heatmap(self.goals[idx])

        # data augmentation: D4 rotations + flips
        if self._rng.random() > 0.5:
            k = self._rng.randint(0, 4)
            elev = torch.rot90(elev, k, dims=[-2, -1])
            path = torch.rot90(path, k, dims=[-2, -1])
            start_map = torch.rot90(start_map, k, dims=[-2, -1])
            goal_map = torch.rot90(goal_map, k, dims=[-2, -1])

        if self._rng.random() > 0.5:
            elev = torch.flip(elev, dims=[-1])
            path = torch.flip(path, dims=[-1])
            start_map = torch.flip(start_map, dims=[-1])
            goal_map = torch.flip(goal_map, dims=[-1])

        return elev, start_map, goal_map, path

    _rng = np.random.RandomState(0)

    def _heatmap(self, pos: tuple[int, int], sigma: float = 1.5) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij",
        )
        dist2 = (xs - pos[0]) ** 2 + (ys - pos[1]) ** 2
        return torch.exp(-dist2 / (2 * sigma * sigma)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_samples(
    num_worlds: int = 200,
    samples_per_world: int = 5,
    world_size: int = 32,
    obstacles_per_world: int = 8,
    seed: int = 42,
) -> PathfindingDataset:
    """
    Generate a dataset by sampling random GridWorlds and running A*
    between multiple random start/goal pairs per world.
    """
    rng = np.random.RandomState(seed)
    elevations, starts, goals, paths = [], [], [], []

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

            path_grid = np.zeros((world_size, world_size), dtype=np.float32)
            for (px, py) in path_coords:
                if 0 <= px < world_size and 0 <= py < world_size:
                    path_grid[py, px] = 1.0

            elevations.append(world.grid.astype(np.float32))
            starts.append((sx, sy))
            goals.append((gx, gy))
            paths.append(path_grid)

        if (w + 1) % 50 == 0:
            print(f"  generated {w+1}/{num_worlds} worlds ...")

    return PathfindingDataset(elevations, starts, goals, paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=200)
    parser.add_argument("--samples-per-world", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--obstacles-per-world", type=int, default=8)
    parser.add_argument("--output", type=str, default="data/dataset.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.num_worlds} worlds × {args.samples_per_world} pairs ...")
    dataset = generate_samples(
        num_worlds=args.num_worlds,
        samples_per_world=args.samples_per_world,
        world_size=args.world_size,
        obstacles_per_world=args.obstacles_per_world,
        seed=args.seed,
    )

    os.makedirs("data", exist_ok=True)
    torch.save(dataset, args.output)
    print(f"Saved {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()
