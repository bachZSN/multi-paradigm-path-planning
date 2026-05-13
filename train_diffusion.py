"""
Sample code: Train a diffusion model to generate paths using A* datasets.

This is a self-contained proof-of-concept. It adapts the existing GridWorld/A*
code from this project into a training pipeline without modifying any original files.

Architecture overview:
  1. Dataset generation: run A* on random worlds, save (world, start, goal, path) pairs.
  2. Path representation: a path is encoded as a fixed-length sequence of (x, y) waypoints
     interpolated from the A* output, plus a mask indicating valid waypoints.
     We also encode the world as a low-resolution cost map and start/goal as one-hot positions.
  3. Diffusion model (DDPM): a conditional denoising model that takes a noisy path
     + world/start/goal conditioning and predicts the noise. At inference, it
     denoises random noise into a path.

Usage:
  python train_diffusion.py              # generate data, train, evaluate
  python train_diffusion.py --load-only  # skip generation, load existing dataset
"""

from __future__ import annotations

import os
import sys
import math
import dataclasses
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Boilerplate to import from this project without modifying it ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments.grid_world import GridWorld, GridLocation
from algorithms.astar import astar, calculate_total_cost
# --------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Dataset generation: run A* on random worlds
# ---------------------------------------------------------------------------

def generate_dataset(num_worlds: int = 500, world_size: int = 32, num_obstacles: int = 8,
                     seed: int = 42) -> list[dict]:
    rng = np.random.RandomState(seed)
    data = []

    for _ in range(num_worlds):
        world = GridWorld(world_size)

        # add random obstacles to make the problem interesting
        for _ in range(num_obstacles):
            x = int(rng.randint(0, world_size))
            y = int(rng.randint(0, world_size))
            world.add_mountain(x, y, height=float(rng.uniform(3.0, 15.0)),
                               radius=int(rng.randint(5, 12)))

        start: GridLocation = (int(rng.randint(0, world_size)),
                               int(rng.randint(0, world_size)))
        goal: GridLocation = (int(rng.randint(0, world_size)),
                              int(rng.randint(0, world_size)))

        if start == goal:
            continue

        came_from, path = astar(start, goal, world)
        if not path:
            continue

        cost = calculate_total_cost(path, world)

        data.append({
            "world": world.grid.copy().astype(np.float32),
            "start": np.array(start, dtype=np.float32),
            "goal": np.array(goal, dtype=np.float32),
            "path": np.array(path, dtype=np.float32),
            "cost": cost,
        })

    return data


# ---------------------------------------------------------------------------
# 2. Path representation: fixed-length waypoint sequences
# ---------------------------------------------------------------------------

def interpolate_path(path: np.ndarray, num_waypoints: int) -> np.ndarray:
    """Resample a variable-length path to a fixed number of waypoints."""
    n = len(path)
    if n == 0:
        return np.zeros((num_waypoints, 2))
    if n == 1:
        return np.tile(path[0], (num_waypoints, 1))

    lengths = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total == 0:
        return np.tile(path[0], (num_waypoints, 1))

    target_dists = np.linspace(0, total, num_waypoints)
    return np.array([_interp_at(cumulative, path, d) for d in target_dists])


def _interp_at(cumulative: np.ndarray, points: np.ndarray, dist: float) -> np.ndarray:
    idx = np.searchsorted(cumulative, dist, side="right") - 1
    idx = max(0, min(idx, len(points) - 2))
    t = (dist - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx] + 1e-8)
    return points[idx] + t * (points[idx + 1] - points[idx])


def encode_world(world_grid: np.ndarray, downsample: int = 4) -> np.ndarray:
    """Downsample the world grid to a smaller representation for the model."""
    H, W = world_grid.shape
    if downsample > 1:
        Hd = H // downsample
        Wd = W // downsample
        world_grid = world_grid[:Hd * downsample, :Wd * downsample]
        world_grid = world_grid.reshape(Hd, downsample, Wd, downsample).mean(axis=(1, 3))
    return world_grid


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------------------------------------

class PathPlanningDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict], num_waypoints: int = 32, downsample: int = 4):
        self.num_waypoints = num_waypoints
        self.downsample = downsample
        self.samples = []

        for d in data:
            path = interpolate_path(d["path"], num_waypoints)
            world = encode_world(d["world"], downsample)

            self.samples.append({
                "world": torch.from_numpy(world).unsqueeze(0),
                "start": torch.from_numpy(d["start"]),
                "goal": torch.from_numpy(d["goal"]),
                "path": torch.from_numpy(path),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_paths(batch: list[dict]) -> dict[str, torch.Tensor]:
    world = torch.stack([b["world"] for b in batch])
    start = torch.stack([b["start"] for b in batch])
    goal = torch.stack([b["goal"] for b in batch])
    path = torch.stack([b["path"] for b in batch])
    return {"world": world, "start": start, "goal": goal, "path": path}


# ---------------------------------------------------------------------------
# 4. Conditional Diffusion Model (DDPM)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class PathDenoiser(nn.Module):
    """
    Predicts the noise added to a path, conditioned on the world, start, goal,
    and diffusion timestep.

    The path is flattened: (N, num_waypoints * 2).
    Conditioning is embedded and fused via feature-wise modulation (FiLM).
    """

    def __init__(self, num_waypoints: int = 32, world_h: int = 8, world_w: int = 8,
                 time_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.path_dim = num_waypoints * 2

        # world encoder: small CNN -> feature vector
        self.world_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # start/goal conditioning (2 * 2 = 4 values -> embed)
        self.pos_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM generators
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        # path denoiser MLP
        self.input_proj = nn.Linear(self.path_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(4)
        ])
        self.output_proj = nn.Linear(hidden_dim, self.path_dim)

    def forward(self, path_noisy: torch.Tensor, timestep: torch.Tensor,
                world: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        B = path_noisy.shape[0]

        t_emb = sinusoidal_embedding(timestep, 128)
        t_feat = self.time_embed(t_emb)

        w_feat = self.world_encoder(world)

        pos_feat = self.pos_embed(torch.cat([start, goal], dim=-1))

        cond = t_feat + w_feat + pos_feat

        gamma = self.film_gamma(cond)
        beta = self.film_beta(cond)

        x = self.input_proj(path_noisy)
        for block in self.blocks:
            x = block(x)
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)

        return self.output_proj(x)


# ---------------------------------------------------------------------------
# 5. DDPM Training
# ---------------------------------------------------------------------------

class DDPM:
    def __init__(self, model: nn.Module, num_timesteps: int = 100, device: str = "cpu"):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return xt, noise

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, cond: dict) -> torch.Tensor:
        with torch.no_grad():
            pred_noise = self.model(xt, t, cond["world"], cond["start"], cond["goal"])
            alpha = self.alphas[t][:, None]
            alpha_bar = self.alphas_cumprod[t][:, None]
            beta = self.betas[t][:, None]
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_bar)
            mu = coef1 * (xt - coef2 * pred_noise)
            if t[0].item() > 0:
                noise = torch.randn_like(xt)
                sigma = torch.sqrt(beta)
                return mu + sigma * noise
            return mu

    def sample(self, cond: dict, num_steps: int | None = None) -> torch.Tensor:
        num_steps = num_steps or self.num_timesteps
        B = cond["world"].shape[0]
        xt = torch.randn(B, self.model.path_dim, device=self.device)
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            xt = self.p_sample(xt, t, cond)
        return xt

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        B = batch["path"].shape[0]
        x0 = batch["path"].reshape(B, -1).to(self.device)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.model(xt, t, batch["world"].to(self.device),
                                batch["start"].to(self.device), batch["goal"].to(self.device))
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


# ---------------------------------------------------------------------------
# 6. Main training loop
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_waypoints = 32
    world_size = 32
    downsample = 4
    world_h = world_w = world_size // downsample

    dataset_path = "data/training_data.npz"

    # generate or load dataset
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        loaded = np.load(dataset_path, allow_pickle=True)
        raw_data = list(loaded["data"])
    else:
        print("Generating dataset with A* ...")
        raw_data = generate_dataset(num_worlds=500, world_size=world_size)
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(dataset_path, data=np.array(raw_data, dtype=object))
        print(f"Saved {len(raw_data)} samples to {dataset_path}")

    dataset = PathPlanningDataset(raw_data, num_waypoints=num_waypoints, downsample=downsample)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                         collate_fn=collate_paths)

    model = PathDenoiser(num_waypoints=num_waypoints, world_h=world_h, world_w=world_w).to(device)
    ddpm = DDPM(model, num_timesteps=100, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 50
    print(f"Training for {num_epochs} epochs on {len(dataset)} samples...")

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            loss = ddpm.train_step(batch, optimizer)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1:3d}/{num_epochs}  |  Loss: {avg_loss:.6f}")

    # --- quick eval: sample a few paths ---
    print("\nSampling paths from the trained model...")
    model.eval()
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        cond = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != "path"}
        pred_flat = ddpm.sample(cond)
        pred_path = pred_flat.cpu().reshape(-1, 2).numpy()
        gt_path = sample["path"].numpy()
        print(f"Sample {i+1}: pred shape={pred_path.shape}, gt shape={gt_path.shape}")

    print("\nDone. Model and dataset are ready for further experimentation.")
    print(f"Dataset location: {dataset_path}")


if __name__ == "__main__":
    main()
