import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


# ---------------------------------------------------------------------------
# Dataset: run A* on GridWorld to create (elevation, start, goal, path) pairs
# ---------------------------------------------------------------------------

class PathfindingDataset(torch.utils.data.Dataset):
    """
    Stores pathfinding problems as grid-based channels:
      - elevation [1, H, W] : terrain height map (normalized)
      - start_map [1, H, W] : one-hot-ish gaussian blob at start position
      - goal_map  [1, H, W] : one-hot-ish gaussian blob at goal position
      - path_map  [1, H, W] : binary occupancy of the optimal A* path
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

    def __len__(self):
        return len(self.elevation)

    def __getitem__(self, idx):
        elev = torch.from_numpy(self.elevation[idx]).unsqueeze(0)
        path = torch.from_numpy(self.paths[idx]).unsqueeze(0)

        # normalize elevation to [0, 1] per map
        if elev.max() > 0:
            elev = elev / elev.max()

        # build start/goal heatmaps (gaussian blob at the position)
        start_map = self._position_heatmap(self.starts[idx])
        goal_map = self._position_heatmap(self.goals[idx])

        elev, path, start_map, goal_map = self.augment(elev, path, start_map, goal_map)
        return elev, start_map, goal_map, path

    def _position_heatmap(self, pos: tuple[int, int], sigma: float = 1.5) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij",
        )
        dist2 = (xs - pos[0]) ** 2 + (ys - pos[1]) ** 2
        heat = torch.exp(-dist2 / (2 * sigma ** 2))
        return heat.unsqueeze(0)

    def augment(self, elev, path, start_map, goal_map):
        k = np.random.randint(0, 4)
        elev = torch.rot90(elev, k, dims=[-2, -1])
        path = torch.rot90(path, k, dims=[-2, -1])
        start_map = torch.rot90(start_map, k, dims=[-2, -1])
        goal_map = torch.rot90(goal_map, k, dims=[-2, -1])

        if np.random.rand() > 0.5:
            elev = torch.flip(elev, dims=[-1])
            path = torch.flip(path, dims=[-1])
            start_map = torch.flip(start_map, dims=[-1])
            goal_map = torch.flip(goal_map, dims=[-1])

        return elev, path, start_map, goal_map


# ---------------------------------------------------------------------------
# Dataset generation using A*
# ---------------------------------------------------------------------------

def generate_dataset(
    world_size: int = 32,
    num_worlds: int = 500,
    num_obstacles: int = 8,
    seed: int = 42,
) -> PathfindingDataset:
    from algorithms.astar import astar
    from environments.grid_world import GridWorld

    rng = np.random.RandomState(seed)
    elevations, starts, goals, paths = [], [], [], []

    for _ in range(num_worlds):
        world = GridWorld(world_size)
        for _ in range(num_obstacles):
            x = int(rng.randint(0, world_size))
            y = int(rng.randint(0, world_size))
            world.add_mountain(x, y, height=float(rng.uniform(3.0, 15.0)),
                               radius=int(rng.randint(5, 12)))

        sx, sy = int(rng.randint(0, world_size)), int(rng.randint(0, world_size))
        gx, gy = int(rng.randint(0, world_size)), int(rng.randint(0, world_size))
        if (sx, sy) == (gx, gy):
            continue

        came_from, path_coords = astar((sx, sy), (gx, gy), world)
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

    return PathfindingDataset(elevations, starts, goals, paths)


# ---------------------------------------------------------------------------
# Diffusion components
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaGNBlock(nn.Module):
    """
    Adaptive GroupNorm residual block with FiLM-style time conditioning.
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = torch.chunk(self.time_mlp(F.silu(t_emb))[:, :, None, None], 2, dim=1)
        h = h * (1 + gamma) + beta
        return self.conv2(F.silu(self.norm2(h)))


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(32, channels) if channels >= 32 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x).view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        return x + attn_out.transpose(1, 2).view(b, c, h, w)


class PathUNet(nn.Module):
    """
    Conditional UNet that denoises a path grid given:
      - x          [B, 1, H, W]  : noisy path occupancy
      - t          [B]           : diffusion timestep
      - elevation  [B, 1, H, W]  : terrain height
      - start_map  [B, 1, H, W]  : start-position heatmap
      - goal_map   [B, 1, H, W]  : goal-position heatmap

    Total input channels = 4 (noisy_path + elevation + start + goal).
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = AdaGNBlock(64, 128, time_dim)
        self.down2 = AdaGNBlock(128, 256, time_dim)

        self.mid_block = AdaGNBlock(256, 256, time_dim)
        self.mid_attn = AttentionBlock(256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.rev1 = AdaGNBlock(256, 128, time_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.rev2 = AdaGNBlock(128, 64, time_dim)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                elevation: torch.Tensor, start_map: torch.Tensor,
                goal_map: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x, elevation, start_map, goal_map], dim=1)

        h0 = self.init_conv(x_in)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(F.max_pool2d(h1, 2), t_emb)

        h_mid = self.mid_block(F.max_pool2d(h2, 2), t_emb)
        h_mid = self.mid_attn(h_mid)

        u1 = self.up1(h_mid)
        u1 = self.rev1(torch.cat([u1, h2], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.rev2(torch.cat([u2, h1], dim=1), t_emb)

        return self.final_conv(u2)


# ---------------------------------------------------------------------------
# DDPM training helpers
# ---------------------------------------------------------------------------

class DDPM:
    """
    Denoising Diffusion Probabilistic Model.
    Wraps the UNet with noise scheduling, forward diffusion (q_sample),
    and loss computation.
    """
    def __init__(self, model: nn.Module, num_timesteps: int = 200, device: str = "cpu"):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_om * noise, noise

    def train_step(self, batch, optimizer: torch.optim.Optimizer) -> float:
        elevation, start_map, goal_map, path = [b.to(self.device) for b in batch]
        B = path.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        xt, noise = self.q_sample(path, t)
        pred_noise = self.model(xt, t, elevation, start_map, goal_map)
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


# ---------------------------------------------------------------------------
# DDIM fast sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model: nn.Module,
                elevation: torch.Tensor,
                start_map: torch.Tensor,
                goal_map: torch.Tensor,
                num_train_steps: int = 200,
                num_sample_steps: int = 50,
                eta: float = 0.0) -> torch.Tensor:
    """
    DDIM sampling: generate a path from noise.
    Goes from t = num_sample_steps-1 down to 0.
    """
    device = next(model.parameters()).device
    B = elevation.shape[0]
    H, W = elevation.shape[-2:]

    # use the same noise schedule as training
    betas = torch.linspace(1e-4, 0.02, num_train_steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # subsample steps for faster inference
    step_indices = torch.linspace(0, num_train_steps - 1, num_sample_steps, dtype=torch.long, device=device)

    # start from pure noise
    x_t = torch.randn((B, 1, H, W), device=device)

    for i in range(num_sample_steps - 1, -1, -1):
        t_idx = step_indices[i]
        t = t_idx.unsqueeze(0).expand(B)

        alpha_cur = alpha_bar[t_idx]
        alpha_prev = alpha_bar[step_indices[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

        pred_noise = model(x_t, t, elevation, start_map, goal_map)

        # DDIM update: x_{t-1} = sqrt(alpha_prev) * x0_pred + sqrt(1 - alpha_prev - sigma^2) * pred_noise + sigma * noise
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_cur) * pred_noise) / torch.sqrt(alpha_cur)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_cur) * (1.0 - alpha_cur / alpha_prev))
        noise = torch.randn_like(x_t) if i > 0 else 0.0

        dir_xt = torch.sqrt(1.0 - alpha_prev - sigma ** 2) * pred_noise if sigma > 0 else torch.sqrt(1.0 - alpha_prev) * pred_noise
        x_t = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise

    return x_t  # B, 1, H, W — predicted path occupancy
