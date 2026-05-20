"""
1D Trajectory Coordinate Space Diffusion Model.

Denoises path coordinates directly as continuous values [B, T, 2]
instead of using 2D heatmap grids.  The model uses 1D temporal
convolutions over the path timeline and queries the elevation map
via grid_sample in the bottleneck.
"""

import math
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = x[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# 1D Temporal residual block with dilation and time conditioning
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    Two 1D convolutions (kernel_size=5) with configurable dilation on the
    first layer.  The timestep embedding is projected and added between
    the two convolutions.  A skip connection wraps the whole block.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 5,
                               padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.skip = (nn.Conv1d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]
        h = self.conv2(h)
        return F.silu(h) + self.skip(x)


# ---------------------------------------------------------------------------
# 1D Trajectory U-Net
# ---------------------------------------------------------------------------

class TrajectoryUNet1D(nn.Module):
    """
    Accepts:
      trajectory    [B, T, 2]   continuous (x, y) waypoints, normalised ~[-1, 1]
      timestep      [B]         diffusion timestep index
      elevation_map [B, 1, H, W]  terrain height grid (any resolution)

    Returns:
      [B, T, 2]  denoised trajectory (or predicted noise, depending on usage)
    """
    def __init__(self, time_dim: int = 128, T: int = 64):
        super().__init__()
        self.T = T

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb1D(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Global terrain context (resolution-agnostic via adaptive pooling).
        # Helps when local elevation queries are noisy early in diffusion.
        self.elev_global_mlp = nn.Sequential(
            nn.Linear(8 * 8, time_dim),
            nn.SiLU(),
        )

        # ---- encoder: 3 downsampling blocks (stride=2) ----
        # Input is (x,y) plus explicit start/goal conditioning repeated over time.
        # Channels: [x,y,start_x,start_y,goal_x,goal_y] => 6.
        self.enc_in = nn.Conv1d(6, 32, 3, padding=1)
        self.enc_block1 = ResidualBlock1D(32, 32, time_dim, dilation=1)
        self.enc_down1 = nn.Conv1d(32, 64, 3, stride=2, padding=1)

        self.enc_block2 = ResidualBlock1D(64, 64, time_dim, dilation=2)
        self.enc_down2 = nn.Conv1d(64, 128, 3, stride=2, padding=1)

        self.enc_block3 = ResidualBlock1D(128, 128, time_dim, dilation=4)
        self.enc_down3 = nn.Conv1d(128, 128, 3, stride=2, padding=1)

        # ---- bottleneck: grid-sample query into the elevation map ----
        self.elev_proj = nn.Conv1d(1, 128, 1)

        # ---- decoder: 3 upsampling blocks (transpose stride=2) ----
        self.dec_up1 = nn.ConvTranspose1d(128, 128, 4, stride=2, padding=1)
        self.dec_block1 = ResidualBlock1D(256, 64, time_dim, dilation=1)

        self.dec_up2 = nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1)
        self.dec_block2 = ResidualBlock1D(128, 32, time_dim, dilation=1)

        self.dec_up3 = nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1)
        self.dec_block3 = ResidualBlock1D(64, 32, time_dim, dilation=1)

        self.dec_out = nn.Conv1d(32, 2, 3, padding=1)

    def forward(self, trajectory: torch.Tensor, timestep: torch.Tensor,
                elevation_map: torch.Tensor,
                start_coords: torch.Tensor,
                goal_coords: torch.Tensor) -> torch.Tensor:
        B, T, _ = trajectory.shape

        t_emb = self.time_mlp(timestep)
        # Add a pooled global terrain embedding (works across map resolutions).
        elev_pool = F.adaptive_avg_pool2d(elevation_map, (8, 8)).flatten(1)
        t_emb = t_emb + self.elev_global_mlp(elev_pool)

        # convert to [B, C, T] for 1D convs
        traj_ch = trajectory.permute(0, 2, 1).contiguous()  # [B, 2, T]
        start_ch = start_coords.to(traj_ch.dtype)[:, :, None].expand(B, 2, T)
        goal_ch = goal_coords.to(traj_ch.dtype)[:, :, None].expand(B, 2, T)
        x = torch.cat([traj_ch, start_ch, goal_ch], dim=1)  # [B, 6, T]

        # ---- encoder ----
        h = self.enc_in(x)                                       # [B, 32, T]
        h = self.enc_block1(h, t_emb)
        skip1 = h
        h = self.enc_down1(h)                                    # [B, 64, T/2]

        h = self.enc_block2(h, t_emb)
        skip2 = h
        h = self.enc_down2(h)                                    # [B, 128, T/4]

        h = self.enc_block3(h, t_emb)
        skip3 = h
        h = self.enc_down3(h)                                    # [B, 128, T/8]

        # ---- bottleneck: query elevation at bottleneck time steps ----
        # down-sample the original trajectory to T/8 steps
        T8 = T // 8
        coords = trajectory.permute(0, 2, 1)                     # [B, 2, T]
        # Early diffusion steps are very noisy; lightly smooth the query points
        # so elevation conditioning isn't pure noise.
        coords = F.avg_pool1d(coords, kernel_size=5, stride=1, padding=2)
        coords_dn = F.interpolate(coords, size=T8, mode="linear", align_corners=False)
        coords_dn = coords_dn.permute(0, 2, 1).unsqueeze(2)     # [B, T8, 1, 2]

        elev_feat = F.grid_sample(elevation_map, coords_dn,
                                   mode="bilinear", padding_mode="border",
                                   align_corners=False)
        elev_feat = elev_feat.squeeze(-1)                        # [B, 1, T8]

        h = h + self.elev_proj(elev_feat)

        # ---- decoder ----
        h = self.dec_up1(h)                                      # [B, 128, T/4]
        h = torch.cat([h, skip3], dim=1)                         # [B, 256, T/4]
        h = self.dec_block1(h, t_emb)                            # [B, 64, T/4]

        h = self.dec_up2(h)                                      # [B, 64, T/2]
        h = torch.cat([h, skip2], dim=1)                         # [B, 128, T/2]
        h = self.dec_block2(h, t_emb)                            # [B, 32, T/2]

        h = self.dec_up3(h)                                      # [B, 32, T]
        h = torch.cat([h, skip1], dim=1)                         # [B, 64, T]
        h = self.dec_block3(h, t_emb)                            # [B, 32, T]

        out = self.dec_out(h).permute(0, 2, 1)                   # [B, T, 2]
        return out


# ---------------------------------------------------------------------------
# DDPM manager with in-painting sampling
# ---------------------------------------------------------------------------

class TrajectoryDDPM:
    """
    DDPM noise schedule + training helpers for 1D trajectory data.

    Sampling applies an **in-painting constraint**: at every denoising step
    the start (index 0) and goal (index T-1) are overwritten with the
    ground-truth coordinates, locking the path boundaries.
    """
    def __init__(self, model: nn.Module, num_timesteps: int = 200,
                 device: str = "cpu",
                 smooth_weight: float = 0.05):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        self.smooth_weight = smooth_weight

        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # ---- forward diffusion (for training) ----

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sab = self.sqrt_alpha_bar[t][:, None, None]
        som = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sab * x0 + som * noise, noise

    def train_step(self, batch, optimizer: torch.optim.Optimizer) -> float:
        trajectory, elevation_map, start_coords, goal_coords = batch
        trajectory = trajectory.to(self.device)
        elevation_map = elevation_map.to(self.device)
        start_coords = start_coords.to(self.device)
        goal_coords = goal_coords.to(self.device)
        B = trajectory.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        xt, noise = self.q_sample(trajectory, t)

        # In-paint endpoints during training so the model learns to denoise the
        # interior conditioned on fixed start/goal.
        xt[:, 0, :] = start_coords
        xt[:, -1, :] = goal_coords

        pred_noise = self.model(xt, t, elevation_map, start_coords, goal_coords)

        if xt.shape[1] > 2:
            core = slice(1, -1)
            loss = F.mse_loss(pred_noise[:, core, :], noise[:, core, :])
        else:
            loss = F.mse_loss(pred_noise, noise)

        # Smoothness bias on the predicted clean trajectory (x0). This reduces
        # jitter without forcing straight lines too aggressively.
        if self.smooth_weight > 0.0 and xt.shape[1] > 2:
            sab = self.sqrt_alpha_bar[t][:, None, None]
            som = self.sqrt_one_minus_alpha_bar[t][:, None, None]
            x0_pred = (xt - som * pred_noise) / (sab + 1e-8)
            acc = x0_pred[:, 2:, :] - 2.0 * x0_pred[:, 1:-1, :] + x0_pred[:, :-2, :]
            loss = loss + self.smooth_weight * (acc.pow(2).mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # ---- reverse sampling with in-painting ----

    @torch.no_grad()
    def sample(self, elevation_map: torch.Tensor,
               start_coords: torch.Tensor, goal_coords: torch.Tensor,
               num_steps: int | None = None) -> torch.Tensor:
        """
        Denoise from pure noise, clamping the first and last waypoint
        to `start_coords` and `goal_coords` at every step.

        Args:
            elevation_map  [B, 1, H, W]
            start_coords   [B, 2]  (normalised -1 … 1)
            goal_coords    [B, 2]  (normalised -1 … 1)
            num_steps       number of denoising steps (default = num_timesteps)

        Returns:
            trajectory [B, T, 2]
        """
        num_steps = num_steps or self.num_timesteps
        device = next(self.model.parameters()).device
        B = elevation_map.shape[0]
        T = self.model.T

        # stride-subsample the schedule for fast sampling (DDPM-style)
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_steps,
                                      dtype=torch.long, device=device)

        x_t = torch.randn(B, T, 2, device=device)

        # helper: clamp boundaries
        def _clamp(x):
            x[:, 0, :] = start_coords
            x[:, -1, :] = goal_coords
            return x

        x_t = _clamp(x_t)

        for i in range(num_steps - 1, -1, -1):
            t_idx = step_indices[i]
            t = t_idx.unsqueeze(0).expand(B)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]

            pred_noise = self.model(x_t, t, elevation_map, start_coords, goal_coords)

            # DDPM reverse step
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_bar)
            mu = coef1 * (x_t - coef2 * pred_noise)

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mu + torch.sqrt(beta) * noise
            else:
                x_t = mu

            x_t = _clamp(x_t)

        return x_t  # [B, T, 2]


# ---------------------------------------------------------------------------
# Inference: run a trained TrajectoryUNet1D on a GridWorld
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_path_coord(model, world, start: tuple[int, int], goal: tuple[int, int],
                     checkpoint: str = "data/diffusion_coord_model.pt",
                     T: int = 64,
                     num_train_steps: int = 200,
                     num_sample_steps: int = 50,
                     device: str = "cpu") -> tuple[list, list]:
    """
    Inference loop for 1D Coordinate Diffusion.
    Generates a continuous sequence of T coordinates, then scales them back to pixels.
    """
    # 1. Load and prepare the model
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    H, W = world.grid.shape  # Original map pixel dimensions

    # 2. Normalize the raw elevation map to [0, 1] context
    elev_np = world.grid.astype(np.float32)
    if elev_np.max() > 0:
        elev_np = elev_np / elev_np.max()
    elev_t = torch.from_numpy(elev_np).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    # 3. Normalize Start/Goal pixels to continuous [-1.0, 1.0] coordinates
    def norm_coords(px, py):
        nx = -1.0 + 2.0 * px / (W - 1)
        ny = -1.0 + 2.0 * py / (H - 1)
        return torch.tensor([nx, ny], dtype=torch.float32, device=device)

    start_norm = norm_coords(start[0], start[1])  # [2]
    goal_norm = norm_coords(goal[0], goal[1])    # [2]
    start_b = start_norm.unsqueeze(0)  # [1, 2]
    goal_b = goal_norm.unsqueeze(0)    # [1, 2]

    # 4. Set up the DDIM alpha noise schedule
    betas = torch.linspace(1e-4, 0.02, num_train_steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    step_indices = torch.linspace(0, num_train_steps - 1, num_sample_steps, dtype=torch.long, device=device)

    # 5. Initialize the path as pure Gaussian noise: [1, T, 2]
    x_t = torch.randn((1, T, 2), device=device)

    # --- FIRST IN-PAINTING LOCK ---
    # Force the very first and last steps of our random noise to match our targets
    x_t[0, 0] = start_norm
    x_t[0, -1] = goal_norm

    # 6. Reverse Denoising Loop
    for i in range(num_sample_steps - 1, -1, -1):
        t_idx = step_indices[i]
        t = t_idx.unsqueeze(0)

        alpha_cur = alpha_bar[t_idx]
        alpha_prev = alpha_bar[step_indices[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

        # Predict the noise vector using our U-Net
        pred_noise = model(x_t, t, elev_t, start_b, goal_b)

        # DDIM update step math
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_cur) * pred_noise) / torch.sqrt(alpha_cur)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)  # Keep coordinates inside the map bounds

        dir_xt = torch.sqrt(1.0 - alpha_prev) * pred_noise
        x_t = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        # --- DYNAMIC IN-PAINTING CONSTRAINT ---
        # Overwrite the endpoints at every single step so they cannot drift!
        x_t[0, 0] = start_norm
        x_t[0, -1] = goal_norm

    # 7. SCALE-BACK PHASE: Convert continuous [-1.0, 1.0] vectors back to grid pixels
    final_coords = x_t[0].cpu().numpy()  # Shape: [T, 2]
    pixel_path = []

    for nx, ny in final_coords:
        # Inverse normalization math formulas
        px = int(np.round(((nx + 1.0) / 2.0) * (W - 1)))
        py = int(np.round(((ny + 1.0) / 2.0) * (H - 1)))

        # Hard safety boundaries to prevent array index out-of-bounds errors
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        # Avoid saving duplicated sequential coordinates if the model stays on a pixel
        if not pixel_path or pixel_path[-1] != (px, py):
            pixel_path.append((px, py))

    if not pixel_path:
        return [], []

    # Hard-enforce endpoints after rounding/dedup so rendering and post-processing
    # can't accidentally "lose" them.
    if pixel_path[0] != start:
        pixel_path.insert(0, start)
    else:
        pixel_path[0] = start
    if pixel_path[-1] != goal:
        pixel_path.append(goal)
    else:
        pixel_path[-1] = goal

    # 8. Symbolic post-pass: convert the (possibly jagged) trajectory to a
    # grid-valid path via a Dijkstra search guided by the trajectory corridor.
    refined = _extract_grid_path_from_trajectory(
        trajectory_pixels=pixel_path,
        elevation_grid=world.grid,
        start=start,
        goal=goal,
    )

    # If refinement fails (e.g. disconnected due to obstacles), fall back to
    # the raw predicted pixels so the UI still shows something.
    out = refined if refined else pixel_path
    return out, out


def _extract_grid_path_from_trajectory(
    trajectory_pixels: list[tuple[int, int]],
    elevation_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    corridor_weight: float = 50.0,
    climb_weight: float = 20.0,
) -> list[tuple[int, int]]:
    """Terrain-aware Dijkstra guided by the predicted trajectory.

    This is the symbolic piece that turns a continuous/jagged waypoint sequence
    into a contiguous grid-valid path.

    Cost per step:
      step_cost   = 1.0
      climb_cost  = max(0, Δh) * climb_weight
      corridor    = (dist_to_trajectory / diag)^2 * corridor_weight
    """
    H, W = elevation_grid.shape

    def passable(x: int, y: int) -> bool:
        v = elevation_grid[y, x]
        return (v >= 0) and (not np.isinf(v))

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return []
    if not passable(sx, sy) or not passable(gx, gy):
        return []

    # Precompute a cheap distance-to-trajectory field.
    # For 100x100 with T~64 this is fast and stable.
    yy, xx = np.indices((H, W))
    dist2 = np.full((H, W), np.inf, dtype=np.float32)
    for px, py in trajectory_pixels:
        if 0 <= px < W and 0 <= py < H:
            d2 = (xx - px) ** 2 + (yy - py) ** 2
            dist2 = np.minimum(dist2, d2.astype(np.float32))
    dist = np.sqrt(dist2)
    diag = float(np.hypot(H - 1, W - 1)) + 1e-8
    corridor_cost = ((dist / diag) ** 2).astype(np.float32)

    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while heap:
        cost, current = heapq.heappop(heap)
        if current == goal:
            break

        cx, cy = current
        cur_elev = float(elevation_grid[cy, cx])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not passable(nx, ny):
                continue

            tgt_elev = float(elevation_grid[ny, nx])
            step_cost = 1.0
            climb_cost = max(0.0, tgt_elev - cur_elev) * climb_weight
            traj_cost = float(corridor_cost[ny, nx]) * corridor_weight
            move_cost = step_cost + climb_cost + traj_cost

            new_cost = cost_so_far[current] + move_cost
            nxt = (nx, ny)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(heap, (new_cost, nxt))

    if goal not in came_from:
        return []

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        path.append(cur)
        parent = came_from.get(cur)
        if parent is None:
            return []
        cur = parent
    path.append(start)
    path.reverse()
    return path
