"""
1D Trajectory Coordinate Space Diffusion Model.

Denoises path coordinates directly as continuous values [B, T, 2]
instead of using 2D heatmap grids.  The model uses 1D temporal
convolutions over the path timeline and queries the elevation map
via grid_sample in the bottleneck.
"""

import math
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

        # ---- encoder: 3 downsampling blocks (stride=2) ----
        self.enc_in = nn.Conv1d(2, 32, 3, padding=1)
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
                elevation_map: torch.Tensor) -> torch.Tensor:
        B, T, _ = trajectory.shape
        t_emb = self.time_mlp(timestep)

        # convert to [B, C, T] for 1D convs
        x = trajectory.permute(0, 2, 1).contiguous()  # [B, 2, T]

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
                 device: str = "cpu"):
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

    # ---- forward diffusion (for training) ----

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sab = self.sqrt_alpha_bar[t][:, None, None]
        som = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sab * x0 + som * noise, noise

    def train_step(self, batch, optimizer: torch.optim.Optimizer) -> float:
        trajectory, elevation_map, _start_coords, _goal_coords = batch
        trajectory = trajectory.to(self.device)
        elevation_map = elevation_map.to(self.device)
        B = trajectory.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        xt, noise = self.q_sample(trajectory, t)
        pred_noise = self.model(xt, t, elevation_map)
        loss = F.mse_loss(pred_noise, noise)
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

            pred_noise = self.model(x_t, t, elevation_map)

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
        pred_noise = model(x_t, t, elev_t)

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

    # Return the clean pixel path list: [(x0, y0), (x1, y1), ...]
    return pixel_path, pixel_path
