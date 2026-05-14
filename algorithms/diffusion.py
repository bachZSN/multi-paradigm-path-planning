import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.rev1 = AdaGNBlock(384, 128, time_dim)   # cat(u1[128], h2[256])

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.rev2 = AdaGNBlock(192, 64, time_dim)    # cat(u2[64], h1[128])

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


# ---------------------------------------------------------------------------
# Inference: run a trained model on a GridWorld and extract a path
# ---------------------------------------------------------------------------

def infer_path(model, world, start, goal,
               checkpoint: str = "data/diffusion_model.pt",
               model_size: int = 32,
               num_train_steps: int = 200,
               num_sample_steps: int = 50,
               device: str = "cpu") -> tuple[list, list]:
    """
    Run the trained diffusion model to predict a path between start and goal.

    Steps:
      1. Load model weights from checkpoint
      2. Downsample the world grid to model_size × model_size
      3. Build start/goal heatmaps in model coordinates
      4. Run DDIM sampling to get a predicted path-occupancy grid
      5. Treat the output as a cost map (high prob = low cost) and run
         Dijkstra on it to extract a valid path
      6. Map the path back to the original world coordinates

    Returns:
        (explored_path, shortest_path) — both set to the same predicted path
        on success, or ([], []) on failure.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    H, W = world.grid.shape  # grid is stored as grid[y, x]

    # 1. Prepare elevation
    elev_t = torch.from_numpy(world.grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if H != model_size or W != model_size:
        elev_t = F.interpolate(elev_t, size=(model_size, model_size),
                               mode="bilinear", align_corners=False)
    if elev_t.max() > 0:
        elev_t = elev_t / elev_t.max()

    # 2. Build start/goal heatmaps in model coordinates
    sx = int(start[0] * model_size / W)
    sy = int(start[1] * model_size / H)
    gx = int(goal[0] * model_size / W)
    gy = int(goal[1] * model_size / H)

    ys, xs = torch.meshgrid(
        torch.arange(model_size, dtype=torch.float32),
        torch.arange(model_size, dtype=torch.float32),
        indexing="ij",
    )
    sigma = 1.5
    start_map = torch.exp(-((xs - sx) ** 2 + (ys - sy) ** 2) / (2 * sigma ** 2))
    start_map = start_map.unsqueeze(0).unsqueeze(0)
    goal_map = torch.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * sigma ** 2))
    goal_map = goal_map.unsqueeze(0).unsqueeze(0)

    # 3. Run DDIM
    elev_t = elev_t.to(device)
    start_map = start_map.to(device)
    goal_map = goal_map.to(device)

    pred = ddim_sample(model, elev_t, start_map, goal_map,
                       num_train_steps=num_train_steps,
                       num_sample_steps=num_sample_steps)
    path_grid = pred[0, 0].cpu().numpy()  # [model_size, model_size]

    # 4. Extract path via Dijkstra on an inverted cost map.
    #    High model output → low cost → path is guided through cells the
    #    model considers likely.
    path_coords = _extract_path(path_grid, (sx, sy), (gx, gy))
    if not path_coords:
        return [], []

    # 5. Map back to world coordinates
    world_path = []
    for px, py in path_coords:
        wx = int(px * W / model_size)
        wy = int(py * H / model_size)
        wx = max(0, min(W - 1, wx))
        wy = max(0, min(H - 1, wy))
        world_path.append((wx, wy))

    return world_path, world_path


def _extract_path(cost_grid: np.ndarray,
                  start: tuple[int, int],
                  goal: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Run Dijkstra on a cost map derived from the model output.

    The model output is normalised to [0, 1] then inverted so that
    high-confidence path cells have low cost.
    """
    import heapq

    H, W = cost_grid.shape
    gmin, gmax = float(cost_grid.min()), float(cost_grid.max())
    if gmax > gmin:
        cost_grid = (cost_grid - gmin) / (gmax - gmin)
    cost_map = 1.0 - cost_grid  # high model output → low cost

    heap = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while heap:
        cost, current = heapq.heappop(heap)
        if current == goal:
            break
        cx, cy = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H:
                new_cost = cost_so_far[current] + cost_map[ny, nx]
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    heapq.heappush(heap, (new_cost, (nx, ny)))
                    came_from[(nx, ny)] = current

    if goal not in came_from:
        return [start]

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path
