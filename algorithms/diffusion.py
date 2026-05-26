import math
import time
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

        # ---- encoder ----
        h0 = self.init_conv(x_in)                           # 32x32, 64ch
        h1 = self.down1(h0, t_emb)                          # 32x32, 128ch
        h1_pooled = F.max_pool2d(h1, 2)                     # 16x16, 128ch
        h2 = self.down2(h1_pooled, t_emb)                   # 16x16, 256ch
        h2_pooled = F.max_pool2d(h2, 2)                     #  8x8, 256ch

        # ---- bottleneck ----
        h_mid = self.mid_block(h2_pooled, t_emb)            #  8x8, 256ch
        h_mid = self.mid_attn(h_mid)                        #  8x8, 256ch

        # ---- decoder ----
        u1 = self.up1(h_mid)                                # 16x16, 128ch
        u1 = self.rev1(torch.cat([u1, h2], dim=1), t_emb)   # concat 128+256=384

        u2 = self.up2(u1)                                   # 32x32,  64ch
        u2 = self.rev2(torch.cat([u2, h1], dim=1), t_emb)   # concat  64+128=192

        return self.final_conv(u2)                          # 32x32,   1ch


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
# DDIM sampling with Classifier-Free Guidance (CFG)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_cfg(model: nn.Module,
                    elevation: torch.Tensor,
                    start_map: torch.Tensor,
                    goal_map: torch.Tensor,
                    num_train_steps: int = 200,
                    num_sample_steps: int = 50,
                    eta: float = 0.0,
                    guidance_scale: float = 4.0) -> torch.Tensor:
    """
    DDIM sampling with Classifier-Free Guidance.

    At each denoising step the model is evaluated twice:
      - conditioned  on (elevation, start_map, goal_map)
      - unconditioned on zero tensors of the same shape

    The final noise estimate is extrapolated:
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    guidance_scale=1.0 reproduces standard conditioned DDIM.
    """
    device = next(model.parameters()).device
    B = elevation.shape[0]
    H, W = elevation.shape[-2:]

    # noise schedule matching training
    betas = torch.linspace(1e-4, 0.02, num_train_steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    step_indices = torch.linspace(0, num_train_steps - 1, num_sample_steps, dtype=torch.long, device=device)

    # null conditioning for CFG unconditional pass
    null_cond = torch.zeros_like(elevation)

    x_t = torch.randn((B, 1, H, W), device=device)

    for i in range(num_sample_steps - 1, -1, -1):
        t_idx = step_indices[i]
        t = t_idx.unsqueeze(0).expand(B)

        alpha_cur = alpha_bar[t_idx]
        alpha_prev = alpha_bar[step_indices[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

        # CFG: two forward passes
        eps_cond = model(x_t, t, elevation, start_map, goal_map)
        eps_uncond = model(x_t, t, null_cond, null_cond, null_cond)
        pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM update
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
               guidance_scale: float = 4.0,
               device: str = "cpu",
               metrics: dict | None = None,
               refine: str = "dijkstra") -> tuple[list, list]:
    """
    Run the trained diffusion model to predict a path between start and goal.

    Steps:
      1. Load model weights from checkpoint
      2. Downsample the world grid to model_size × model_size
      3. Build start/goal heatmaps in model coordinates
      4. Run CFG-guided DDIM sampling to get a path-occupancy grid
      5. Upsample the predicted grid to the original world resolution
      6. Run Dijkstra on the full-res confidence map with terrain awareness

    Returns:
        (explored_path, shortest_path) — both set to the same predicted path
        on success, or ([], []) on failure.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    H, W = world.grid.shape  # grid is stored as grid[y, x]
    raw_elevation = world.grid.astype(np.float32)

    # 1. Prepare elevation for model input (downsampled + normalised)
    elev_t = torch.from_numpy(raw_elevation).unsqueeze(0).unsqueeze(0)
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

    # 3. Run CFG-guided DDIM
    elev_t = elev_t.to(device)
    start_map = start_map.to(device)
    goal_map = goal_map.to(device)

    t_sample_start = time.perf_counter()
    pred = ddim_sample_cfg(
        model,
        elev_t,
        start_map,
        goal_map,
        num_train_steps=num_train_steps,
        num_sample_steps=num_sample_steps,
        guidance_scale=guidance_scale,
    )
    t_sample_end = time.perf_counter()
    if metrics is not None:
        metrics["grid.sample_seconds"] = t_sample_end - t_sample_start
    # pred is [1, 1, model_size, model_size]

    # 4. Upsample to full world resolution
    pred_world = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    path_confidence = pred_world[0, 0].cpu().numpy()  # [H, W]

    # 5. Extract path at full resolution using terrain-aware search
    t_refine_start = time.perf_counter()
    path_coords, explored, refine_stats = _extract_path(path_confidence, raw_elevation, start, goal, refine=refine)
    t_refine_end = time.perf_counter()
    if metrics is not None:
        metrics["grid.refine_seconds"] = t_refine_end - t_refine_start
        metrics["grid.total_seconds"] = t_refine_end - t_sample_start
        metrics["grid.refine_expanded"] = int(refine_stats.get("expanded", 0))
        metrics["grid.refine_pushed"] = int(refine_stats.get("pushed", 0))
        metrics["grid.refine_discovered"] = int(refine_stats.get("discovered", 0))
    return (explored, path_coords) if path_coords else ([], [])


def _extract_path(confidence_grid: np.ndarray,
                  elevation_grid: np.ndarray,
                  start: tuple[int, int],
                  goal: tuple[int, int],
                  *,
                  refine: str = "dijkstra") -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    """
    Terrain-aware Dijkstra guided by the model's confidence map.

    Cost per step is a weighted sum of three terms:
      1.  step_cost = 1.0                    (base movement)
      2.  climb_penalty = max(0, Δh) × 20    (uphill penalty)
      3.  model_penalty = 50 × (1 − prob)    (deviation from model corridor)

    The model output is passed through a sigmoid to produce a smooth
    probability field — the search is encouraged to stay within the
    model's high-confidence corridor while respecting terrain cost.
    """
    import heapq

    H, W = confidence_grid.shape
    prob_map = 1.0 / (1.0 + np.exp(-confidence_grid))

    passable = (elevation_grid >= 0) & (~np.isinf(elevation_grid))
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return [], [], {"expanded": 0, "pushed": 0, "discovered": 0}
    if not passable[sy, sx] or not passable[gy, gx]:
        return [], [], {"expanded": 0, "pushed": 0, "discovered": 0}

    # Costs derived from the model output (confidence) and terrain.
    # refine:
    #   - "dijkstra": priority = g
    #   - "astar_model": priority = g + h_model (admissible h from relaxed cost)
    #   - "greedy_model": priority = g + h + extra_model_bias (fast, not optimal)

    if refine not in {"dijkstra", "astar_model", "greedy_model"}:
        raise ValueError(f"Unknown refine mode: {refine}")

    # Precompute an admissible heuristic to-go under a relaxed cost that
    # drops climb but keeps step + model terms. This is a lower bound on the
    # full cost so A* remains optimal.
    h_to_goal = None
    if refine == "astar_model":
        h_to_goal = _relaxed_cost_to_goal(prob_map, passable, goal)

    stats = {"expanded": 0, "pushed": 0, "discovered": 1}

    # Heap items are (priority, g_cost, node). We keep g for stale-pop checks.
    heap: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    explored: list[tuple[int, int]] = []

    while heap:
        priority, g_cost, current = heapq.heappop(heap)
        # Skip stale heap entries.
        if g_cost != cost_so_far.get(current):
            continue

        explored.append(current)
        stats["expanded"] += 1
        if current == goal:
            break
        cx, cy = current
        current_elev = elevation_grid[cy, cx]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not passable[ny, nx]:
                continue

            target_elev = elevation_grid[ny, nx]
            step_cost = 1.0
            climb_penalty = max(0.0, target_elev - current_elev) * 20.0
            model_penalty = 50.0 * (1.0 - prob_map[ny, nx])
            total_move_cost = step_cost + climb_penalty + model_penalty

            new_cost = cost_so_far[current] + total_move_cost
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                if (nx, ny) not in cost_so_far:
                    stats["discovered"] += 1
                cost_so_far[(nx, ny)] = new_cost

                # Priority selection
                priority = new_cost
                if refine == "astar_model":
                    # Admissible heuristic (lower bound) from relaxed cost.
                    priority = new_cost + float(h_to_goal[ny, nx])
                elif refine == "greedy_model":
                    # Fast but not optimal: bias directly toward high confidence.
                    # Note: this double-counts model guidance (also in model_penalty).
                    priority = new_cost + 10.0 * (1.0 - prob_map[ny, nx])

                heapq.heappush(heap, (priority, new_cost, (nx, ny)))
                stats["pushed"] += 1
                came_from[(nx, ny)] = current

    if goal not in came_from:
        return [], explored, stats

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path, explored, stats


def _relaxed_cost_to_goal(prob_map: np.ndarray,
                          passable: np.ndarray,
                          goal: tuple[int, int],
                          *,
                          step_cost: float = 1.0,
                          model_weight: float = 50.0) -> np.ndarray:
    """Compute admissible heuristic h(n) to goal under relaxed cost.

    Relaxed edge cost drops climb and keeps:
      w' = step_cost + model_weight * (1 - prob_map)

    Since w' <= w_full, shortest-path distance under w' is a lower bound
    on the true remaining cost, so it is admissible for A*.
    """
    import heapq

    H, W = prob_map.shape
    gx, gy = goal
    h = np.full((H, W), np.inf, dtype=np.float32)
    if not (0 <= gx < W and 0 <= gy < H):
        return h
    if not passable[gy, gx]:
        return h

    h[gy, gx] = 0.0
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, goal)]

    while heap:
        cost, (cx, cy) = heapq.heappop(heap)
        if cost != h[cy, cx]:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not passable[ny, nx]:
                continue
            w = step_cost + model_weight * (1.0 - float(prob_map[ny, nx]))
            nc = cost + w
            if nc < h[ny, nx]:
                h[ny, nx] = nc
                heapq.heappush(heap, (nc, (nx, ny)))

    return h
