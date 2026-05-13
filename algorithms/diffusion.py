import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ElevationGridDataset(torch.utils.data.Dataset):
    """
    Handles raw NumPy 2D ndarrays where values are height (z).
    Input: List of [H, W] ndarrays.
    """
    def __init__(self, elevation_grids, path_grids):
        # Ensure data is float32 for PyTorch
        self.elevation_grids = [g.astype(np.float32) for g in elevation_grids]
        self.path_grids = [p.astype(np.float32) for p in path_grids]

    def __len__(self):
        return len(self.elevation_grids)

    def __getitem__(self, idx):
        # Convert NumPy [H, W] -> PyTorch [1, H, W]
        elev = torch.from_numpy(self.elevation_grids[idx]).unsqueeze(0)
        path = torch.from_numpy(self.path_grids[idx]).unsqueeze(0)

        # SENIOR TIP: Normalization
        # Neural nets hate raw values like "1200 meters".
        # We normalize elevation to [0, 1] based on the map's local max.
        if elev.max() > 0:
            elev = elev / elev.max()

        # Apply the augmentations we discussed
        return self.augment(elev, path)

    def augment(self, elev, path):
        # Random D4 rotations (90, 180, 270)
        k = np.random.randint(0, 4)
        elev = torch.rot90(elev, k, dims=[-2, -1])
        path = torch.rot90(path, k, dims=[-2, -1])

        # Horizontal Flip
        if np.random.rand() > 0.5:
            elev = torch.flip(elev, dims=[-1])
            path = torch.flip(path, dims=[-1])

        return elev, path


class SinusoidalPosEmb(nn.Module):
    """
    Standard Sinusoidal Positional Embedding for Diffusion Timesteps.
    Transforms a scalar 't' into a high-dimensional vector.

    Rationale: Neural networks struggle to learn from raw integers. By projecting
    time onto sine and cosine waves of varying frequencies, we provide a
    continuous geometric representation that the model can easily interpret.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaGNBlock(nn.Module):
    """
    Adaptive Group Normalization (AdaGN) Residual Block.

    Why AdaGN?
    In diffusion, the data distribution changes at every step t. AdaGN allows
    the timestep to 'rescale' and 're-bias' the feature maps globally.
    This is significantly more expressive than simple addition.

    Components:
    - GroupNorm: Standardizes features across channel groups (stable at Batch Size 1).
    - Time Projection: Predicts Gamma (scale) and Beta (shift) for the normalization.
    - SiLU: Sigmoid Linear Unit (Swish); standard in modern diffusion for smoother gradients.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Time MLP projects the time embedding into a Scale and Shift pair
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.SiLU()

    def forward(self, x, t_emb):
        # 1. Initial normalization and convolution
        h = self.conv1(self.relu(self.norm1(x)))

        # 2. Extract Adaptive Parameters (Gamma and Beta)
        # Using .chunk(2, dim=1) splits the linear output into two equal halves
        t_params = self.time_mlp(self.relu(t_emb))[:, :, None, None]
        gamma, beta = torch.chunk(t_params, 2, dim=1)

        # 3. Apply the time-conditioned transformation
        # The '1 +' allows the model to start with an identity mapping
        h = h * (1 + gamma) + beta

        # 4. Final convolution
        h = self.conv2(self.relu(self.norm2(h)))
        return h

class AttentionBlock(nn.Module):
    """
    Self-Attention Block for Global Spatial Reasoning.

    Why Attention?
    Convolutions are 'local' (only see 3x3). In pathfinding, the model needs
    to see across the whole map to understand if a valley leads to a dead end.
    This layer allows every pixel to 'attend' to every other pixel.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Using 4 heads for a balance between detail and computation
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        # Re-arrange from [B, C, H, W] to [B, Sequence_Length, Channels]
        # where Sequence_Length = H * W
        x_norm = self.norm(x).view(b, c, h * w).transpose(1, 2)

        attn_out, _ = self.mha(x_norm, x_norm, x_norm)

        # Residual connection: Add the attention result back to the original input
        return x + attn_out.transpose(1, 2).view(b, c, h, w)

class SeniorElevationUNet(nn.Module):
    """
    High-Fidelity Diffusion UNet for 2.5D Pathfinding.

    Inputs:
    - x: The noisy trajectory/path [Batch, 1, H, W]
    - t: The current diffusion timestep [Batch]
    - elevation: The static elevation map [Batch, 1, H, W]
    """
    def __init__(self, in_channels=2, out_channels=1, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Encoder: Compressing the image to extract high-level terrain features
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = AdaGNBlock(64, 128, time_dim)
        self.down2 = AdaGNBlock(128, 256, time_dim)

        # Bottleneck: The most compressed representation
        self.mid_block = AdaGNBlock(256, 256, time_dim)
        self.mid_attn = AttentionBlock(256)

        # Decoder: Reconstructing the clean path using skip-connections
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.rev1 = AdaGNBlock(256, 128, time_dim) # Concat channels (128+128=256)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.rev2 = AdaGNBlock(128, 64, time_dim) # Concat channels (64+64=128)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t, elevation):
        t_emb = self.time_mlp(t)
        # Concatenate path and terrain so the model treats them as one environment
        x_in = torch.cat([x, elevation], dim=1)

        # Downward Pass
        h0 = self.init_conv(x_in)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(F.max_pool2d(h1, 2), t_emb)

        # Bottleneck (Low-res global reasoning)
        h_mid = self.mid_block(F.max_pool2d(h2, 2), t_emb)
        h_mid = self.mid_attn(h_mid)

        # Upward Pass with Skip Connections (concatenating 'hi' variables)
        u1 = self.up1(h_mid)
        u1 = self.rev1(torch.cat([u1, h2], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.rev2(torch.cat([u2, h1], dim=1), t_emb)

        return self.final_conv(u2)

@torch.no_grad()
def ddim_sample(model, elevation, start_map, end_map, steps=50, eta=0.0):
    """
    DDIM Sampling Loop for Path Generation.
    - elevation, start_map, end_map: Your [1, 1, H, W] grid constraints.
    - steps: Number of denoising steps (50 is usually plenty).
    - eta: 0.0 for deterministic sampling (faster/cleaner), 1.0 for stochastic (DDPM style).
    """
    device = next(model.parameters()).device
    b, _, h, w = elevation.shape

    # 1. Start with pure Gaussian Noise in the 'Path' channel
    x_t = torch.randn((b, 1, h, w), device=device)

    # 2. Define the noise schedule (alphas)
    # In a real setup, these would match your training schedule
    alphas = torch.linspace(0.99, 0.01, steps).to(device)

    for i in range(steps):
        # Current timestep t
        t = torch.full((b,), i, device=device, dtype=torch.long)

        # 3. Predict the noise using the UNet
        # UNet input: Noisy Path + Elevation + Start + End (4 Channels)
        pred_noise = model(x_t, t, elevation, start_map, end_map)

        # 4. DDIM Update Rule: Move x_t toward the "clean" path
        # We calculate the 'predicted x0' (the clean path)
        alpha = alphas[i]
        alpha_prev = alphas[i+1] if i < steps - 1 else torch.tensor(1.0)

        # Simplified DDIM step:
        # 1. Estimate the clean path (x0_recon)
        x0_recon = (x_t - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)

        # 2. Project forward to the next step
        direction = torch.sqrt(1 - alpha_prev) * pred_noise
        x_t = torch.sqrt(alpha_prev) * x0_recon + direction

    # Final result is the denoised path
    return x_t
