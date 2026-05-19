"""
Train the grid-based diffusion model from algorithms/diffusion.py.

Recommended workflow (from project root):
  1. Generate dataset separately:
       python -m experiments.make_dataset --num-worlds 200 --samples-per-world 5

  2. Train the model (uses cached dataset if already generated):
       python -m experiments.train_diffusion

  3. Launch the app and click "Diffusion" to see the model in action:
       python main.py

Expected training signal:
  - Epoch   1-10:  loss ~1.0 → ~0.6  (model learns basic map structure)
  - Epoch  10-30:  loss ~0.6 → ~0.3  (model learns to follow terrain)
  - Epoch  30-50+:  loss ~0.3 → ~0.15 (model refines path predictions)
  - Below 0.1:  overfitting on training worlds (consider more data)

Hyperparameter guidelines:
  --num-worlds 200     × 5 samples  =  1000 training samples  (minimum)
  --num-worlds 500     × 5 samples  =  2500 training samples  (recommended)
  --num-worlds 1000    × 5 samples  =  5000 training samples  (better generalization)
  --epochs 50-100       watch loss curve, stop when it plateaus
  --batch-size 32       good default; reduce to 16 if GPU memory is tight
"""

import os
import argparse
import numpy as np
import torch

from experiments.make_dataset import PathfindingDataset, generate_samples
from algorithms.diffusion import PathUNet, DDPM, ddim_sample_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--num-worlds", type=int, default=200,
                        help="number of random worlds (×5 samples each)")
    parser.add_argument("--obstacles-per-world", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--data-path", type=str, default="data/training_dataset.pt")
    parser.add_argument("--checkpoint", type=str, default="data/diffusion_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- dataset ---
    if os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        dataset = torch.load(args.data_path, weights_only=False)
    else:
        print(f"Generating {args.num_worlds} worlds with A* ...")
        dataset = generate_samples(
            num_worlds=args.num_worlds,
            samples_per_world=5,
            world_size=args.world_size,
            obstacles_per_world=args.obstacles_per_world,
        )
        os.makedirs("data", exist_ok=True)
        torch.save(dataset, args.data_path)
        print(f"Saved dataset ({len(dataset)} samples) to {args.data_path}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # --- model ---
    model = PathUNet(in_channels=4, out_channels=1, time_dim=256).to(device)
    ddpm = DDPM(model, num_timesteps=args.num_timesteps, device=device)
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
            print(f"  -> checkpoint saved to {args.checkpoint}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"\nFinal model saved to {args.checkpoint}")

    # --- evaluation: sample paths with DDIM ---
    print("\nSampling paths with DDIM ...")
    model.eval()
    sample_batch = next(iter(loader))
    elevation, start_map, goal_map, gt_path = [t.to(device) for t in sample_batch]

    with torch.no_grad():
        pred_path = ddim_sample_cfg(
            model,
            elevation[:4],
            start_map[:4],
            goal_map[:4],
            num_train_steps=args.num_timesteps,
            num_sample_steps=args.ddim_steps,
            guidance_scale=1.0,  # standard DDIM for evaluation
        )

    for i in range(min(4, pred_path.shape[0])):
        pred_pixels = (pred_path[i, 0] > 0.5).int().cpu().numpy()
        gt_pixels = (gt_path[i, 0] > 0.5).int().cpu().numpy()
        overlap = (pred_pixels == gt_pixels).mean()
        print(f"  Sample {i+1}: pixel-accuracy = {overlap:.3f}")

    print("\nDone. Run with different seeds or more epochs to improve quality.")


if __name__ == "__main__":
    main()
