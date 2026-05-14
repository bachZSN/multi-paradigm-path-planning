"""
Train the grid-based diffusion model from algorithms/diffusion.py.

Pipeline:
  1. Generate (world, start, goal, path) pairs using A* on random GridWorlds
  2. Save / load the dataset to disk
  3. Train PathUNet via DDPM noise prediction
  4. Sample paths with DDIM and compare to ground-truth A* paths

Usage (from project root):
  python -m experiments.train_diffusion                    # fresh generate + train
  python -m experiments.train_diffusion --epochs 100       # override training epochs
"""

import os
import argparse
import numpy as np
import torch

from algorithms.diffusion import (
    PathfindingDataset,
    generate_dataset,
    PathUNet,
    DDPM,
    ddim_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--num-worlds", type=int, default=500)
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
        dataset = generate_dataset(
            world_size=args.world_size,
            num_worlds=args.num_worlds,
            num_obstacles=args.obstacles_per_world,
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
        pred_path = ddim_sample(
            model,
            elevation[:4],
            start_map[:4],
            goal_map[:4],
            num_train_steps=args.num_timesteps,
            num_sample_steps=args.ddim_steps,
        )

    for i in range(min(4, pred_path.shape[0])):
        pred_pixels = (pred_path[i, 0] > 0.5).int().cpu().numpy()
        gt_pixels = (gt_path[i, 0] > 0.5).int().cpu().numpy()
        overlap = (pred_pixels == gt_pixels).mean()
        print(f"  Sample {i+1}: pixel-accuracy = {overlap:.3f}")

    print("\nDone. Run with different seeds or more epochs to improve quality.")


if __name__ == "__main__":
    main()
