"""Train discrete RSSM separately; Gaussian checkpoints remain untouched."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model.categorical_rssm import CategoricalRSSM
from world_model.data import SequenceDataset, split_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/episodes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--unimix", type=float, default=0.01)
    parser.add_argument("--dyn-scale", type=float, default=0.05)
    parser.add_argument("--recon-mode", choices=["mse","l1_ssim_edge"], default="mse")
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--rep-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints/categorical_best.pt")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_files, val_files = split_episodes(args.data, seed=args.seed)
    train = DataLoader(
        SequenceDataset(train_files, args.seq_len, args.stride),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    val = DataLoader(
        SequenceDataset(val_files, args.seq_len, args.stride),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    config = dict(hidden=args.hidden, groups=args.groups,
                  classes=args.classes, embed=args.embed, unimix=args.unimix)
    model = CategoricalRSSM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        summary = {}
        for split, loader in (("train", train), ("val", val)):
            training = split == "train"
            model.train(training)
            totals = {}
            count = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch} {split}"):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.set_grad_enabled(training):
                    losses = model.loss(
                        **batch, dyn_scale=args.dyn_scale,
                        rep_scale=args.rep_scale,
                        recon_mode=args.recon_mode, ssim_weight=args.ssim_weight,
                        edge_weight=args.edge_weight)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        losses["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                        optimizer.step()
                count += 1
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach())
            summary[split] = {key: total / count for key, total in totals.items()}
        print(json.dumps({"epoch": epoch, **summary}))
        if summary["val"]["loss"] < best:
            best = summary["val"]["loss"]
            torch.save({
                "model_type": "categorical",
                "model": model.state_dict(), "config": config,
                "epoch": epoch, "val": summary["val"],
                "train_files": [path.name for path in train_files],
                "val_files": [path.name for path in val_files],
                "seq_len": args.seq_len,
                "dyn_scale": args.dyn_scale, "rep_scale": args.rep_scale,
                "recon_mode": args.recon_mode, "ssim_weight": args.ssim_weight,
                "edge_weight": args.edge_weight,
            }, output)
            print("Saved", output)
    print("Best validation loss:", best, "device:", device)


if __name__ == "__main__":
    main()
