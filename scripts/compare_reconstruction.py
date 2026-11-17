"""Compare observed images, posterior reconstructions, and prior-only imagined frames.

Posterior is conditioned on each true future observation, while prior uses only
the initial observation and the recorded actions. Both share the same decoder.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from world_model.data import SequenceDataset
from world_model.checkpoints import model_from_checkpoint


def to_image(frame):
    pixels = (frame.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    return Image.fromarray(pixels.astype("uint8"))


@torch.inference_mode()
def compare_frames(model, observations, actions):
    """Return (ground_truth, posterior, prior) as [T,3,96,96] tensors.

    observations: [1,T+1,3,96,96]; actions: [1,T,3].
    Fix the random seed in main for reproducible stochastic posterior samples.
    """
    if observations.shape[0] != 1 or actions.shape[0] != 1:
        raise ValueError("This visualization expects exactly one sequence.")
    if observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("Expected T+1 observations for T actions.")

    state = model.observe_first(observations[:, 0])
    reconstructed = []
    for t in range(actions.shape[1]):
        state, _, _ = model.observe_step(
            state, actions[:, t], observations[:, t + 1]
        )
        frame, _, _ = model.decode(state)
        reconstructed.append(frame[0])
    prior_frames, _ = model.predict_open_loop(observations[:, 0], actions)
    return (
        observations[0, 1:],
        torch.stack(reconstructed),
        prior_frames[0],
    )


def render_comparison(rows, out_path):
    titles = ("Ground truth", "Posterior reconstruction", "Prior open-loop (actions only)")
    if len(rows) != 3 or any(row.ndim != 4 for row in rows):
        raise ValueError("Expected three tensors of shape [T,3,H,W].")
    count = rows[0].shape[0]
    if not count or any(row.shape != rows[0].shape for row in rows):
        raise ValueError("All rows must have the same nonempty shape.")

    _, _, height, width = rows[0].shape
    header = 24
    canvas = Image.new("RGB", (count * width, 3 * (header + height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (title, frames) in enumerate(zip(titles, rows)):
        y = row_index * (header + height)
        draw.text((4, y + 4), title, fill="black")
        for t in range(count):
            canvas.paste(to_image(frames[t]), (t * width, y + header))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/episodes")
    parser.add_argument("--checkpoint", default="checkpoints/rssm_best.pt")
    parser.add_argument("--out", default="results/compare_reconstruction.png")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = model_from_checkpoint(checkpoint).to(device)
    model.eval()

    files = [Path(args.data) / name for name in checkpoint["val_files"]]
    dataset = SequenceDataset(files, checkpoint["seq_len"], checkpoint["seq_len"])
    if not 0 <= args.index < len(dataset):
        parser.error(f"--index must be between 0 and {len(dataset) - 1}")

    sample = dataset[args.index]
    observations = sample["observations"].unsqueeze(0).to(device)
    actions = sample["actions"].unsqueeze(0).to(device)

    rows = compare_frames(model, observations, actions)
    path = render_comparison(rows, args.out)
    ground_truth, posterior, prior = rows
    posterior_mse = (posterior - ground_truth).square().mean().item()
    prior_mse = (prior - ground_truth).square().mean().item()

    print(f"Saved {path}")
    print(f"Validation window index: {args.index} (episode holdout from checkpoint)")
    print(f"Posterior reconstruction MSE: {posterior_mse:.6f}")
    print(f"Prior open-loop MSE:          {prior_mse:.6f}")
    print("These are single-window diagnostic scores, not dataset-level metrics.")


if __name__ == "__main__":
    main()
