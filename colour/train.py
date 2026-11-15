"""Fit an RGB greeting and save the learned continuous representation."""

import argparse
from contextlib import ExitStack
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from model import Siren, coordinate_grid
from plot_loss import save_loss_curve

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


@torch.no_grad()
def render(model, height, width, device):
    # Chunk network evaluation to bound activation memory for larger exports.
    coords = coordinate_grid(height, width)
    values = torch.cat([
        model(chunk.to(device)).cpu() for chunk in coords.split(65536)
    ])
    pixels = (values.reshape(height, width, 3).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "data/target.png")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--steps", type=positive_int, default=6000)
    parser.add_argument("--width", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=16384,
                        help="Pixels sampled per update; capped at the image size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-scale", type=positive_int, default=1)
    parser.add_argument("--no-video", action="store_true", help="Skip PNG frame capture and MP4 export")
    parser.add_argument("--frame-schedule", choices=["log", "power"], default="log",
                        help="Video timing: log strongly emphasizes the earliest updates (default)")
    parser.add_argument("--frame-power", type=float, default=3.0,
                        help="Exponent used with --frame-schedule power (default: 3)")
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.frame_power) or args.frame_power < 1:
        parser.error("--frame-power must be finite and at least 1")
    if not args.image.is_file():
        parser.error(f"Missing {args.image}; run prepare_target.py first.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    with Image.open(args.image) as source:
        target_image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = target_image.size
    targets = torch.from_numpy(np.array(target_image, dtype=np.float32) / 255).reshape(-1, 3).to(device)
    coords = coordinate_grid(height, width, device)
    model = Siren(width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    batch_size = min(args.batch_size, len(coords))
    args.output.mkdir(parents=True, exist_ok=True)
    target_image.save(args.output / "target.png")

    print(f"Device: {device} | Image: {width} x {height} | Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    initial_image = render(model, height, width, device)
    initial_image.save(args.output / "initial.png")
    history = []
    started = time.perf_counter()
    with ExitStack() as stack:
        video = None
        if not args.no_video:
            from video import TrainingVideo
            video = stack.enter_context(TrainingVideo(
                args.output, args.steps,
                lambda: render(model, height, width, device),
                power=args.frame_power,
                schedule=args.frame_schedule,
            ))
            video.capture(0, initial_image)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if batch_size < len(coords):
                indices = torch.randint(len(coords), (batch_size,), device=device)
                batch_coords, batch_targets = coords[indices], targets[indices]
            else:
                batch_coords, batch_targets = coords, targets
            loss = loss_fn(model(batch_coords), batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            history.append({"step": step - 1, "mse": loss.item()})
            if video is not None:
                video.capture(step)
            if step == 1 or step % 50 == 0 or step == args.steps:
                print(f"Step {step:4d}/{args.steps} | Batch MSE before update: {loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        squared_error = 0.0
        for chunk, target_chunk in zip(coords.split(65536), targets.split(65536)):
            squared_error += (model(chunk) - target_chunk).square().sum().item()
        final_loss = squared_error / targets.numel()
    history.append({"step": args.steps, "mse": final_loss})
    render(model, height * args.render_scale, width * args.render_scale, device).save(args.output / "prediction.png")
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "model_config": {"width": args.width, "omega0": 30.0},
        "colour_mode": "RGB",
        "output_channels": 3,
        "image_size": {"height": height, "width": width},
        "coordinate_convention": "pixel centers, x in [-aspect, aspect], y up",
        "steps": args.steps,
        "seed": args.seed,
        "learning_rate": args.lr,
        "batch_size": batch_size,
        "training_loss": "sampled pixel RGB MSE before each update",
        "final_loss": "full image RGB MSE after training",
        "final_mse": final_loss,
    }, args.output / "model.pt")
    (args.output / "loss.json").write_text(json.dumps(history, indent=2) + "\n")
    save_loss_curve(history, args.output / "loss_curve.png")
    print(f"Final MSE: {final_loss:.6f} | Elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    print(f"Saved model.pt, prediction.png, initial.png, target.png, loss.json, and loss_curve.png in {args.output}/")
    if not args.no_video:
        print(f"Saved {args.output}/training.mp4 (7 seconds, 30 fps), frames/, and frames.json")


if __name__ == "__main__":
    main()
