"""Fit the greeting image and save the learned continuous representation."""

import argparse
from contextlib import ExitStack
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from model import Siren, coordinate_grid
from plot_loss import save_loss_curve


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
    pixels = (values.reshape(height, width).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("data/target.png"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-scale", type=positive_int, default=1)
    parser.add_argument("--no-video", action="store_true", help="Skip PNG frame capture and MP4 export")
    parser.add_argument("--frame-power", type=float, default=3.0,
                        help="Frame timing exponent: 1=uniform, 2=quadratic, 3=stronger early emphasis (default)")
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error("--lr must be positive")
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
        target_image = source.convert("L")
    width, height = target_image.size
    targets = torch.from_numpy(np.array(target_image, dtype=np.float32) / 255).reshape(-1, 1).to(device)
    coords = coordinate_grid(height, width, device)
    model = Siren(width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
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
            ))
            video.capture(0, initial_image)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(coords), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            history.append({"step": step - 1, "mse": loss.item()})
            if video is not None:
                video.capture(step)
            if step == 1 or step % 50 == 0 or step == args.steps:
                print(f"Step {step:4d}/{args.steps} | MSE before update: {loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(coords), targets).item()
    history.append({"step": args.steps, "mse": final_loss})
    render(model, height * args.render_scale, width * args.render_scale, device).save(args.output / "prediction.png")
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "model_config": {"width": args.width, "omega0": 30.0},
        "image_size": {"height": height, "width": width},
        "coordinate_convention": "pixel centers, x in [-aspect, aspect], y up",
        "steps": args.steps,
        "seed": args.seed,
        "learning_rate": args.lr,
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
