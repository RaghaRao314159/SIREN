"""Resize a reference for RGB training, preserving the entire composition."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path,
                        default=ROOT / "data/greeting_reference_colour.jpg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--output", type=Path, default=ROOT / "data/target.png")
    args = parser.parse_args()
    if args.width < 1:
        parser.error("--width must be positive")
    if not args.image.is_file():
        parser.error(f"Missing image: {args.image}")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        height = max(1, round(image.height * args.width / image.width))
        target = image.resize((args.width, height), Image.Resampling.LANCZOS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target.save(args.output)
    print(f"Saved {args.output} ({target.width} x {target.height}, RGB)")


if __name__ == "__main__":
    main()
