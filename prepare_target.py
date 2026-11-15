from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH, HEIGHT = 320, 160
SCALE = 4
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

image = Image.new("L", (WIDTH * SCALE, HEIGHT * SCALE), 0)
draw = ImageDraw.Draw(image)


def centered_text(
    text, font_file, size_pct, center_y_pct, center_x_pct=50, max_width_pct=86
):
    """Position by canvas percentages; font size is a percentage of height."""
    size = max(1, round(image.height * size_pct / 100))
    center_x = image.width * center_x_pct / 100
    center_y = image.height * center_y_pct / 100
    max_width = min(
        image.width * max_width_pct / 100,
        2 * min(center_x, image.width - center_x),
    )
    max_height = 2 * min(center_y, image.height - center_y)

    # Shrink long text automatically to keep it inside the canvas margins.
    while True:
        font = ImageFont.truetype(str(FONT_DIR / font_file), size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if (right - left <= max_width and bottom - top <= max_height) or size == 1:
            break
        size -= 1

    # Center the visible letters, including italic overhangs.
    x = center_x - (right - left) / 2 - left
    y = center_y - (bottom - top) / 2 - top
    draw.text((x, y), text, font=font, fill=255)


# These percentages stay the same when WIDTH and HEIGHT change.
centered_text("Happy Birthday", "Georgia Italic.ttf", size_pct=11, center_y_pct=32)
centered_text("Sankarshana", "Georgia.ttf", size_pct=20, center_y_pct=60)

output = Path("data")
output.mkdir(exist_ok=True)

# Keep a crisp, larger reference for reviewing the typography.
image.save(output / "greeting_reference.png")

# Generate the training target with smooth, slightly softened edges.
target = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
target = target.filter(ImageFilter.GaussianBlur(radius=HEIGHT * 0.0025))
target.save(output / "target.png")

print("Created data/greeting_reference.png and data/target.png")
