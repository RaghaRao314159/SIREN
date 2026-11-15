"""Capture scheduled model predictions and stream them to an MP4."""

from collections import Counter
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class TrainingVideo:
    def __init__(self, output, steps, render_frame, frame_count=180, fps=30, power=3.0):
        self.output = output
        self.render_frame = render_frame
        self.fps = fps
        self.power = power
        # Include step zero and the final trained model; retain duplicates so
        # even short runs still produce exactly frame_count animation frames.
        self.schedule = Counter(
            int(steps * (k / (frame_count - 1)) ** power)
            for k in range(frame_count)
        )
        self.frames_dir = output / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = []
        self.writer = None
        self.last_frame = None

    def __enter__(self):
        self.writer = imageio.get_writer(
            str(self.output / "training.mp4"), fps=self.fps,
            codec="libx264", quality=None, macro_block_size=1,
            pixelformat="yuv420p", output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        return self

    def capture(self, step, image=None):
        count = self.schedule.get(step, 0)
        if not count:
            return
        if image is None:
            image = self.render_frame()
        # H.264 needs even dimensions. Pad odd canvases without resizing glyphs.
        rgb = image.convert("RGB")
        frame = Image.new("RGB", (rgb.width + rgb.width % 2, rgb.height + rgb.height % 2))
        frame.paste(rgb, (0, 0))
        self.last_frame = np.asarray(frame)
        for _ in range(count):
            self._append(self.last_frame, step, hold=False)

    def _append(self, frame, step, hold):
        filename = f"frame_{len(self.manifest):04d}.png"
        Image.fromarray(frame).save(self.frames_dir / filename)
        self.writer.append_data(frame)
        self.manifest.append({"file": filename, "step": step, "hold": hold})

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None and self.last_frame is not None:
                final_step = self.manifest[-1]["step"]
                for _ in range(self.fps):
                    self._append(self.last_frame, final_step, hold=True)
        finally:
            self.writer.close()
            (self.output / "frames.json").write_text(json.dumps({
                "fps": self.fps,
                "schedule": "power",
                "power": self.power,
                "frame_count": len(self.manifest),
                "complete": exc_type is None,
                "frames": self.manifest,
            }, indent=2) + "\n")
