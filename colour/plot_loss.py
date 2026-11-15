"""Plot a saved training history without running training again."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_loss_curve(history, output):
    steps = [entry["step"] for entry in history]
    losses = [entry["mse"] for entry in history]
    figure = Figure(figsize=(10, 4.5), dpi=180, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.suptitle("RGB SIREN training loss", fontsize=17, fontweight="bold")
    for axis, title in zip(axes, ("Linear scale", "Logarithmic scale")):
        axis.plot(steps, losses, color="#2563eb", linewidth=1.8)
        axis.set(title=title, xlabel="Training step", ylabel="Mean squared error (MSE)")
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(steps[0], max(steps[-1], steps[0] + 1))
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    figure.supxlabel(
        f"Initial batch MSE: {losses[0]:.6f}    |    Final full-image MSE: {losses[-1]:.6f}",
        fontsize=10,
    )
    figure.savefig(output, facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/loss.json")
    args = parser.parse_args()
    output = args.history.parent / "loss_curve.png"
    save_loss_curve(json.loads(args.history.read_text()), output)
    print(f"Saved {output}")
