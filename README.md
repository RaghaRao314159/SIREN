# SIREN birthday greeting

Two independent versions live here. The original code, data, and completed
training outputs are preserved in `greyscale/`. The full-colour version in
`colour/` learns three RGB values at each coordinate from the supplied reference.
Both use the shared `.venv/` at the repository root.

## Colour training

Run from the repository root:

```sh
.venv/bin/python colour/prepare_target.py
.venv/bin/python colour/train.py
```

The preparation step keeps the whole reference and its aspect ratio, producing
a 640 × 316 RGB target. Training defaults to a 256-wide SIREN, 6,000 updates,
and 16,384 randomly sampled pixels per update. Sampling bounds update memory;
the final reported MSE is evaluated on the entire target. The video and saved
predictions also render the entire image.

Outputs are saved in `colour/outputs/`: `prediction.png`, `training.mp4`,
`model.pt`, `target.png`, `initial.png`, `loss.json`, `loss_curve.png`, and
the captured frames. RGB checkpoints use a three-channel output layer and
are separate from the original grayscale checkpoints.

Video timing defaults to `--frame-schedule log`: the first three seconds show
approximately the first 78 updates of a 6,000-step run, then the remaining
three seconds accelerate through the later fit, followed by a one-second final
hold. To use the earlier cubic timing, pass
`--frame-schedule power --frame-power 3`. The previous cubic run is retained in
`colour/outputs_power3/`.

For a larger training target or longer fit:

```sh
.venv/bin/python colour/prepare_target.py --width 960
.venv/bin/python colour/train.py --steps 12000 --output colour/outputs_960
```

`--render-scale 2` evaluates the learned field on a denser grid for the final PNG;
fine detail still depends on the training target and fit. `--no-video` skips
frame capture and MP4 export. Each run should use a new `--output` directory
if you want to retain an earlier colour run.

## Original grayscale version

Its source and existing outputs have been moved without modification. Run its
commands from inside that directory so its original relative paths resolve:

```sh
cd greyscale
../.venv/bin/python prepare_target.py
../.venv/bin/python train.py --output outputs_new
```

The original completed run is in `greyscale/outputs/`.

## One-page model summary

The equations and architecture diagram are in
`output/pdf/siren_colour_function_one_page.pdf`.
To rebuild from its LaTeX source with `pdflatex` and TikZ installed:

```sh
cd output/pdf
pdflatex -interaction=nonstopmode -halt-on-error siren_colour_function_one_page.tex
```
