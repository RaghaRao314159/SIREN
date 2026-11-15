# Project Plan: Mathematical Birthday Greeting via SIREN
**Target Message:** *"Happy Birthday Sankarshana"*  
**Core Concept:** Encoding a visual birthday greeting as a continuous implicit neural representation $f_\theta : [-1, 1]^2 \rightarrow [0, 1]$ using a Sinusoidal Representation Network (SIREN), capturing its gradient-descent optimization as an animated morph from interference waves into crisp calligraphy.

---

## 1. Mathematical & Architectural Specification

### 1.1 Domain & Target Function
* **Spatial Domain:** Normalize canvas pixel coordinates to $\mathbf{x} = (x, y)^T \in [-1, 1]^2$ (or aspect-adjusted $[-a, a] \times [-1, 1]$ where $a = \text{width} / \text{height}$ to prevent horizontal stretching of *"Sankarshana"*).
* **Target Signal:** Let $S \subset [-a, a] \times [-1, 1]$ represent the glyph strokes. Define the smoothed target intensity field $I(\mathbf{x}) \in [0, 1]$, applying a slight Gaussian blur ($\sigma \approx 0.8\text{ px}$) to give the gradient non-zero support near glyph boundaries $\partial S$.

### 1.2 Network Architecture
A 4-layer Multi-Layer Perceptron ($L = 4$) with sinusoidal activations:
$$\begin{aligned}
\mathbf{h}_1 &= \sin\!\big(\omega_0 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)\big) \\
\mathbf{h}_2 &= \sin\!\big(\omega (\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)\big) \\
\mathbf{h}_3 &= \sin\!\big(\omega (\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)\big) \\
f_\theta(\mathbf{x}) &= \sigma(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4)
\end{aligned}$$

* **Hidden Width ($d$):** $d = 128$ (fast training, $\approx 33\text{k}$ parameters) or $d = 256$ (ultra-crisp serif/cursive details, $\approx 132\text{k}$ parameters).
* **Frequency Scaling:** $\omega_0 = 30.0$ on the first layer to span multiple spatial cycles across the canvas; $\omega = 1.0$ (with scaled initialization) on hidden layers.
* **Output Activation:** Sigmoid $\sigma(z) = (1 + e^{-z})^{-1}$ or hard clamping $\min(\max(z, 0), 1)$ to bound brightness cleanly.

### 1.3 Spectral Initialization
Initialize weights so pre-activations follow an arcsine distribution, creating smooth water-ripple interference at step $t = 0$:
$$\mathbf{W}_1 \sim \mathcal{U}\!\left(-\frac{1}{2},\; \frac{1}{2}\right), \qquad \mathbf{W}_{i} \sim \mathcal{U}\!\left(-\frac{\sqrt{6/d}}{\omega},\; \frac{\sqrt{6/d}}{\omega}\right) \quad \text{for } i \in \{2, 3\}$$

### 1.4 Optimization Objective
Minimize Mean Squared Error (MSE) across all $N$ sampled coordinates:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{k=1}^{N} \big(f_\theta(\mathbf{x}_k) - I(\mathbf{x}_k)\big)^2$$
* **Optimizer:** Adam with learning rate $\eta = 1 \times 10^{-4}$ (or $2 \times 10^{-4}$ for faster convergence).

---

## 2. Execution Roadmap

### Phase 1: Canvas & Typography Setup
1. **Aspect Ratio & Layout:** Use a $2:1$ widescreen ratio (e.g., $640 \times 320$ training grid) so *"Sankarshana"* has generous horizontal breathing room.
   * **Line 1:** `Happy Birthday` (smaller, tracked-out sans-serif or italic serif)
   * **Line 2:** `Sankarshana` (larger, elegant calligraphy or high-contrast serif)
2. **Coordinate Grid Generation:** Build a meshgrid of $(x, y)$ pairs corresponding to every pixel center, flattened into an $(N, 2)$ tensor.

### Phase 2: Training & Non-Linear Frame Capture
Neural networks learn low frequencies rapidly in early epochs and refine fine edges slowly in later epochs. A uniform frame capture makes the beginning feel too abrupt and the end feel static.
1. **Total Steps:** Run $600$ optimization steps.
2. **Logarithmic / Eased Frame Sampling:** Capture snapshots according to a quadratic schedule $t_k = \lfloor 600 \cdot (k / K)^2 \rfloor$ for $K = 180$ total video frames (at $30\text{ fps}$, yielding a $6$-second animation):
   * **Frames $0\text{--}60$ ($0\text{--}2\text{s}$):** Captures steps $0\text{--}66$ in slow motion—showing the initial interference ripples bending into glowing clouds.
   * **Frames $60\text{--}140$ ($2\text{--}4.7\text{s}$):** Captures steps $67\text{--}360$—showing the letters pulling into sharp focus.
   * **Frames $140\text{--}180$ ($4.7\text{--}6\text{s}$):** Captures steps $361\text{--}600$ plus a $1$-second hold on the crisp final message.

### Phase 3: Visual Styling & Mathematical HUD
Transform the raw grayscale $[0, 1]$ scalar field into a polished visual experience:
1. **Custom Colormap:** Map scalar intensity $v = f_\theta(x, y)$ to an aesthetic color gradient rather than plain black-and-white:
   * *Option A (Celestial Gold):* Deep midnight indigo background ($v = 0$) $\rightarrow$ warm amber ($v = 0.5$) $\rightarrow$ luminous champagne gold ($v = 1.0$).
   * *Option B (Bioluminescent):* Obsidian background $\rightarrow$ deep teal $\rightarrow$ electric cyan/white core.
2. **Subtle Bloom / Glow:** Add a soft Gaussian bloom pass to pixels where $v > 0.7$ so the finished letters radiate gently.
3. **Mathematical Telemetry Overlay (Optional):** Render clean monospace telemetry in the bottom corner that updates per frame:
   * Current step $t$ and live loss $\mathcal{L}(\theta_t)$
   * The governing equation: $f_\theta(\mathbf{x}) = \mathbf{W}_4(\sin \circ \dots \circ \sin(\omega_0 \mathbf{W}_1 \mathbf{x}))$

### Phase 4: Infinite-Resolution Finale
Once step $600$ completes, evaluate the trained network $f_{\theta^*}(x, y)$ on an ultra-dense $3840 \times 1920$ ($4\text{K}$) coordinate grid. Because the representation is continuous, the final still image will exhibit zero pixelation.

---

## 3. Delivery Formats (Ways to Share the Gift)

| Format | Best For | Technical Approach |
| :--- | :--- | :--- |
| **1. High-Res Video (`MP4` / `GIF`)** | Sending directly via WhatsApp, Telegram, or email | Export the $180$ rendered frames via `ffmpeg` (`libx264`, `-crf 18`) with a gentle fade-out/loop. |
| **2. Interactive Web Page (`HTML`)** | Letting Sankarshana watch it train live or zoom in infinitely | Either embed the pre-trained weights in a lightweight WebGL fragment shader (for infinite interactive zoom) or run live training in JavaScript. |
| **3. Framed Print + QR Code** | A physical keepsake | Print the $4\text{K}$ render on matte archival paper with the equation printed underneath and a QR code linking to the training animation. |

---

## 4. Tech Stack & Dependencies

* **Python Pipeline (Recommended for highest visual quality & $4\text{K}$ export):**
  * `torch` (network definition, autograd, Adam optimizer)
  * `numpy` & `Pillow` (typography rasterization and coordinate grid creation)
  * `matplotlib` (custom perceptual colormaps)
  * `imageio[ffmpeg]` (exporting smooth $30\text{ fps}$ or $60\text{ fps}$ MP4 video)
* **Browser Pipeline (Zero-install shareable link):**
  * Single self-contained `.html` file using `HTML5 Canvas` + vanilla JavaScript matrix math (or `WebGL` shaders) to train and render live on any phone or laptop.