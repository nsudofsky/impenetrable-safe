"""Figures for the Notion page and the movie, made from the real data and the trained net."""
import glob
import re
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "python"))
from fingernet_np import FingerNetNumpy  # noqa: E402

# ---------------------------------------------------------------- 02: training pictures, 6 per number
rows = []
for n in range(6):
    files = sorted(glob.glob(str(APP / "laptop" / "data" / str(n) / "*.jpg")))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, 6).astype(int)]
    row = np.hstack([cv2.copyMakeBorder(cv2.resize(cv2.imread(f), (200, 95)), 3, 3, 3, 3,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255)) for f in pick])
    label = np.full((101, 60, 3), 255, np.uint8)
    cv2.putText(label, str(n), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 200), 3)
    rows.append(np.hstack([label, row]))
cv2.imwrite("02_training_pictures.png", np.vstack(rows))
counts = {n: len(glob.glob(str(APP / "laptop" / "data" / str(n) / "*.jpg"))) for n in range(6)}
print("pictures per number:", counts, "total", sum(counts.values()))

# ---------------------------------------------------------------- 03: what each layer sees
net = FingerNetNumpy(APP / "python" / "finger_cnn.npz")
sample = sorted(glob.glob(str(APP / "laptop" / "data" / "3" / "*.jpg")))[-40]
img = cv2.imread(sample)
rgb = cv2.cvtColor(cv2.resize(img, (128, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
x = (rgb.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5) / 0.5
layers = []
for i, (w, b) in enumerate(net.convs):
    conv = np.maximum(net.conv(x, w, b), 0)
    x = net.pool(conv)
    layers.append((conv, x))
scores = net.fc_w @ x.reshape(-1) + net.fc_b
p = np.exp(scores - scores.max()); p /= p.sum()

fig = plt.figure(figsize=(14, 6.4))
gs = fig.add_gridspec(4, 8, hspace=0.45, wspace=0.12, height_ratios=[1.6, 1, 1, 1])
ax = fig.add_subplot(gs[0, 0:3]); ax.imshow(rgb); ax.set_title("1. input: 3 x 64 x 128 (color, height, width)"); ax.axis("off")
ax = fig.add_subplot(gs[0, 4:8]); ax.bar(range(6), p, color=["#bbb"] * 6); ax.bar(int(p.argmax()), p.max(), color="#d6283b")
ax.set_xticks(range(6)); ax.set_ylim(0, 1); ax.set_title(f"5. output: how sure for 0-5 fingers (says {p.argmax()})"); ax.set_xlabel("fingers")
names = ["2. conv 5x5, 16 filters + ReLU + pool -> 16 x 32 x 64",
         "3. conv 3x3, 32 filters + ReLU + pool -> 32 x 16 x 32",
         "4. conv 3x3, 64 filters + ReLU + pool -> 64 x 8 x 16"]
for r, ((conv, pooled), name) in enumerate(zip(layers, names), start=1):
    strongest = np.argsort(-pooled.reshape(pooled.shape[0], -1).mean(1))[:8]   # the 8 busiest filters
    for c, k in enumerate(strongest):
        ax = fig.add_subplot(gs[r, c]); ax.imshow(pooled[k], cmap="magma"); ax.axis("off")
        if c == 0:
            ax.set_title(name, loc="left", fontsize=11)
fig.suptitle("What the net sees: one of my training pictures going through each layer", fontsize=14)
plt.savefig("03_what_each_layer_sees.png", dpi=130, bbox_inches="tight")

# ---------------------------------------------------------------- 04: LED screen
src = (APP / "sketch" / "sketch.ino").read_text()
icons = re.findall(r'\{"([.#]{13})", "([.#]{13})", "([.#]{13})", "([.#]{13})",\s*//[^\n]*\n\s*"([.#]{13})", "([.#]{13})", "([.#]{13})", "([.#]{13})"\}', src)
DIG = {2: [7,1,7,4,7], 1: [2,6,2,2,7], 0: [7,5,5,5,7]}


def grid(rows):
    g = np.zeros((8, 13), bool)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            g[r, c] = ch == "#"
    return g


def progress(count):
    g = np.zeros((8, 13), bool)
    for k in range(3):
        for c in range(3):
            if k < count:
                g[1:6, 1 + 4 * k + c] = True
            else:
                g[6, 1 + 4 * k + c] = True
    return g


def countdown(n):
    g = np.zeros((8, 13), bool); g[0, :] = g[7, :] = g[:, 0] = g[:, 12] = True
    for k, d in enumerate([n // 10, n % 10]):
        for r in range(5):
            for c in range(3):
                if DIG[d][r] & (4 >> c):
                    g[1 + r, 3 + 4 * k + c] = True
    return g


panels = [("locked", grid(icons[0])), ("ready: _ _ _", progress(0)), ("2 numbers in", progress(2)),
          ("number read (flash)", grid(icons[2])), ("open", grid(icons[1])), ("wrong code", grid(icons[3])),
          ("lockout: 10 s", countdown(10))]
fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.3))
for ax, (name, g) in zip(axes, panels):
    ax.set_facecolor("#111")
    ys, xs = np.nonzero(g)
    ax.scatter(xs, ys, s=55, c="#ff4040")
    ax.scatter(*np.nonzero(~g)[::-1], s=55, c="#2a2a2a")
    ax.set_xlim(-1, 13); ax.set_ylim(8, -1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name, fontsize=10)
fig.suptitle("The UNO Q's 13 x 8 LED screen (never shows the numbers, it's a safe)", fontsize=12)
plt.savefig("04_led_screen.png", dpi=150, bbox_inches="tight", facecolor="white")
print("made figures")
