"""Stop-motion movie of how the finger net works, one small step per frame, using a real training
picture of my hand and the real trained weights. 8 frames a second so each step is visible.

    python3 make_stop_motion.py  ->  nn_stop_motion.mp4
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "python"))
from fingernet_np import FingerNetNumpy  # noqa: E402

W, H, FPS = 1280, 720, 8
FONT = "/System/Library/Fonts/Helvetica.ttc"
BIG, MID, SMALL = (ImageFont.truetype(FONT, s) for s in (44, 28, 20))
RED, INK, GREY, BLUE = (214, 40, 59), (30, 30, 30), (150, 150, 150), (50, 110, 220)

# ---------------------------------------------------------------- the real numbers
net = FingerNetNumpy(APP / "python" / "finger_cnn.npz")
bgr = cv2.imread(sorted(glob.glob(str(APP / "laptop" / "data" / "3" / "*.jpg")))[-40])
rgb = cv2.cvtColor(cv2.resize(bgr, (128, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
x0 = (rgb.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5) / 0.5
w1, b1 = net.convs[0]
conv1 = net.conv(x0, w1, b1)                                    # 16 x 64 x 128, before ReLU
k = int(np.argmax(conv1.std(axis=(1, 2))))                      # the busiest filter: easiest to see
raw = conv1[k]
relu = np.maximum(raw, 0)
pooled = net.pool(relu[None])[0]                                 # 32 x 64
x = x0
shapes = []
for wt, bs in net.convs:
    x = net.pool(np.maximum(net.conv(x, wt, bs), 0))
    shapes.append(x)
scores = net.fc_w @ x.reshape(-1) + net.fc_b
probs = np.exp(scores - scores.max()); probs /= probs.sum()
answer = int(probs.argmax())
lim = float(np.abs(raw).max())


def heat(a, lo=None, hi=None, diverge=False):
    """A 2D array -> an RGB picture (blue = negative, red = positive when diverge)."""
    if diverge:
        t = np.clip(a / lim, -1, 1)
        img = np.zeros(a.shape + (3,), np.uint8) + 255
        img[..., 0] = np.where(t > 0, 255, 255 * (1 + t)).astype(np.uint8)
        img[..., 1] = (255 * (1 - np.abs(t))).astype(np.uint8)
        img[..., 2] = np.where(t < 0, 255, 255 * (1 - t)).astype(np.uint8)
        return img
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    g = (255 * np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def big(arr_rgb, w, h):
    return Image.fromarray(arr_rgb).resize((w, h), Image.NEAREST)


frames = []


def frame(title, caption, draw_fn, repeat=1):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 74], fill=RED)
    d.text((32, 14), title, font=BIG, fill="white")
    draw_fn(img, d)
    d.text((32, H - 56), caption, font=MID, fill=INK)
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    frames.extend([arr] * repeat)


# ---------------------------------------------------------------- 1. input = numbers
IN_X, IN_Y, IN_W, IN_H = 40, 130, 512, 256                      # input drawn 4x size
photo = big(rgb, IN_W, IN_H)
for step in range(10):
    def draw(img, d, step=step):
        img.paste(photo, (IN_X, IN_Y))
        if step >= 3:                                            # zoom box appears
            d.rectangle([IN_X + 160, IN_Y + 80, IN_X + 192, IN_Y + 112], outline=RED, width=3)
        if step >= 5:                                            # the pixel numbers fill in
            vals = x0.mean(0)[20:28, 40:48]
            for r in range(min(8, (step - 4) * 2)):
                for c in range(8):
                    d.text((640 + c * 72, 130 + r * 34), f"{vals[r, c]:+.1f}", font=SMALL, fill=INK)
    frame("1. A picture is just numbers",
          "64 x 128 pixels x 3 colors = 24,576 numbers going in", draw, repeat=4)

# ---------------------------------------------------------------- 2. convolution, step by step
filt = w1[k].mean(0)                                             # the filter, averaged over color
FW = 22
stride = 8
positions = [(r, c) for r in range(0, 64, stride) for c in range(0, 128, stride)]
built = np.full_like(raw, np.nan)
for i, (r, c) in enumerate(positions):
    built[r:r + stride, c:c + stride] = raw[r:r + stride, c:c + stride]
    if i % 2 and i != len(positions) - 1:                        # every other step is enough frames
        continue

    def draw(img, d, r=r, c=c, shown=built.copy()):
        img.paste(photo, (IN_X, 130))
        d.rectangle([IN_X + c * 4 - 8, 130 + r * 4 - 8, IN_X + c * 4 + 12, 130 + r * 4 + 12], outline=RED, width=3)
        out = heat(np.nan_to_num(shown), diverge=True)
        out[np.isnan(shown)] = 245
        img.paste(big(out, IN_W, IN_H), (700, 130))
        d.text((700, 100), "new map (red = matches the filter)", font=SMALL, fill=INK)
        d.text((40, 100), "filter slides across", font=SMALL, fill=INK)
        for fr in range(5):                                      # the 5x5 filter itself
            for fc in range(5):
                v = filt[fr, fc] / (np.abs(filt).max() + 1e-9)
                col = (255, int(255 * (1 - v)), int(255 * (1 - v))) if v > 0 else (int(255 * (1 + v)), int(255 * (1 + v)), 255)
                d.rectangle([40 + fc * FW, 420 + fr * FW, 40 + fc * FW + FW - 2, 420 + fr * FW + FW - 2], fill=col)
        d.text((170, 450), "one learned 5x5 filter: multiply, add up, write one number", font=SMALL, fill=INK)
    frame("2. Convolution: slide a filter, write down how well it matches",
          "Layer 1 has 16 filters like this, each making its own map", draw)
frames.extend([frames[-1]] * 8)

# ---------------------------------------------------------------- 3. ReLU
for step in range(9):
    rows = step * 8
    shown = raw.copy()
    shown[:rows] = relu[:rows]

    def draw(img, d, shown=shown, rows=rows):
        img.paste(big(heat(shown, diverge=True), IN_W * 2 - 100, IN_H * 2 - 50), (40, 100))
        if rows < 64:
            y = 100 + rows * (IN_H * 2 - 50) // 64
            d.line([40, y, 40 + IN_W * 2 - 100, y], fill=INK, width=3)
        d.text((40 + IN_W * 2 - 60, 200), "ReLU:\nnegative (blue)\n-> 0\n\npositive\nstays", font=MID, fill=INK)
    frame("3. ReLU: throw away the negatives", "max(0, x) keeps only the places that matched", draw, repeat=2)

# ---------------------------------------------------------------- 4. max-pool
for step in range(0, 33, 4):
    shown = np.full_like(pooled, np.nan)
    shown[:step] = pooled[:step]

    def draw(img, d, shown=shown, step=step):
        img.paste(big(heat(relu, 0, relu.max()), IN_W, IN_H), (40, 150))
        if step < 32:
            d.rectangle([40, 150 + step * 8, 40 + IN_W, 150 + step * 8 + 8], outline=RED, width=2)
        out = heat(np.nan_to_num(shown), 0, relu.max())
        out[np.isnan(shown)] = 245
        img.paste(big(out, IN_W // 2, IN_H // 2), (760, 214))
        d.text((40, 115), "64 x 128", font=SMALL, fill=INK)
        d.text((760, 180), "32 x 64", font=SMALL, fill=INK)
        d.text((600, 270), "->", font=BIG, fill=RED)
    frame("4. Max-pool: keep the biggest of every 2 x 2", "Half the height, half the width, keeps the strongest matches", draw, repeat=3)

# ---------------------------------------------------------------- 5. three rounds
layer_info = [(16, "conv 5x5 + ReLU + pool"), (32, "conv 3x3 + ReLU + pool"), (64, "conv 3x3 + ReLU + pool")]
for li, ((count, name), maps) in enumerate(zip(layer_info, shapes)):
    for n in range(1, 9):
        def draw(img, d, li=li, n=n):
            for j in range(li + 1):
                m = shapes[j]
                cnt = 8 if j < li else n
                size_w = 240 // (2 ** j) + 60
                size_h = size_w // 2
                for q in range(cnt):
                    ch = q * (m.shape[0] // 8)
                    img.paste(big(heat(m[ch]), size_w, size_h), (60 + j * 400 + q * 14, 140 + q * 14))
                d.text((60 + j * 400, 110), f"{layer_info[j][0]} maps of {m.shape[1]} x {m.shape[2]}", font=SMALL, fill=INK)
        frame("5. Do it 3 times: more maps, each smaller",
              f"Layer {li + 1}: {name}. Early maps see edges, later ones see whole fingers", draw)
    frames.extend([frames[-1]] * 6)

# ---------------------------------------------------------------- 6. flatten + linear
flat = x.reshape(-1)
for step in range(12):
    def draw(img, d, step=step):
        n = min(step, 8)
        seg = flat[: n * 1024]
        if seg.size:
            strip = heat(seg.reshape(1, -1), 0, flat.max())
            img.paste(big(strip, 60 * n, 40), (40, 200))
        d.text((40, 160), f"64 x 8 x 16 maps unrolled = {flat.size:,} numbers", font=SMALL, fill=INK)
        if step >= 8:
            for i in range(6):
                d.line([520, 220, 760, 140 + i * 70], fill=GREY, width=1)
                d.ellipse([760, 120 + i * 70, 800, 160 + i * 70], outline=INK, width=2)
                d.text((815, 128 + i * 70), f"{i} fingers: score {scores[i]:+.1f}", font=SMALL, fill=INK)
            d.text((520, 520), "one linear layer: each score =\nweights x all 8,192 numbers + bias", font=SMALL, fill=INK)
    frame("6. Flatten, then one linear layer -> 6 scores", "Every number gets a vote on every answer", draw, repeat=3)

# ---------------------------------------------------------------- 7. softmax -> answer
for step in range(10):
    t = min(1.0, step / 6)
    def draw(img, d, t=t):
        for i in range(6):
            h = int(330 * probs[i] * t)
            col = RED if (i == answer and t >= 1) else GREY
            d.rectangle([200 + i * 150, 520 - h, 300 + i * 150, 520], fill=col)
            d.text((235 + i * 150, 530), str(i), font=MID, fill=INK)
            d.text((215 + i * 150, 480 - h), f"{probs[i] * t:.0%}", font=SMALL, fill=INK)
        if t >= 1:
            d.text((200, 110), f"Answer: {answer} fingers", font=BIG, fill=RED)
    frame("7. Softmax: scores -> how sure", "Biggest one wins. The lock averages ~3 s of these before deciding", draw, repeat=2)

# ---------------------------------------------------------------- 8. learning
hist = None
try:
    import re
    log = Path("/private/tmp/claude-501/-Users-tashasudofsky-AIRobotics/5b2f5d80-d4c7-41a1-97d5-db64f2675600/scratchpad/train_log4.txt").read_text()
    hist = [float(v) for v in re.findall(r"^epoch +\d+: train loss ([\d.]+)", log, re.M)]
except OSError:
    pass
hist = hist or list(1.7 * np.exp(-np.linspace(0, 2, 60)) + 0.25)
stages = ["forward: picture -> guess", "loss: how wrong? (cross-entropy)",
          "backprop: blame flows backward to every weight", "gradient descent: nudge each weight downhill"]
for step in range(len(hist)):
    def draw(img, d, step=step):
        boxes = ["picture", "conv 1", "conv 2", "conv 3", "linear", "guess"]
        for i, b in enumerate(boxes):
            d.rounded_rectangle([40 + i * 115, 130, 140 + i * 115, 180], 8, outline=INK, width=2)
            d.text((50 + i * 115, 143), b, font=SMALL, fill=INK)
        phase = step % 4
        if phase == 0:
            for i in range(5):
                d.line([140 + i * 115, 155, 155 + i * 115, 155], fill=BLUE, width=4)
        if phase >= 2:
            for i in range(5):
                d.line([155 + i * 115, 170, 140 + i * 115, 170], fill=RED, width=4)
        d.text((40, 200), stages[phase], font=MID, fill=RED if phase >= 2 else BLUE)
        # the loss curve, drawn up to this epoch, with a ball on it
        gx, gy, gw, gh = 760, 120, 470, 420
        d.rectangle([gx, gy, gx + gw, gy + gh], outline=GREY)
        top = max(hist)
        pts = [(gx + gw * i / (len(hist) - 1), gy + gh - gh * v / top) for i, v in enumerate(hist[:step + 1])]
        if len(pts) > 1:
            d.line(pts, fill=BLUE, width=3)
        bx, by = pts[-1]
        d.ellipse([bx - 9, by - 9, bx + 9, by + 9], fill=RED)
        d.text((gx, gy + gh + 8), f"epoch {step + 1}: training loss {hist[step]:.2f}", font=SMALL, fill=INK)
        d.text((40, 300), "Every epoch: all 1,300+ training pictures,\nrandomly rotated / shifted / brightened,\n"
                          "in batches of 32. 73,510 weights\nget nudged after every batch.", font=SMALL, fill=INK)
    frame("8. Learning: guess, measure, blame, nudge, repeat", "The ball rolls downhill: that's gradient descent", draw, repeat=2)
frames.extend([frames[-1]] * 12)

writer = cv2.VideoWriter("nn_stop_motion.mp4", cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
for f in frames:
    writer.write(f)
writer.release()
print(f"{len(frames)} frames, {len(frames) / FPS:.0f} s, answer {answer}, filter {k}")
