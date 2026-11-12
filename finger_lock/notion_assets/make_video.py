"""A captioned slideshow to record my voice-over on. Each slide: a heading, a picture, and the
points to talk about. A thin bar along the bottom shows how long is left on the slide."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 10
FONT = "/System/Library/Fonts/Helvetica.ttc"
title_font, text_font, small_font = (ImageFont.truetype(FONT, s) for s in (58, 34, 26))

S = []   # (seconds, heading, [pictures], [talking points])
add = lambda sec, head, pics, points: S.append((sec, head, pics, points))

add(6, "Finger Lock", ["04_led_screen.png"], ["Press the button, show 3 numbers with your fingers, it opens or stays locked."])
add(8, "How it fits together", ["00_system_diagram.png"], ["Camera -> UNO Q runs the net, the code check and the web page -> servo and LEDs."])
add(8, "Step 1: one neuron + ReLU", ["01_vdata_fit_and_loss.png"], ["Gradient descent on the V data: loss 150 -> 93 in 5 steps."])
add(10, "How the CNN works", ["03_what_each_layer_sees.png"], ["3 conv + ReLU + pool layers find edges, then fingers. A linear layer gives 6 scores."])
add(8, "Training (and a bug)", ["05_training_buggy_augmentation.png", "07_training_final.png"], ["My augmentation was wrecking the pictures. Fixing it took the test accuracy from ~50% to ~68%."])
add(8, "What it mixes up", ["08_confusion_final.png"], ["1 and 5 are easy. 2, 3 and 4 get confused with their neighbors."])
add(10, "The safe's web page", ["09_web_page_states.png"], ["Show a number, it reads for ~3 s, check mark, hand out. Wrong 3 times = lockout. Right = open."])
add(8, "UNO Q vs my computer", ["00_system_diagram.png"], ["Trained in PyTorch on the laptop, runs in numpy on the board with the same answers."])
add(4, "Thanks!", ["04_led_screen.png"], ["Tasha Sudofsky, ME193"])


def fit(img, w, h):
    s = min(w / img.width, h / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


writer = cv2.VideoWriter("finger_lock_voiceover.mp4", cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
total = 0
for sec, head, pics, points in S:
    slide = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(slide)
    d.rectangle([0, 0, W, 110], fill=(214, 40, 59))
    d.text((60, 26), head, font=title_font, fill="white")
    text_h = 70 + 48 * len(points)
    area_top, area_bottom = 140, H - text_h - 30
    if pics:
        imgs = [Image.open(p).convert("RGB") for p in pics]
        slot = (W - 120 - 40 * (len(imgs) - 1)) // len(imgs)
        fitted = [fit(i, slot, area_bottom - area_top) for i in imgs]
        x = (W - sum(f.width for f in fitted) - 40 * (len(fitted) - 1)) // 2
        for f in fitted:
            slide.paste(f, (x, area_top + (area_bottom - area_top - f.height) // 2))
            x += f.width + 40
    y = H - text_h if pics else 220
    for p in points:
        d.text((80, y), "•  " + p, font=text_font if pics else ImageFont.truetype(FONT, 44), fill=(30, 30, 30))
        y += 48 if pics else 90
    base = cv2.cvtColor(np.array(slide), cv2.COLOR_RGB2BGR)
    for i in range(int(sec * FPS)):
        frame = base.copy()
        cv2.rectangle(frame, (0, H - 8), (int(W * (i + 1) / (sec * FPS)), H), (59, 40, 214), -1)
        writer.write(frame)
    total += sec
writer.release()
print(f"{len(S)} slides, {total} s")
