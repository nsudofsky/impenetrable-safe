"""Watch the trained net count your fingers live, using the webcam on the board.

    python3 laptop/live.py
    python3 laptop/live.py --board 10.1.2.3

Shows the camera with the box, the net's guess, and a bar for how sure it is of each answer 0-5.
r fits the box to the red square again (also done at the start), q quits.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))
from fingernet import CLASSES, FingerNet, find_red_square, prepare  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--board", default="10.247.137.100")
args = parser.parse_args()

net = FingerNet()
net.load_state_dict(torch.load(HERE.parent / "python" / "finger_cnn.pt"))
net.eval()
MODEL_INFO = HERE.parent / "python" / "finger_cnn.json"
roi = json.loads(MODEL_INFO.read_text())["roi"]
url = f"http://{args.board}:8000/frame.jpg"


def refit_box(img):
    """Fit the box inside the red square again (the camera got bumped). Saved for next time too."""
    found = find_red_square(img)
    if not found or found[2] < 150 or found[3] < 80:
        print("couldn't find the red square; is all of it in the picture?")
        return
    roi.update(x=int(found[0]), y=int(found[1]), w=int(found[2]), h=int(found[3]))
    info = json.loads(MODEL_INFO.read_text())
    info["roi"] = roi
    MODEL_INFO.write_text(json.dumps(info, indent=2))
    (HERE / "roi.json").write_text(json.dumps(roi))
    print("box fitted to the red square:", roi)


first = True

while True:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        print("no picture from the board, retrying...")
        continue
    if first:
        refit_box(img)
        first = False
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    with torch.no_grad():
        probs = torch.softmax(net(prepare(img[y:y + h, x:x + w])), dim=1)[0].numpy()
    guess = int(probs.argmax())

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{guess} fingers ({probs[guess]:.0%} sure)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    for n in range(CLASSES):                         # one bar per answer
        bx = 10 + n * 60
        cv2.rectangle(img, (bx, 470 - int(probs[n] * 100)), (bx + 40, 470),
                      (0, 255, 255) if n == guess else (200, 200, 200), -1)
        cv2.putText(img, str(n), (bx + 12, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("live", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        refit_box(img)

cv2.destroyAllWindows()
