"""Save the trained net's weights for the UNO Q (python/finger_cnn.npz), and check that the numpy
version on the board gives the same answers as PyTorch.

    python3 laptop/export.py        (run after every train.py, then deploy)
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))
from fingernet import FingerNet, prepare  # noqa: E402
from fingernet_np import FingerNetNumpy  # noqa: E402

state = torch.load(HERE.parent / "python" / "finger_cnn.pt")
np.savez(HERE.parent / "python" / "finger_cnn.npz", **{k: v.numpy() for k, v in state.items()})

net = FingerNet()
net.load_state_dict(state)
net.eval()
npnet = FingerNetNumpy()
worst, same = 0.0, 0
files = sorted(glob.glob(str(HERE / "data" / "*" / "*.jpg")))[::40]
for f in files:
    img = cv2.imread(f)
    with torch.no_grad():
        p_torch = torch.softmax(net(prepare(img)), 1)[0].numpy()
    p_np = npnet.probs(img)
    worst = max(worst, float(np.abs(p_torch - p_np).max()))
    same += int(p_torch.argmax() == p_np.argmax())
print(f"saved python/finger_cnn.npz; numpy vs PyTorch on {len(files)} pictures: "
      f"same answer {same}/{len(files)}, biggest difference {worst:.6f}")
