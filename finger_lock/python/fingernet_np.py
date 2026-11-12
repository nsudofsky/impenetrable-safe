"""The finger-counting net, run with numpy on the UNO Q (PyTorch is too big to install there).

Same layers and weights as fingernet.py: the laptop trains with PyTorch, laptop/export.py saves
the weights to finger_cnn.npz, and this file does the forward pass by hand:

    picture 3x64x128 -> conv 5x5 (16) -> ReLU -> max-pool -> conv 3x3 (32) -> ReLU -> max-pool
                     -> conv 3x3 (64) -> ReLU -> max-pool -> flatten -> linear -> 6 scores
"""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IMG_H, IMG_W = 64, 128


class FingerNetNumpy:
    def __init__(self, path=Path(__file__).with_name("finger_cnn.npz")):
        w = np.load(path)
        self.convs = [(w[f"features.{i}.weight"], w[f"features.{i}.bias"]) for i in (0, 3, 6)]
        self.fc_w, self.fc_b = w["classify.2.weight"], w["classify.2.bias"]

    @staticmethod
    def conv(x, weight, bias):
        """x: channels x H x W. Slide every filter over the picture (zero padding keeps the size)."""
        k = weight.shape[-1]
        pad = k // 2
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
        windows = sliding_window_view(x, (k, k), axis=(1, 2))          # C x H x W x k x k
        return np.tensordot(weight, windows, axes=([1, 2, 3], [0, 3, 4])) + bias[:, None, None]

    @staticmethod
    def pool(x):
        """2x2 max-pool: keep the strongest of each 4 neighbours, halving height and width."""
        c, h, w = x.shape
        return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))

    def probs(self, bgr_crop):
        """A cropped camera picture (OpenCV BGR) -> how sure the net is of 0-5 fingers."""
        import cv2
        rgb = cv2.cvtColor(cv2.resize(bgr_crop, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5) / 0.5
        for weight, bias in self.convs:
            x = self.pool(np.maximum(self.conv(x, weight, bias), 0))
        scores = self.fc_w @ x.reshape(-1) + self.fc_b
        e = np.exp(scores - scores.max())                                # softmax
        return e / e.sum()


def find_red_square(img):
    """The area inside the biggest red outline in the picture, or None."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 60), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, hierarchy = cv2.findContours(red, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    hole = hierarchy[0][biggest][2]                   # the inside of the red frame, if it has one
    holes = [i for i in range(len(contours)) if hierarchy[0][i][3] == biggest]
    if holes:
        hole = max(holes, key=lambda i: cv2.contourArea(contours[i]))
        return cv2.boundingRect(contours[hole])
    x, y, bw, bh = cv2.boundingRect(contours[biggest])   # no inside found: just stay inside the outline
    m = min(bw, bh) // 10
    return x + m, y + m, bw - 2 * m, bh - 2 * m
