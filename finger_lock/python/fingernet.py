"""The finger-counting neural net, shared by the laptop (training) and the board (unlocking).

A picture of the red-square area goes in (3 x 64 x 128: color, height, width); 6 scores come out,
one for each answer 0-5 fingers. The biggest score is the guess.

    picture 3x64x128
      conv 5x5, 16 filters -> ReLU -> max-pool 2x2  ->  16 x 32 x 64   (edges, skin vs wood)
      conv 3x3, 32 filters -> ReLU -> max-pool 2x2  ->  32 x 16 x 32   (finger tips, gaps)
      conv 3x3, 64 filters -> ReLU -> max-pool 2x2  ->  64 x  8 x 16   (whole fingers)
      flatten (8192 numbers) -> linear -> 6 scores
"""
import torch
from torch import nn

IMG_H, IMG_W = 64, 128
CLASSES = 6


class FingerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),                     # randomly hide 30% while training so it can't memorize
            nn.Linear(64 * (IMG_H // 8) * (IMG_W // 8), CLASSES),
        )

    def forward(self, x):
        return self.classify(self.features(x))


def prepare(bgr_crop):
    """A cropped camera picture (OpenCV, BGR, any size) -> the 1 x 3 x 64 x 128 tensor the net expects."""
    import cv2
    rgb = cv2.cvtColor(cv2.resize(bgr_crop, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    return ((x - 0.5) / 0.5).unsqueeze(0)


from fingernet_np import find_red_square  # noqa: E402,F401  (shared with the board)
