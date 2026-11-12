"""Train the finger-counting net on the pictures in laptop/data/<fingers>/.

    python3 laptop/train.py              # 40 epochs
    python3 laptop/train.py --epochs 60

Pass 1 trains on the older 80% and tests on the newest 20% (the score and plots). Pass 2 trains
again on every picture, for the best number of epochs from pass 1, so the newest corrections are
learned too. Saves that net to python/finger_cnn.pt (so deploying the app takes it to the board) and
plots to laptop/results/: loss and accuracy per epoch, and which numbers it mixes up.

The last 20% of each number's pictures (the newest ones) are kept out of training and used only
to test. Pictures saved in a burst look almost the same, so a random split would put near-copies
in both piles and make the accuracy look better than it is.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))
from fingernet import CLASSES, FingerNet, prepare  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
args = parser.parse_args()

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------- load the pictures
train_x, train_y, test_x, test_y = [], [], [], []
for n in range(CLASSES):
    files = sorted((HERE / "data" / str(n)).glob("*.jpg"))      # names are times, so this is oldest first
    cut = int(len(files) * 0.8)
    for i, f in enumerate(files):
        x = prepare(cv2.imread(str(f)))[0]
        (train_x if i < cut else test_x).append(x)
        (train_y if i < cut else test_y).append(n)
train_x, test_x = torch.stack(train_x), torch.stack(test_x)
train_y, test_y = torch.tensor(train_y), torch.tensor(test_y)
print(f"{len(train_y)} training pictures, {len(test_y)} test pictures")

# Make new training pictures every epoch by moving the old ones around (the hint: rotate and shift).
# Empty space is filled with the desk's average color instead of black.
# The changes work on pixels from 0 to 1, so undo prepare()'s -1..1 scaling first, then redo it.
desk = (train_x[train_y == 0].mean(dim=(0, 2, 3)) * 0.5 + 0.5).tolist()
augment = v2.Compose([
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=desk),
    v2.RandomHorizontalFlip(),                    # a left hand counts the same as a right hand
    v2.ColorJitter(brightness=0.3, contrast=0.3),
])

# ---------------------------------------------------------------- train
loss_fn = nn.CrossEntropyLoss()                   # picking a category, not predicting a number


def train(x_all, y_all, epochs, test=None):
    """Train a new net on (x_all, y_all). With test pictures, also score it after every epoch."""
    torch.manual_seed(0)
    net = FingerNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(y_all))
        total = 0.0
        for i in range(0, len(order), 32):        # batches of 32 pictures
            idx = order[i:i + 32]
            x = torch.stack([augment(img * 0.5 + 0.5) for img in x_all[idx]]) * 2 - 1
            loss = loss_fn(net(x), y_all[idx])
            opt.zero_grad()
            loss.backward()                        # back-propagation: how each weight affected the loss
            opt.step()                             # gradient descent: nudge every weight downhill
            total += loss.item() * len(idx)
        history["train_loss"].append(total / len(y_all))
        if test is None:
            print(f"  epoch {epoch:3d}: train loss {total / len(y_all):.3f}")
            continue
        net.eval()
        with torch.no_grad():
            scores = net(test[0])
        history["test_loss"].append(loss_fn(scores, test[1]).item())
        history["test_acc"].append((scores.argmax(1) == test[1]).float().mean().item())
        print(f"epoch {epoch:3d}: train loss {total / len(y_all):.3f}, "
              f"test loss {history['test_loss'][-1]:.3f}, test accuracy {history['test_acc'][-1]:.0%}")
    return net, history


# Pass 1: learn on the older 80%, test on the newest 20% -> an honest score, and how many epochs work best.
print(f"{sum(p.numel() for p in FingerNet().parameters()):,} weights to learn")
print("\nPass 1: train on 80%, test on the newest 20%")
net, history = train(train_x, train_y, args.epochs, test=(test_x, test_y))
acc = history["test_acc"]
smooth = [np.mean(acc[max(0, i - 2):i + 3]) for i in range(len(acc))]   # accuracy jumps around: average 5 epochs
best_epoch = int(np.argmax(smooth)) + 1
print(f"test accuracy: {acc[-1]:.0%} at the end, about {smooth[best_epoch - 1]:.0%} around epoch {best_epoch} (the best stretch)")

# Pass 2: the lock should learn from every picture, including the newest corrections.
print(f"\nPass 2: train on all {len(train_y) + len(test_y)} pictures for {best_epoch} epochs")
final, _ = train(torch.cat([train_x, test_x]), torch.cat([train_y, test_y]), best_epoch)

# ---------------------------------------------------------------- save
out = HERE.parent / "python"
torch.save(final.state_dict(), out / "finger_cnn.pt")
roi = json.loads((HERE / "roi.json").read_text())
(out / "finger_cnn.json").write_text(json.dumps(
    {"roi": roi, "test_accuracy": round(smooth[best_epoch - 1], 3), "epochs": best_epoch}, indent=2))
print(f"saved to python/finger_cnn.pt")
best_acc = smooth[best_epoch - 1]
net.eval()

# ---------------------------------------------------------------- plots for the Notion page
results = HERE / "results"
results.mkdir(exist_ok=True)
with torch.no_grad():
    guess = net(test_x).argmax(1)
confusion = np.zeros((CLASSES, CLASSES), int)
for t, g in zip(test_y.tolist(), guess.tolist()):
    confusion[t, g] += 1
print("\nrows = real number of fingers, columns = what the net said")
print("     " + " ".join(f"{g:3d}" for g in range(CLASSES)))
for t in range(CLASSES):
    print(f"  {t}: " + " ".join(f"{c:3d}" for c in confusion[t]))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
epochs = range(1, args.epochs + 1)
a.axvline(best_epoch, color="gray", linestyle=":")
a.plot(epochs, history["train_loss"], label="training pictures")
a.plot(epochs, history["test_loss"], label="test pictures")
a.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
a.legend()
b.plot(epochs, [100 * v for v in history["test_acc"]])
b.set(xlabel="epoch", ylabel="% correct", title="Test accuracy", ylim=(0, 105))
fig.tight_layout()
fig.savefig(results / "training.png", dpi=150)

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(confusion, cmap="Blues")
for t in range(CLASSES):
    for g in range(CLASSES):
        ax.text(g, t, confusion[t, g], ha="center", va="center",
                color="white" if confusion[t, g] > confusion.max() / 2 else "black")
ax.set(xticks=range(CLASSES), yticks=range(CLASSES), xlabel="net said", ylabel="real fingers",
       title=f"Test pictures, pass 1 end ({(guess == test_y).float().mean():.0%} correct)")
fig.tight_layout()
fig.savefig(results / "confusion.png", dpi=150)
print(f"plots saved to {results}")
