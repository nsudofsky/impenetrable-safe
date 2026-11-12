"""00: how the whole lock fits together."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=color, ec="#333", lw=1.4))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=13, weight="bold")
    ax.text(x + w / 2, y + h - 0.85, "\n".join(lines), ha="center", va="top", fontsize=10.5, linespacing=1.5)


def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.8, color="#333"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", fontsize=9.5, color="#333")


box(0.2, 3.3, 2.8, 2.6, "Webcam + stand", ["USB webcam on the hub", "fixed height over", "the red square"], "#fde2e2")
box(4.0, 2.2, 5.0, 3.8, "UNO Q: Linux side (Python)", ["grabs camera pictures", "crops the red square", "neural net (numpy): 0-5 fingers",
                                                         "votes over ~3 s per number", "checks the 3-number code",
                                                         "lockout after 3 wrong codes", "web page on port 8000"], "#e3eefc")
box(10.0, 3.3, 3.8, 2.6, "UNO Q: microcontroller", ["arcade button (D10)", "servo latch (D9)", "13x8 LED screen", "buzzer (D6)"], "#e5f5e5")
box(4.0, 0.15, 5.0, 1.6, "Laptop (training only)", ["collect pictures -> train PyTorch CNN", "-> export weights to the board"], "#f2f2f2")
box(10.0, 0.6, 3.8, 2.0, "Web page", ["status, camera, confidence", "start / lock, change code", "who's testing"], "#fff4d6")
arrow(3.0, 4.6, 4.0, 4.6, "pictures")
arrow(9.0, 4.9, 10.0, 4.9, "Bridge: servo,")
ax.text(9.5, 4.55, "screen, beep", ha="center", fontsize=9.5, color="#333")
arrow(10.0, 4.0, 9.0, 4.0, "button press")
arrow(9.0, 2.6, 10.0, 1.9, "Wi-Fi")
arrow(6.5, 1.75, 6.5, 2.2, "")
ax.text(6.65, 1.93, "finger_cnn.npz", fontsize=9.5, color="#333")
ax.set_title("Finger Lock: how it fits together", fontsize=16, weight="bold")
plt.savefig("00_system_diagram.png", dpi=150, bbox_inches="tight")
