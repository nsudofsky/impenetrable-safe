"""Step 1: fit one neuron with a ReLU to the V data from Anna's code, by gradient descent."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Data from my spreadsheet (only rows where x > 0; target = speed = -y)
x = np.array([6,8,10,13,14,14,14,12,11,9,8]+[7]*16
           + [7,9,10,12,14,14,18]+[19]*12
           + [7,8,9,11,13,16,16,24,30,36,43,47,42,33,25,22,23]+[25]*10
           + [7,7,8,10,13,16,20,20,32,48,48,53,61,66,61,51,38,29,29,30,31,31]+[30]*8
           + [7,9,9,15,18,23,29,37,44,53,63,73,73,60,42,30,23,19,17,18,24,25,25,23,23,23]+[24]*8
           + [8,12,17,22,27,33,38,42,48,49]+[49]*10, dtype=float)
y = np.array([-29,-29,-25,-25,-17,-17,-13.5,-10,-6.5,-3,0,1,1.5,1,.5,.5]+[0]*11
           + [-31,-28,-24.5,-21.5,-19,-16.5,-14,-14,-9,-7,-6,-6,-3.5,-3,-2,-1.5,-1,-1,-1]
           + [-21.5,-20.5,-19,-17.5,-16.5,-15,-14,-13,-12,-12,-10,-7,-4,-4]+[0]*13
           + [-26,-23.5,-21.5,-20.5,-19,-18,-16,-15,-13.5,-12,-10,-9,-8,-6,-4,-1]+[0]*14
           + [-37,-33,-30,-26,-23,-23,-18.5,-16,-16,-12,-10.5,-8.5,-6,-3,0,1,3,4,2,1,0]+[0]*13
           + [-32.5,-30.5,-28,-24.5,-21,-18,-15,-13,-11,-8.5,-7,-5,-5,-3,-3,-2,-2,-1,-1,-1], dtype=float)
Y = -y   # ReLU can't output negatives, so fit speed

# Standardize x so one learning rate works for both m and b
x_mean, x_std = x.mean(), x.std()
xs = (x - x_mean) / x_std

learning_rate, epochs = 0.3, 100
m, b = 0.0, 1.0
losses = []
for epoch in range(epochs):
    z = m * xs + b                  # forward: linear
    a = np.maximum(0, z)            # forward: ReLU
    losses.append(np.mean((Y - a) ** 2))
    relu_grad = (z > 0).astype(float)
    dm = -2 * np.mean((Y - a) * relu_grad * xs)   # backprop
    db = -2 * np.mean((Y - a) * relu_grad)
    m -= learning_rate * dm
    b -= learning_rate * db

m_real = m / x_std
b_real = b - m * x_mean / x_std
print(f"speed = ReLU({m_real:.4f} * x + {b_real:.4f})   final loss = {losses[-1]:.2f}   N = {len(x)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
x_line = np.linspace(0, 80, 200)
ax1.scatter(x, Y, s=15, alpha=0.6, label="data")
ax1.plot(x_line, np.maximum(0, m_real * x_line + b_real), "r", lw=2, label=f"ReLU({m_real:.3f}x + {b_real:.2f})")
ax1.set_xlabel("x"); ax1.set_ylabel("speed (−y)"); ax1.set_title("V data and the ReLU fit"); ax1.legend()
ax2.plot(losses)
ax2.set_xlabel("epoch"); ax2.set_ylabel("MSE loss"); ax2.set_title("Loss vs. epoch")
plt.tight_layout()
plt.savefig("01_vdata_fit_and_loss.png", dpi=150)
