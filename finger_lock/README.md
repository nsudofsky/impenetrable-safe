# Finger Lock (ME193 midterm: the impenetrable safe)

Press the button, show 3 numbers (1-5) with your fingers under the camera, and the safe opens or stays
locked. A CNN built in PyTorch reads the fingers, trained on the laptop and run on the UNO Q.

## How it works

- **Camera:** USB webcam on the UNO Q's hub, on a stand over a red square.
- **UNO Q, Linux side** (`python/main.py`): grabs pictures, runs the net (`python/fingernet_np.py`, numpy),
  votes over ~3 s per number, checks the code, 10 s lockout after 3 wrong codes, and serves a web page on
  port 8000 (`python/web.html`).
- **UNO Q, microcontroller** (`sketch/sketch.ino`): arcade button (D10), servo latch (D9), 13x8 LED screen,
  buzzer (D6).
- **Laptop** (`laptop/`): collect pictures, train, export the weights to the board.

## The net (`python/fingernet.py`)

```
picture 3x64x128
  conv 5x5, 16 filters -> ReLU -> max-pool   16 x 32 x 64
  conv 3x3, 32 filters -> ReLU -> max-pool   32 x 16 x 32
  conv 3x3, 64 filters -> ReLU -> max-pool   64 x  8 x 16
  flatten -> linear -> 6 scores (0-5 fingers), cross-entropy loss
```

Training pictures are randomly rotated, shifted, flipped and brightened every epoch.

## Run it

```bash
source ~/AIRobotics/.venv/bin/activate
cd finger_lock
python3 laptop/collect.py            # take training pictures (keys 0-5)
python3 laptop/train.py --epochs 60  # train (PyTorch) -> python/finger_cnn.pt + plots in laptop/results/
python3 laptop/export.py             # weights -> python/finger_cnn.npz, checks numpy == PyTorch
cd .. && python3 tools/deploy.py finger_lock   # upload to the UNO Q
```

Then press the arcade button and open `http://<board ip>:8000`. `laptop/lock_app.py` runs the same lock
from the laptop instead (don't run both). `laptop/live.py` shows the net's guesses live.

## Files

| Path | What |
|---|---|
| `python/main.py` | the safe's brain on the board: camera, finger reader, code, lockout, web API |
| `python/fingernet.py` / `fingernet_np.py` | the CNN in PyTorch (training) and numpy (board) |
| `python/web.html` | the web page |
| `sketch/sketch.ino` | button, servo, LED screen, buzzer |
| `laptop/` | collect, train, export, live view, laptop lock app, test log |
| `laptop/data/` | 1,600+ labeled training pictures |
| `notion_assets/` | figures, the stop-motion movie and the scripts that make them |
