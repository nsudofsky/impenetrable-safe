# Finger lock: in-class testing plan

Goal for class: run the lock with as many different hands as possible, write down what fails,
and save the pictures the net got wrong so the next training run fixes them.

## Before class (10 min)

1. **Power and Wi-Fi.** Hub powered from the laptop (or a strong 5V 3A power bank). UNO Q on the same Wi-Fi as the laptop.
   Check it answers:
   ```bash
   cd ~/AIRobotics/ArduinoApps
   python3 tools/deploy.py --cmd "hostname -I"
   ```
   If the IP changed, update `HOSTS` in `tools/board_secrets.py` and use `--board <ip>` below.
2. **Upload the latest app** (flash + beep on each accepted number):
   ```bash
   python3 tools/deploy.py finger_lock
   ```
3. **Optional buzzer.** Piezo buzzer + leg to **D6**, - leg to **GND**. Without one, the LED screen still flashes.
4. **Camera check.** Webcam on the hub, whole red square in view. Do not bump the camera stand.
   If it moved, run `python3 laptop/collect.py` and press **r** once.

## In class: run the lock

```bash
cd ~/AIRobotics/ArduinoApps/finger_lock
source ~/AIRobotics/.venv/bin/activate
python3 laptop/lock_app.py --tester "Tasha"
```

1. Press the arcade button. The screen shows `_ _ _`.
2. Show a number in the red square and move your hand a little while the green bar fills (8 pictures, ~1.5 s). The net's answers are averaged; the safe **flashes a check mark and beeps**, and a block (not the number) appears on the LED screen.
3. Take your hand out of the square, then show the next number.
4. Code **324**: check mark + high beep + lever opens. Wrong: X + low buzz.
5. Press the button again to lock.

Web page (phone or laptop, same Wi-Fi): **http://10.247.137.100:8000** shows the status, the camera,
how sure the net is (never the numbers), Start/Lock, and **Change the code** (only while the safe is open).
If the camera gets bumped, press **r** in the lock app window.

## Testing checklist

Run each test 3 times and mark it in the table. Use `--tester "<name>"` for each person so the log keeps track.

| Test | Who | Works 3/3? | What went wrong |
|---|---|---|---|
| Code 324, my hand, ring on | Tasha | | |
| Code 324, my hand, ring off | Tasha | | |
| Code 324, left hand | Tasha | | |
| Wrong code (e.g. 325) stays locked | Tasha | | |
| Code 324, classmate 1 | | | |
| Code 324, classmate 2 | | | |
| Code 324, classmate 3 | | | |
| Each number 1-5 shown alone | | | |
| Bright vs dim room / shadow over square | | | |
| Fingers spread vs pressed together | | | |

**When the net reads a number wrong:** keep the hand still, click the window, and press the key for the **right** number (0-5).
That saves the picture into `laptop/data/` so the next training run learns it.
Get classmates to do this too: other people's hands are the biggest gap in the training data.

Every attempt (numbers read, how sure the net was, open or wrong) goes into `laptop/test_log.csv`.
Use it for the limitations section.

## After class: retrain

```bash
python3 laptop/train.py
python3 laptop/live.py          # check the numbers that failed in class
```
Save a copy of `laptop/results/` first if you want the before/after plots.

## Notion page notes to grab while testing

- Video of the lock opening with the right code, and staying locked with a wrong one
- Which numbers fail most (look at `test_log.csv` and `results/confusion.png`)
- Limitations seen in class (lighting, other hands, ring, fingers together, camera bumped, Wi-Fi delay)
- Before/after training plots
