"""Laptop side of the finger lock: waits for the safe's button, then sends it 3 numbers, one at a time.

For now you type the numbers. Later read_number() becomes: webcam -> neural net -> finger count.

    python3 laptop/safe_client.py [board-ip]
"""
import json
import sys
import time
import urllib.request

BOARD = sys.argv[1] if len(sys.argv) > 1 else "10.247.137.100"
URL = f"http://{BOARD}:8000"


def ask(path="/state", method="GET"):
    with urllib.request.urlopen(urllib.request.Request(URL + path, method=method), timeout=3) as r:
        return json.load(r)


def read_number(i):
    """Stand-in for the camera: type one finger count (1-5)."""
    while True:
        n = input(f"  number {i} of 3 (1-5): ").strip()
        if len(n) == 1 and n in "12345":
            return n


print(f"Connected to the safe at {URL}. Press the button on the safe to start.")
last = None
while True:
    mode = ask()["mode"]
    if mode != last:
        print("safe is", mode)
        last = mode
    if mode == "listening":
        for i in range(1, 4):
            result = ask(f"/digit?n={read_number(i)}", "POST")
        print("UNLOCKED! Press the button to lock it again." if result["ok"] else "Wrong code, still locked.")
        time.sleep(3)                    # let the safe show the check or X
    time.sleep(0.3)
