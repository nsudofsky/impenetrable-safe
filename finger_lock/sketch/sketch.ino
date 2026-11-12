// Finger lock, microcontroller side: reads the arcade button, moves the latch servo, draws on the LED matrix.
// The decisions (is the code right?) are made in python/main.py, which calls:
//   set_locked(bool)      move the latch
//   set_light(bool)       built-in LED (on while listening)
//   show(icon)            0 locked, 1 unlocked, 2 check mark, 3 X
//   progress(count)       how many numbers are in, as blocks ("# # _"), never the numbers: it's a safe
//   countdown(seconds)    lockout: the seconds left inside a border
//   flash(times)          flash a check mark and beep, e.g. when a number is accepted
//   beep(freq, ms)        a beep on the buzzer
#include "Arduino_RouterBridge.h"
#include <Arduino_LED_Matrix.h>
#include <Servo.h>

Servo latch;
const int BUTTON = 10;     // arcade button: D10 and GND (pressed = LOW)
const int SERVO_PIN = 9;   // servo signal wire
const int LOCKED = 0;      // angle when the lever is in the slot
const int UNLOCKED = 90;   // angle when the lever is clear of the slot
const int BUZZER = 6;      // optional piezo buzzer: + to D6, - to GND (nothing wired = silent)

// ---------------------------------------------------------------- LED matrix
const int COLS = 13, ROWS = 8;
const bool MIRROR_X = true;              // how the board sits in the safe: everything drawn below
const bool MIRROR_Y = true;              // (icons, blocks) reads the right way round from the front
Arduino_LED_Matrix matrix;
uint8_t frame[COLS * ROWS];

const char* ICONS[4][ROWS] = {
  {"....#####....", "...#.....#...", "...#.....#...", "..#########..",   // 0 locked
   "..####.####..", "..####.####..", "..#########..", "............."},
  {".......#####.", "......#.....#", "......#......", "..#########..",   // 1 unlocked
   "..####.####..", "..####.####..", "..#########..", "............."},
  {".............", "...........#.", "..........#..", ".........#...",   // 2 check mark
   ".#.......#...", "..#.....#....", "...#...#.....", "....#.#......"},
  {"...#.....#...", "....#...#....", ".....#.#.....", "......#......",   // 3 X
   ".....#.#.....", "....#...#....", "...#.....#...", "............."},
};
const uint8_t DIGITS[10][5] = {
  {7,5,5,5,7}, {2,6,2,2,7}, {7,1,7,4,7}, {7,1,7,1,7}, {5,5,7,1,1},
  {7,4,7,1,7}, {7,4,7,5,7}, {7,1,1,1,1}, {7,5,7,5,7}, {7,5,7,1,7},
};

volatile int icon = 0;                   // what loop() draws; -1 = progress blocks, -2 = lockout countdown
volatile int countdownLeft = 0;
volatile int codeCount = 0;
volatile bool dirty = true;
volatile int flashesLeft = 0;            // on/off steps still to do (2 per flash)
unsigned long nextFlash = 0;

void setPixel(int col, int row) {
  if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return;
  if (MIRROR_X) col = COLS - 1 - col;
  if (MIRROR_Y) row = ROWS - 1 - row;
  frame[row * COLS + col] = 255;
}

void drawIcon(int i) {
  memset(frame, 0, sizeof(frame));
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (ICONS[i][r][c] == '#') setPixel(c, r);
  matrix.draw(frame);
}

void drawProgress(int count) {          // three slots: a solid block per number entered, "_" for the rest
  memset(frame, 0, sizeof(frame));
  for (int k = 0; k < 3; k++) {
    int col0 = 1 + 4 * k;                // first number on the left as you look at the safe
    for (int c = 0; c < 3; c++) {
      if (k < count)
        for (int r = 1; r <= 5; r++) setPixel(col0 + c, r);
      else
        setPixel(col0 + c, 6);
    }
  }
  matrix.draw(frame);
}

void drawCountdown(int n) {              // lockout: a border around the seconds left
  memset(frame, 0, sizeof(frame));
  for (int c = 0; c < COLS; c++) { setPixel(c, 0); setPixel(c, ROWS - 1); }
  for (int r = 0; r < ROWS; r++) { setPixel(0, r); setPixel(COLS - 1, r); }
  n = constrain(n, 0, 99);
  int digits[2] = {n / 10, n % 10};
  int len = n >= 10 ? 2 : 1;
  int col0 = (COLS - (len * 4 - 1)) / 2;
  for (int k = 0; k < len; k++) {
    int d = digits[2 - len + k];
    for (int r = 0; r < 5; r++)
      for (int c = 0; c < 3; c++)
        if (DIGITS[d][r] & (4 >> c)) setPixel(col0 + 4 * k + c, 1 + r);
  }
  matrix.draw(frame);
}

// ---------------------------------------------------------------- called from Python
void set_locked(bool locked) {
  latch.write(locked ? LOCKED : UNLOCKED);
}

void set_light(bool on) {
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH);   // LOW means the LED is ON
}

void show(int i) {
  if (i < 0 || i > 3) return;
  icon = i;
  dirty = true;
}

void flash(int times) {
  flashesLeft = 2 * constrain(times, 1, 5);
  nextFlash = 0;
}

void beep(int freq, int ms) {
  tone(BUZZER, freq, ms);
}

void countdown(int seconds) {
  icon = -2;
  countdownLeft = seconds;
  dirty = true;
}

void progress(int count) {
  icon = -1;
  codeCount = constrain(count, 0, 3);
  dirty = true;
}

// ---------------------------------------------------------------- main
void setup() {
  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);
  set_light(false);
  latch.attach(SERVO_PIN);
  latch.write(LOCKED);
  matrix.begin();
  matrix.setGrayscaleBits(8);

  Bridge.begin();
  Bridge.provide("set_locked", set_locked);
  Bridge.provide("set_light", set_light);
  Bridge.provide("show", show);
  Bridge.provide("progress", progress);
  Bridge.provide("countdown", countdown);
  Bridge.provide("flash", flash);
  Bridge.provide("beep", beep);
}

void loop() {
  if (flashesLeft > 0 && millis() >= nextFlash) {   // flash: check mark, then back to the picture
    if (flashesLeft % 2 == 0) {
      drawIcon(2);
      tone(BUZZER, 2000, 80);
    } else {
      dirty = true;
    }
    flashesLeft--;
    nextFlash = millis() + (flashesLeft % 2 == 1 ? 300 : 150);   // check mark 0.3 s, gap 0.15 s
  }
  if (dirty && flashesLeft % 2 == 0) {                           // draw here, not inside the Bridge callbacks
    dirty = false;
    if (icon >= 0) drawIcon(icon);
    else if (icon == -2) drawCountdown(countdownLeft);
    else drawProgress(codeCount);
  }
  if (digitalRead(BUTTON) == LOW) {
    delay(30);                           // ignore switch bounce
    if (digitalRead(BUTTON) == LOW) {
      Bridge.notify("button", 1);
      while (digitalRead(BUTTON) == LOW) {}   // wait for release so one press = one event
      delay(30);
    }
  }
}
