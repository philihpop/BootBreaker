# vision.md — CV Pipeline Skill

Skill doc for implementing `vision/`. This module converts raw screen frames
into the structured game state dict consumed by the agent.

## Responsibilities

- Capture screen frames from the game region
- Parse the brick grid (type per cell)
- Track ball position and estimate velocity
- Track paddle position
- Output one game state dict per frame

## capture.py

Use `mss` for screen capture. It is external, fast, and returns numpy arrays.

```python
import mss
import numpy as np

class ScreenCapture:
    def __init__(self, region: dict):
        # region = {"left": int, "top": int, "width": int, "height": int}
        self.region = region
        self.sct = mss.mss()

    def grab(self) -> np.ndarray:
        # Returns BGR numpy array (OpenCV convention)
        raw = self.sct.grab(self.region)
        return np.array(raw)[..., :3]  # drop alpha channel
```

Grab at the main loop tick rate (~30fps). Do not grab more frequently than
the agent inference rate — it wastes CPU.

## calibrate.py

Interactive one-time setup. Must produce HSV threshold ranges for each brick
type and the pixel bounds of the game region.

**Game region selection:**
Use `cv2.selectROI` on a full-screen capture. The user drags a rectangle
over the game window. Write `left, top, width, height` to config.yaml.

**Color sampling per brick type:**
For each brick type (broken, full, diamond, golden, boot):
1. Display the captured game region
2. Prompt user to click on an example brick of that type
3. Sample a 5×5 pixel patch around the click point
4. Convert to HSV, record mean ± 2*std for H, S, V
5. Write as threshold range to config.yaml

HSV is preferred over BGR because hue is lighting-invariant and the Dark
Carnival minigame renders under consistent synthetic lighting.

## brick_parser.py

Slices the game region into a regular grid and classifies each cell.

```python
class BrickParser:
    def __init__(self, grid_rows, grid_cols, hsv_thresholds):
        self.rows = grid_rows
        self.cols = grid_cols
        self.thresholds = hsv_thresholds  # loaded from config

    def parse(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Returns int8 array of shape (rows, cols)
        # Values: 0=empty 1=broken 2=full 3=diamond 4=golden 5=boot
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        cell_h = frame_bgr.shape[0] // self.rows
        cell_w = frame_bgr.shape[1] // self.cols

        for r in range(self.rows):
            for c in range(self.cols):
                cell = hsv[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                grid[r, c] = self._classify_cell(cell)
        return grid

    def _classify_cell(self, cell_hsv):
        # Sample center patch to avoid border artifacts
        h, w = cell_hsv.shape[:2]
        patch = cell_hsv[h//4:3*h//4, w//4:3*w//4]
        mean_hsv = patch.mean(axis=(0,1))
        # Check each brick type threshold in priority order
        # Return 0 if no threshold matches (empty cell)
        ...
```

**Classification priority order:** diamond → golden → boot → full → broken → empty.
Check diamond first because it must never be misclassified (permanent obstacle).
Check golden second because the +50 reward depends on correct identification.

**Cell border exclusion:** When sampling each cell, crop 20–25% from each edge.
Brick borders/gaps between cells will contaminate the color sample otherwise.

**Empty cell detection:** If mean saturation of the cell patch is below a low
threshold (e.g., S < 20), classify as empty regardless of hue. Empty cells
are typically dark/black background.

## tracker.py

Tracks ball and paddle using frame differencing and contour detection.

### Ball tracking

```python
class BallTracker:
    def __init__(self, history_len=8):
        self.prev_frame = None
        self.history = []  # list of (x, y) positions

    def update(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return None, (0.0, 0.0)

        # Frame difference isolates moving elements
        diff = cv2.absdiff(gray, self.prev_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find ball contour (small, roughly circular)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        ball_pos = self._find_ball_contour(contours)

        self.prev_frame = gray
        if ball_pos:
            self.history.append(ball_pos)
            if len(self.history) > self.history_len:
                self.history.pop(0)

        velocity = self._estimate_velocity()
        return ball_pos, velocity

    def _find_ball_contour(self, contours):
        # Filter by area and circularity
        # Ball is small (area 50–300px typically) and roughly round
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < 30 or area > 500:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.6:
                M = cv2.moments(c)
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy)
        return None

    def _estimate_velocity(self):
        if len(self.history) < 3:
            return (0.0, 0.0)
        # Linear regression over position history for smoothed velocity
        xs = [p[0] for p in self.history]
        ys = [p[1] for p in self.history]
        t = np.arange(len(xs), dtype=float)
        vx = np.polyfit(t, xs, 1)[0]
        vy = np.polyfit(t, ys, 1)[0]
        return (vx, vy)
```

**Velocity via linear regression:** Fitting a line over the last N positions
gives a smoothed velocity that is more stable than raw frame-to-frame delta.
N=8 is a good default. Reduce if the ball moves very fast (many pixels/frame).

**Intercept prediction:** To compute where the ball will be at paddle height,
use the velocity vector to project forward:

```python
def predict_intercept(ball_pos, ball_vel, paddle_y, frame_height):
    # How many frames until ball reaches paddle y?
    dy = paddle_y - ball_pos[1]
    if ball_vel[1] == 0 or (dy > 0 and ball_vel[1] < 0):
        return ball_pos[0]  # ball moving away, hold position
    frames_to_paddle = dy / ball_vel[1]
    predicted_x = ball_pos[0] + ball_vel[0] * frames_to_paddle
    # Clamp to game width with wall bounce reflection
    # Simple modular bounce: reflect off walls
    game_width = frame_height  # adjust to actual width
    predicted_x = predicted_x % (2 * game_width)
    if predicted_x > game_width:
        predicted_x = 2 * game_width - predicted_x
    return predicted_x
```

### Paddle tracking

The paddle is a horizontal rectangle near the bottom of the frame. Detect it
by masking the bottom 15% of the frame and finding the largest horizontal
contour:

```python
class PaddleTracker:
    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        bottom_strip = frame_bgr[int(h * 0.85):, :]
        gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return w / 2  # fallback to center
        # Paddle is the widest contour
        paddle = max(contours, key=lambda c: cv2.boundingRect(c)[2])
        x, y, bw, bh = cv2.boundingRect(paddle)
        return x + bw / 2  # return center x
```

## Output assembly

Each frame, assemble the state dict:

```python
def get_state(capture, brick_parser, ball_tracker, paddle_tracker):
    frame = capture.grab()
    brick_grid = brick_parser.parse(frame)
    ball_pos, ball_vel = ball_tracker.update(frame)
    paddle_x = paddle_tracker.update(frame)

    return {
        "ball_pos": ball_pos or (0.0, 0.0),
        "ball_vel": ball_vel,
        "paddle_x": paddle_x,
        "brick_grid": brick_grid,
        "lives": None,   # parse from HUD if needed, else omit
        "score": None,
    }
```

Lives and score can be omitted for the initial demo — the agent doesn't
strictly need them to advise paddle position.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Ball not detected | Ball too small, under area threshold | Lower area min in `_find_ball_contour` |
| Ball position jitters | Frame diff picks up background | Raise absdiff threshold (25→35) |
| Wrong brick type | HSV thresholds miscalibrated | Re-run calibrate.py, re-sample |
| Paddle detected at wrong Y | Minigame has UI elements in bottom strip | Narrow the bottom strip mask |
| Diamond classified as another type | Priority order wrong | Always check diamond first |
