# Dota 2 Dark Carnival Breakout — Overlay Assistant

## Project overview

A visual overlay assistant for the Breakout-like minigame in Dota 2's Dark Carnival event.
The system watches the game screen, parses game state via CV, runs an RL agent to compute
the optimal paddle position, and renders a dot overlay on a transparent external window.
Human plays; agent advises. No input automation.

## Architecture

```
screen capture (mss)
       ↓
  vision pipeline          ← see vision.md
       ↓
  game state dict
       ↓
  RL agent inference       ← see agent.md
       ↓
  target paddle X
       ↓
  overlay renderer
```

## Project structure

```
breakout-overlay/
├── CLAUDE.md
├── vision.md
├── agent.md
├── requirements.txt
│
├── vision/
│   ├── capture.py          # screen region capture via mss
│   ├── brick_parser.py     # grid slicing + brick type classification
│   ├── tracker.py          # ball and paddle tracking
│   └── calibrate.py        # interactive tool to set game region bounds
│
├── agent/
│   ├── env.py              # gymnasium env wrapping game state
│   ├── train.py            # SB3 training entry point (uses mock env)
│   └── inference.py        # load policy, return target paddle X
│
├── overlay/
│   └── renderer.py         # transparent pygame window, draws dot
│
├── demo.py                 # main entry point wiring all components
└── config.yaml             # region bounds, grid dims, colors, thresholds
```

## Component responsibilities

### vision/
Owns everything between raw screen pixels and the structured game state dict.
Produces one state per frame. Must not make game decisions.

### agent/
Owns the policy. Takes a game state dict, returns a float: target paddle X
in normalized game coordinates [0.0, 1.0]. Training happens offline against
a mock environment; inference runs live.

### overlay/
Owns the transparent window. Takes a screen pixel X coordinate, draws a
circle indicator at the bottom of the game region. No game logic here.

### demo.py
The main loop. Wires vision → agent → overlay at ~30fps. Also handles
calibration flow on first run.

## Game state dict schema

```python
{
    "ball_pos": (float, float),       # center x, y in game pixels
    "ball_vel": (float, float),       # dx, dy estimated from frame history
    "paddle_x": float,                # center x of paddle in game pixels
    "brick_grid": np.ndarray,         # shape (rows, cols), dtype int8
                                      # 0=empty 1=broken 2=full 3=diamond 4=golden 5=boot
    "lives": int,                     # current life count
    "score": int,                     # current score
}
```

## Brick type encoding

| Code | Type    | Hits to break | Notes                        |
|------|---------|---------------|------------------------------|
| 0    | Empty   | —             | cleared cell                 |
| 1    | Broken  | 1             | standard                     |
| 2    | Full    | 2             | two-hit                      |
| 3    | Diamond | ∞             | indestructible, route around |
| 4    | Golden  | 1             | +50 bonus points on break    |
| 5    | Boot    | 1             | grants +1 life on break      |

## config.yaml schema

```yaml
capture:
  game_region:            # pixel coords of game window on screen
    left: 0
    top: 0
    width: 800
    height: 600

grid:
  rows: 10
  cols: 14

vision:
  ball_history_frames: 8  # frames used for velocity smoothing
  hsv_thresholds:         # per brick type, set during calibration
    broken:  { h: [0,0],   s: [0,0],   v: [0,0] }
    full:    { h: [0,0],   s: [0,0],   v: [0,0] }
    diamond: { h: [0,0],   s: [0,0],   v: [0,0] }
    golden:  { h: [20,35], s: [150,255], v: [150,255] }
    boot:    { h: [0,0],   s: [0,0],   v: [0,0] }

overlay:
  dot_radius: 12
  dot_color: [255, 80, 80]   # RGB
  dot_alpha: 200
  fps: 30
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running

**Step 1 — Calibrate (first run only):**
```bash
python -m vision.calibrate
```
Follow the interactive prompts to drag-select the game region and sample
each brick type color. Writes results to config.yaml.

**Step 2 — Train the agent (offline, once):**
```bash
python -m agent.train
```
Trains on a mock gymnasium environment. Saves policy to `agent/policy.zip`.

**Step 3 — Run the overlay:**
```bash
python demo.py
```
Opens the transparent overlay window. Launch Dota 2, enter the minigame.

## Requirements

```
mss
opencv-python
numpy
pygame
gymnasium
stable-baselines3
torch
pyyaml
```

## Implementation order

Build in this sequence to allow testing at each stage:

1. `vision/capture.py` — verify you can grab frames
2. `vision/calibrate.py` — set region and color thresholds interactively
3. `vision/brick_parser.py` — verify grid parsing on a static screenshot
4. `vision/tracker.py` — verify ball/paddle detection on a recorded clip
5. `agent/env.py` — mock env with no CV dependency, test in isolation
6. `agent/train.py` + `agent/inference.py` — train and smoke-test policy
7. `overlay/renderer.py` — test overlay draws correctly over a static window
8. `demo.py` — integrate everything, run live

## Notes

- The overlay window must be external (transparent always-on-top OS window).
  Do not inject into the Dota 2 process. Do not use DirectX/Vulkan hooks.
- Screen capture via `mss` is entirely external and safe.
- Ball velocity is estimated from frame history, not ground truth. Use a
  rolling linear regression over `ball_history_frames` positions.
- Diamond bricks are permanent obstacles. The agent must learn to route
  around them, not target them.
- Boot bricks (+1 life) should have positive reward on break to incentivize
  targeting them when accessible.
- Golden bricks (+50 pts) should have elevated reward weight in the reward
  function.
