# agent.md — RL Agent Skill

Skill doc for implementing `agent/`. This module defines the gymnasium
environment, reward function, training procedure, and live inference.

## Responsibilities

- Define a gymnasium-compatible env that wraps the game state dict
- Define a reward function that reflects the brick type table
- Train a policy offline using stable-baselines3
- At inference time, take a state dict and return a target paddle X [0.0, 1.0]

## env.py — Gymnasium environment

The environment abstracts the game state dict into gym-standard obs/action
spaces. Training runs against a **mock physics simulation**, not the live
game — this means you can train entirely offline without Dota 2 open.

### Observation space

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

GRID_ROWS = 10
GRID_COLS = 14
BRICK_TYPES = 6  # 0–5

obs_space = spaces.Dict({
    "ball_pos": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
    "ball_vel": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    "paddle_x": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    "brick_grid": spaces.Box(low=0, high=5,
                              shape=(GRID_ROWS, GRID_COLS), dtype=np.int8),
})
```

All positions normalized to [0.0, 1.0] within the game region.
Ball velocity normalized to max expected speed.

### Action space

Continuous: target paddle X position in [0.0, 1.0].

```python
action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
```

The overlay renderer maps this normalized value back to a screen pixel X.

### Reward function

This is the most important design decision. Encode the brick type table:

```python
BRICK_REWARDS = {
    0: 0.0,    # empty — no reward
    1: 1.0,    # broken — standard
    2: 2.0,    # full — two hits total, reward per hit
    3: 0.0,    # diamond — indestructible, never reward
    4: 6.0,    # golden — +50pts bonus, weight higher
    5: 3.0,    # boot — +1 life, high value
}

LIFE_LOST_PENALTY = -15.0
SURVIVAL_REWARD = 0.01    # small reward per step for staying alive
```

**Reward rationale:**
- Boot brick reward (3.0) is higher than full brick (2.0) because an extra
  life has compounding value — it allows more bricks to be cleared later.
- Golden brick reward (6.0) is highest per-hit because +50pts is significant
  and it only takes 1 hit.
- Life lost penalty (-15.0) dominates to prioritize ball survival above all.
- Survival reward (0.01/step) encourages longer episodes.

**Step reward assembly:**
```python
def compute_reward(prev_grid, curr_grid, life_lost, steps_survived):
    reward = 0.0
    # Reward for each brick that changed state this step
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            prev = prev_grid[r, c]
            curr = curr_grid[r, c]
            if prev != curr:
                # Brick was hit or cleared
                reward += BRICK_REWARDS.get(prev, 0.0)
    if life_lost:
        reward += LIFE_LOST_PENALTY
    reward += SURVIVAL_REWARD
    return reward
```

### Mock physics simulation

For offline training, implement a simple Breakout physics sim inside env.py.
It does not need to match Dark Carnival's exact physics — the agent learns
general paddle positioning strategy that transfers.

```python
class BreakoutMockEnv(gym.Env):
    def __init__(self, grid_rows=10, grid_cols=14):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.observation_space = obs_space
        self.action_space = action_space
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ball_pos = np.array([0.5, 0.7], dtype=np.float32)
        self.ball_vel = np.array([0.015, -0.02], dtype=np.float32)
        self.paddle_x = 0.5
        self.paddle_w = 0.12   # paddle width as fraction of game width
        self.brick_grid = self._generate_grid()
        self.lives = 3
        self.prev_grid = self.brick_grid.copy()
        return self._obs(), {}

    def _generate_grid(self):
        # Random grid respecting brick type distribution
        # Adjust probabilities to match observed Dark Carnival layouts
        grid = np.random.choice(
            [0, 1, 2, 3, 4, 5],
            size=(self.grid_rows, self.grid_cols),
            p=[0.1, 0.4, 0.25, 0.1, 0.1, 0.05]
        ).astype(np.int8)
        # Empty top rows (bricks only in upper portion)
        grid[7:, :] = 0
        return grid

    def step(self, action):
        # Move paddle toward target
        target_x = float(action[0])
        self.paddle_x += np.clip(target_x - self.paddle_x, -0.05, 0.05)
        self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

        # Physics step
        self.ball_pos += self.ball_vel

        # Wall bounce (left/right)
        if self.ball_pos[0] <= 0 or self.ball_pos[0] >= 1:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], 0, 1)

        # Ceiling bounce
        if self.ball_pos[1] <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = 0

        # Paddle collision
        life_lost = False
        if self.ball_pos[1] >= 0.9:
            half_paddle = self.paddle_w / 2
            if abs(self.ball_pos[0] - self.paddle_x) <= half_paddle:
                self.ball_vel[1] *= -1
                # Angle deflection based on hit position relative to paddle center
                offset = (self.ball_pos[0] - self.paddle_x) / half_paddle
                self.ball_vel[0] += offset * 0.005
            else:
                life_lost = True
                self.lives -= 1
                self.ball_pos = np.array([0.5, 0.7])
                self.ball_vel = np.array([0.015, -0.02])

        # Brick collision (simplified: check ball grid cell)
        br = int(self.ball_pos[1] * self.grid_rows)
        bc = int(self.ball_pos[0] * self.grid_cols)
        br = np.clip(br, 0, self.grid_rows - 1)
        bc = np.clip(bc, 0, self.grid_cols - 1)

        brick = self.brick_grid[br, bc]
        if brick == 3:
            # Diamond: bounce back, no break
            self.ball_vel[1] *= -1
        elif brick > 0:
            # All other non-empty bricks: reduce or clear
            if brick == 2:
                self.brick_grid[br, bc] = 1  # full → broken
            else:
                self.brick_grid[br, bc] = 0  # cleared
            self.ball_vel[1] *= -1

        reward = compute_reward(self.prev_grid, self.brick_grid,
                                 life_lost, steps_survived=1)
        self.prev_grid = self.brick_grid.copy()

        terminated = self.lives <= 0 or self.brick_grid[self.brick_grid != 3].sum() == 0
        truncated = False

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return {
            "ball_pos": self.ball_pos.copy(),
            "ball_vel": np.clip(self.ball_vel / 0.05, -1, 1).astype(np.float32),
            "paddle_x": np.array([self.paddle_x], dtype=np.float32),
            "brick_grid": self.brick_grid.copy(),
        }
```

## train.py

```python
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from agent.env import BreakoutMockEnv

def train(total_timesteps=500_000, save_path="agent/policy"):
    env = BreakoutMockEnv()
    check_env(env, warn=True)

    vec_env = DummyVecEnv([lambda: BreakoutMockEnv()])

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        tensorboard_log="./logs/",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Policy saved to {save_path}.zip")

if __name__ == "__main__":
    train()
```

**Why SAC:** The action space is continuous (target X position). SAC
(Soft Actor-Critic) handles continuous actions well and is sample-efficient.
PPO with a continuous action head is a valid alternative if SAC is unfamiliar.

**Training time estimate:** 500k steps on CPU takes ~20–40 minutes. On GPU,
under 10 minutes. The mock env is fast since there's no rendering.

## inference.py

```python
from stable_baselines3 import SAC
import numpy as np

class AgentInference:
    def __init__(self, policy_path="agent/policy.zip"):
        self.model = SAC.load(policy_path)

    def predict(self, state_dict: dict) -> float:
        # Returns target paddle X in [0.0, 1.0]
        obs = self._normalize_obs(state_dict)
        action, _ = self.model.predict(obs, deterministic=True)
        return float(np.clip(action[0], 0.0, 1.0))

    def _normalize_obs(self, state_dict):
        # Convert raw game state dict to normalized obs format
        # Requires game region dimensions for normalization
        return {
            "ball_pos": np.array(state_dict["ball_pos"], dtype=np.float32),
            "ball_vel": np.array(state_dict["ball_vel"], dtype=np.float32),
            "paddle_x": np.array([state_dict["paddle_x"]], dtype=np.float32),
            "brick_grid": state_dict["brick_grid"].astype(np.int8),
        }
```

## Domain gap note

The mock env's physics will not perfectly match Dark Carnival's. After the
agent performs reasonably in demo, consider:

1. Recording ~30 minutes of live gameplay (ball trajectories, brick breaks)
2. Fitting the mock env's speed/bounce parameters to match the recording
3. Re-training on the calibrated mock env

This closes the sim-to-real gap without needing to train on live game frames.

## Observation flattening for SB3

SB3's `MultiInputPolicy` handles Dict observation spaces natively, but the
brick grid (2D array) needs to be flattened. Wrap the env:

```python
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

env = FlattenObservation(BreakoutMockEnv())
```

Or manually flatten `brick_grid` inside `_obs()` to a 1D array of length
`rows * cols` and update the observation space accordingly.
