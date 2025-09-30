import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_body_parts = 100  # number of tail segments to track

        # Filled after first reset (controller/grid not built until reset())
        self.grid_size = None
        self.unit_size = None

        # fixed-size feature vector (independent of grid size)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6 + 2 * self.max_body_parts,), dtype=np.float32
        )

    def _maybe_init_grid_params(self):
        # only read controller/grid after reset; use base env directly
        base = self.unwrapped
        if (self.grid_size is None or self.unit_size is None) and hasattr(base, "controller"):
            grid = base.controller.grid
            self.grid_size = grid.grid_size
            self.unit_size = grid.unit_size

    def get_food_position(self, grid):
        color = np.array([0, 0, 255], dtype=np.uint8)
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)
        if coords.shape[0] == 0:
            return None
        y_px, x_px = coords[0]
        x_unit = x_px // grid.unit_size
        y_unit = y_px // grid.unit_size
        return np.array([x_unit, y_unit], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._maybe_init_grid_params()
        return self.observation(obs), info

    def observation(self, obs):
        # build features each step, do nothing if controller not ready yet
        self._maybe_init_grid_params()

        if self.grid_size is None:
            # not initialized yet, return zeros to keep shapes stable
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        grid_w, grid_h = self.grid_size
        norm = np.array([float(grid_w), float(grid_h)], dtype=np.float32)

        base = self.unwrapped
        snake = getattr(base.controller, "snakes", [None])[0] if hasattr(base, "controller") else None
        if snake is None:
            # dead or not initialized, return zeros
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        head = np.array(snake.head, dtype=np.float32)
        food = self.get_food_position(base.controller.grid)
        if food is None:
            food = np.array([0.0, 0.0], dtype=np.float32)

        delta = food - head

        head_norm  = head / norm
        food_norm  = food / norm
        delta_norm = delta / norm

        # Body part positions (up to max_body_parts)
        body_coords = list(snake.body)[-self.max_body_parts:]
        padded_coords = [np.array(bc, dtype=np.float32) / norm for bc in body_coords]

        # Pad if less than max_body_parts
        while len(padded_coords) < self.max_body_parts:
            padded_coords.append(np.array([0.0, 0.0], dtype=np.float32))

        body_flat = np.concatenate(padded_coords).astype(np.float32)

        return np.concatenate([head_norm, food_norm, delta_norm, body_flat]).astype(np.float32)
