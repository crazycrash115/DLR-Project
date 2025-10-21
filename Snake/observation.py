import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_body_parts=100):
        super().__init__(env)
        self.max_body_parts = int(max_body_parts)
        self.grid_size = None
        self.unit_size = None

        d = 12 + 3 * self.max_body_parts
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(d,), dtype=np.float32)
        self._prev_head = None

    def _maybe_init_grid_params(self):
        base = self.unwrapped
        if (self.grid_size is None or self.unit_size is None) and hasattr(base, "controller"):
            grid = base.controller.grid
            self.grid_size = grid.grid_size
            self.unit_size = grid.unit_size

    def _get_food_positions_units(self, base):
        foods = []
        if hasattr(base, "controller"):
            c = base.controller
            if hasattr(c, "foods") and c.foods:
                for f in c.foods:
                    if hasattr(f, "position"):
                        foods.append(np.array(f.position, dtype=np.float32))
                    elif hasattr(f, "pos"):
                        foods.append(np.array(f.pos, dtype=np.float32))
                    elif hasattr(f, "location"):
                        foods.append(np.array(f.location, dtype=np.float32))
        if len(foods) == 0 and hasattr(base, "controller"):
            grid = base.controller.grid
            color = np.array([0, 0, 255], dtype=np.uint8)
            matches = np.all(grid.grid == color, axis=2)
            coords = np.argwhere(matches)
            for y_px, x_px in coords:
                foods.append(np.array([x_px // grid.unit_size, y_px // grid.unit_size], dtype=np.float32))
        return foods

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._maybe_init_grid_params()
        self._prev_head = None
        return self.observation(obs), info

    def observation(self, obs):
        self._maybe_init_grid_params()
        if self.grid_size is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        base = self.unwrapped
        snake = getattr(base.controller, "snakes", [None])[0] if hasattr(base, "controller") else None
        if snake is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        gw, gh = self.grid_size
        norm = np.array([float(gw), float(gh)], dtype=np.float32)

        head = np.array(snake.head, dtype=np.float32)
        head_norm = head / norm

        def _onehot_from_dxdy(dx, dy):
            return np.array([
                1.0 if (dx, dy) == (0, -1) else 0.0,
                1.0 if (dx, dy) == (0,  1) else 0.0,
                1.0 if (dx, dy) == (-1, 0) else 0.0,
                1.0 if (dx, dy) == (1,  0) else 0.0,
            ], dtype=np.float32)

        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_val = getattr(snake, "direction", None)
        used = False
        if dir_val is not None:
            try:
                arr = np.asarray(dir_val)
                if arr.shape == (2,):
                    dx, dy = int(arr[0]), int(arr[1])
                    dir_onehot = _onehot_from_dxdy(dx, dy)
                    used = True
            except Exception:
                pass

        if not used:
            if self._prev_head is None:
                dir_onehot = np.zeros(4, dtype=np.float32)
            else:
                d = head - np.array(self._prev_head, dtype=np.float32)
                if abs(d[0]) > abs(d[1]):
                    dir_onehot = np.array(
                        [0.0, 0.0, 1.0 if d[0] < 0 else 0.0, 1.0 if d[0] > 0 else 0.0],
                        dtype=np.float32
                    )
                else:
                    dir_onehot = np.array(
                        [1.0 if d[1] < 0 else 0.0, 1.0 if d[1] > 0 else 0.0, 0.0, 0.0],
                        dtype=np.float32
                    )

        foods = self._get_food_positions_units(base)
        if len(foods) == 0:
            food_norm = np.array([0.0, 0.0], dtype=np.float32)
            delta_norm = np.array([0.0, 0.0], dtype=np.float32)
            dist_norm = np.array([0.0], dtype=np.float32)
        else:
            foods = np.stack(foods, axis=0)
            dists = np.linalg.norm(foods - head[None, :], axis=1)
            j = int(np.argmin(dists))
            food = foods[j]
            food_norm = food / norm
            delta_norm = (food - head) / norm
            max_d = np.linalg.norm(norm)
            dist_norm = np.array([float(dists[j] / max_d)], dtype=np.float32)

        body_coords = list(snake.body)[-self.max_body_parts:]
        k = len(body_coords)
        if k > 0:
            body_arr = np.stack([np.array(p, dtype=np.float32) for p in body_coords], axis=0) / norm
        else:
            body_arr = np.zeros((0, 2), dtype=np.float32)
        if k < self.max_body_parts:
            pad = np.zeros((self.max_body_parts - k, 2), dtype=np.float32)
            body_arr = np.concatenate([body_arr, pad], axis=0)
        body_flat = body_arr.reshape(-1)

        body_mask = np.zeros((self.max_body_parts,), dtype=np.float32)
        if k > 0:
            body_mask[:k] = 1.0

        length_norm = np.array([min(1.0, k / float(self.max_body_parts))], dtype=np.float32)

        self._prev_head = tuple(head.tolist())

        feat = np.concatenate([
            head_norm,
            dir_onehot,
            food_norm,
            delta_norm,
            dist_norm,
            length_norm,
            body_flat,
            body_mask,
        ]).astype(np.float32)

        return feat
