import gym
import numpy as np

class SnakeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.controller.grid.grid_size
        self.max_body_parts = 100  # number of tail segments to track

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6 + 2 * self.max_body_parts,), dtype=np.float32
        )

    def observation(self, obs):
        snake = self.env.controller.snakes[0]
        if snake is None:
            #dead or not initialized, return zeros
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        head = np.array(snake.head, dtype=np.float32)
        food = self.get_food_position(self.env.controller.grid)

        if food is None:
            food = np.array([0, 0], dtype=np.float32)

        delta = food - head

        grid_w, grid_h = self.grid_size
        norm = np.array([grid_w, grid_h], dtype=np.float32)

        head_norm = head / norm
        food_norm = food / norm
        delta_norm = delta / norm

        # Body part positions (up to max_body_parts)
        body_coords = list(snake.body)[-self.max_body_parts:]
        padded_coords = [np.array(bc, dtype=np.float32) / norm for bc in body_coords]

        # Pad if less than max_body_parts
        while len(padded_coords) < self.max_body_parts:
            padded_coords.append(np.array([0.0, 0.0], dtype=np.float32))

        body_flat = np.concatenate(padded_coords)

        return np.concatenate([head_norm, food_norm, delta_norm, body_flat]).astype(np.float32)

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
