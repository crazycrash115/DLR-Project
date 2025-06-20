import gym
import numpy as np

class SnakeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None
        self.steps_since_last_apple = 0
        self.steps_near_food = 0

        # Tuning thresholds
        self.min_eat_dist = 1.5           # close to food
        self.min_progress_delta = 0.5     # significant distance improvement
        self.max_orbit_steps = 4          # how long it's allowed to linger near food

    def reset(self):
        obs = self.env.reset()
        head = self.env.controller.snakes[0].head
        food = self.get_food_position(self.env.controller.grid)

        if food:
            self.prev_distance = np.linalg.norm(np.array(head) - np.array(food))
        else:
            self.prev_distance = None

        self.steps_since_last_apple = 0
        self.steps_near_food = 0
        return obs

    def get_food_position(self, grid):
        color = np.array([0, 0, 255], dtype=np.uint8)
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)

        if coords.shape[0] == 0:
            return None

        y_px, x_px = coords[0]
        return (x_px // grid.unit_size, y_px // grid.unit_size)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps_since_last_apple += 1

        if reward == 1:
            # Ate food + speed bonus
            fast_bonus = max(0, 1.0 - 0.02 * self.steps_since_last_apple)
            reward = 2 + fast_bonus
            self.prev_distance = None
            self.steps_since_last_apple = 0
            self.steps_near_food = 0

        elif reward == -1:
            # Died
            reward = -1
            self.prev_distance = None
            self.steps_near_food = 0

        else:
            # Time penalty every step to prevent stalling (try upping?)
            reward -= 0.01

            snake = self.env.controller.snakes[0]
            if snake:
                head = snake.head
                food = self.get_food_position(self.env.controller.grid)

                if food:
                    dist = np.linalg.norm(np.array(head) - np.array(food))

                    if self.prev_distance is not None:
                        delta = self.prev_distance - dist

                        # Only reward if the improvement is meaningful
                        if delta > self.min_progress_delta:
                            reward += 0.2
                        elif delta < -self.min_progress_delta:
                            reward -= 0.1

                        # Penalize orbiting around food too long
                        if dist < self.min_eat_dist:
                            self.steps_near_food += 1
                            if self.steps_near_food > self.max_orbit_steps:
                                reward -= 1.0
                        else:
                            self.steps_near_food = 0

                        self.prev_distance = dist

        return obs, reward, done, info#
