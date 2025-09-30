import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GymV21toGymnasium(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = self._to_gymnasium_space(env.action_space)
        self.observation_space = self._to_gymnasium_space(env.observation_space)
        self.spec = getattr(env, "spec", None)
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            elif hasattr(self.env, "reset_seed"):
                self.env.reset_seed(seed)
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, (info or {})
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
            return obs, float(reward), terminated, truncated, info
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _to_gymnasium_space(self, space):
        if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
            return spaces.Box(low=np.array(space.low), high=np.array(space.high),
                              shape=getattr(space, "shape", None), dtype=getattr(space, "dtype", np.float32))
        if hasattr(space, "n") and not hasattr(space, "nvec"):
            return spaces.Discrete(int(space.n))
        if hasattr(space, "nvec"):
            return spaces.MultiDiscrete(np.array(space.nvec, dtype=np.int64))
        if hasattr(space, "shape") and space.shape != ():
            return spaces.MultiBinary(space.shape)
        return space

class SnakeActionListWrapper(gym.ActionWrapper):
    def action(self, act):
        if isinstance(act, (np.integer, int)):
            return [int(act)]
        arr = np.asarray(act)
        if arr.ndim == 0:
            return [int(arr)]
        return list(arr)
    
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

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        base = self.unwrapped
        if hasattr(base, "controller") and base.controller.snakes:
            head = base.controller.snakes[0].head
            food = self.get_food_position(base.controller.grid)
            if food is not None:
                self.prev_distance = np.linalg.norm(np.array(head) - np.array(food))
            else:
                self.prev_distance = None
        else:
            self.prev_distance = None

        self.steps_since_last_apple = 0
        self.steps_near_food = 0
        return obs, info

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_food_position(self, grid):
        color = np.array([0, 0, 255], dtype=np.uint8)
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)

        if coords.shape[0] == 0:
            return None

        y_px, x_px = coords[0]
        return (x_px // grid.unit_size, y_px // grid.unit_size)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_since_last_apple += 1

        if reward == 1:
            # Ate food + speed bonus
            fast_bonus = max(0, 1.0 - 0.02 * self.steps_since_last_apple)
            reward = 2 + fast_bonus
            self.prev_distance = None
            self.steps_since_last_apple = 0
            self.steps_near_food = 0

        # Died
        elif reward == -1:
            reward = -5 # Hopefully itll not want to die anymore 
            self.prev_distance = None
            self.steps_near_food = 0

        else:
            # Time penalty every step to prevent stalling (try upping?)
            reward -= 0.01

            base = self.unwrapped
            snake = base.controller.snakes[0] if hasattr(base, "controller") and base.controller.snakes else None
            if snake:
                head = snake.head
                food = self.get_food_position(base.controller.grid)

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

        return obs, reward, terminated, truncated, info
