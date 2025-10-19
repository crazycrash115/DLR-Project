import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

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
        self.prev_head = None
        self.steps_since_last_apple = 0
        self.steps_near_food = 0
        self.recent = deque(maxlen=6)
        self.min_eat_dist = 1.5
        self.progress_scale = 0.06
        self.align_scale = 0.05
        self.linger_eps = 0.05
        self.linger_penalty = 0.03
        self.osc_penalty = 0.4

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        base = self.unwrapped
        self.prev_head = None
        self.steps_since_last_apple = 0
        self.steps_near_food = 0
        self.recent.clear()
        if hasattr(base, "controller") and base.controller.snakes:
            head = np.array(base.controller.snakes[0].head, dtype=np.float32)
            food = self.get_food_position(base.controller.grid)
            self.prev_head = tuple(head.tolist())
            self.prev_distance = np.linalg.norm(head - np.array(food, dtype=np.float32)) if food is not None else None
        else:
            self.prev_distance = None
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
            fast_bonus = max(0.0, 1.0 - 0.02 * self.steps_since_last_apple)
            reward = 2.0 + fast_bonus
            self.prev_distance = None
            self.prev_head = None
            self.steps_since_last_apple = 0
            self.steps_near_food = 0
            self.recent.clear()
            return obs, reward, terminated, truncated, info

        if reward == -1:
            reward = -30.0
            self.prev_distance = None
            self.prev_head = None
            self.steps_near_food = 0
            self.recent.clear()
            return obs, reward, terminated, truncated, info

        reward -= 0.005

        base = self.unwrapped
        snake = base.controller.snakes[0] if hasattr(base, "controller") and base.controller.snakes else None
        if snake:
            head = np.array(snake.head, dtype=np.float32)
            food = self.get_food_position(base.controller.grid)
            if food:
                food = np.array(food, dtype=np.float32)
                dist = np.linalg.norm(head - food)

                grid = base.controller.grid
                diag = float(np.hypot(grid.grid_size[0], grid.grid_size[1]))
                if self.prev_distance is not None and diag > 0:
                    reward += self.progress_scale * ((self.prev_distance - dist) / diag)

                move_vec = np.zeros(2, dtype=np.float32) if self.prev_head is None else head - np.array(self.prev_head, dtype=np.float32)
                to_food = food - head
                if np.linalg.norm(move_vec) > 0 and np.linalg.norm(to_food) > 0:
                    mv = move_vec / (np.linalg.norm(move_vec) + 1e-8)
                    tf = to_food / (np.linalg.norm(to_food) + 1e-8)
                    reward += self.align_scale * float(np.dot(mv, tf))

                if dist < self.min_eat_dist:
                    self.steps_near_food += 1
                    if self.prev_distance is not None and (self.prev_distance - dist) < self.linger_eps:
                        reward -= self.linger_penalty * (1.05 ** self.steps_near_food)
                else:
                    self.steps_near_food = 0

                self.recent.append(tuple(head.tolist()))
                if len(self.recent) >= 4 and self.recent[-1] == self.recent[-3] and self.recent[-2] == self.recent[-4]:
                    reward -= self.osc_penalty

                self.prev_distance = dist
                self.prev_head = tuple(head.tolist())

        if truncated and not terminated:
            reward -= 5.0

        return obs, float(reward), terminated, truncated, info
