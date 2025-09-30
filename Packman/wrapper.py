import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Controls(gym.ActionWrapper):
    """
    Discrete(4): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Mapped to ALE's action indices via get_action_meanings().
    """
    def __init__(self, env):
        super().__init__(env)
        meanings = self.unwrapped.get_action_meanings()
        want = {"UP": None, "DOWN": None, "LEFT": None, "RIGHT": None}
        for i, m in enumerate(meanings):
            mm = m.upper()
            if mm in want and want[mm] is None:
                want[mm] = i
        missing = [k for k, v in want.items() if v is None]
        if missing:
            raise RuntimeError(f"Missing actions {missing} in {meanings}")
        self._map = np.array([want["UP"], want["DOWN"], want["LEFT"], want["RIGHT"]], dtype=np.int64)
        self.action_space = spaces.Discrete(4)

    def action(self, a):
        return int(self._map[int(a)])


class ScaleWrapper(gym.ObservationWrapper):
    """(N,) uint8 RAM -> float32 in [0,1] (keeps shape)."""
    def __init__(self, env, scale=True):
        super().__init__(env)
        self.scale = bool(scale)
        n = int(env.observation_space.shape[0])
        low, high = (0.0, 1.0) if self.scale else (0, 255)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(n,),
            dtype=np.float32 if self.scale else np.uint8
        )

    def observation(self, obs):
        arr = np.asarray(obs)
        return (arr.astype(np.float32) / 255.0) if self.scale else arr


class FrameStack(gym.ObservationWrapper):
    """Stack k RAM vectors: (N,) -> (N*k,)."""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.k = int(num_stack)
        n = int(env.observation_space.shape[0])
        self._buf = np.zeros((self.k, n), dtype=env.observation_space.dtype)
        low = np.repeat(env.observation_space.low, self.k)
        high = np.repeat(env.observation_space.high, self.k)
        # shape inferred from low/high
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf[...] = obs
        return self._buf.reshape(-1), info

    def observation(self, obs):
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = obs
        return self._buf.reshape(-1)

import gymnasium as gym

class OneLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._lives = self.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get("lives", self.unwrapped.ale.lives())
        if lives < self._lives:
            terminated = True
        self._lives = lives
        return obs, reward, terminated, truncated, info

class PacmanRewardWrapper(gym.Wrapper):
    """
    r = base_scale*raw + survive_bonus
        - no_score_penalty (after patience)
        - death_penalty on life loss (if lives is reported)
    Adds info['raw_return'] at episode end.
    """
    def __init__(self, env,
                 base_scale=0.02,
                 survive_bonus=0.02,
                 no_score_patience=60,
                 no_score_penalty=0.0,
                 death_penalty=10.0,
                 max_steps=None):
        super().__init__(env)
        self.base_scale = base_scale
        self.survive_bonus = survive_bonus
        self.no_score_patience = no_score_patience
        self.no_score_penalty = no_score_penalty
        self.death_penalty = death_penalty
        self.max_steps = max_steps

        self._raw_return = 0.0
        self._no_score = 0
        self._prev_lives = None
        self._steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._raw_return = 0.0
        self._no_score = 0
        self._steps = 0
        self._prev_lives = info.get("lives", None)
        return obs, info

    def step(self, action):
        obs, raw_r, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        r = self.base_scale * float(raw_r) + self.survive_bonus
        self._raw_return += float(raw_r)

        if raw_r <= 0.0:
            self._no_score += 1
            if self._no_score > self.no_score_patience and self.no_score_penalty:
                r -= self.no_score_penalty
        else:
            self._no_score = 0

        lives = info.get("lives", self._prev_lives)
        if (self._prev_lives is not None) and (lives is not None) and (lives < self._prev_lives):
            r -= self.death_penalty
            self._no_score = 0
        self._prev_lives = lives

        if (self.max_steps is not None) and (self._steps >= self.max_steps):
            truncated = True

        if terminated or truncated:
            info = dict(info)
            info["raw_return"] = self._raw_return

        return obs, r, terminated, truncated, info
