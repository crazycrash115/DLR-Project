import numpy as np
import gymnasium as gymn
from gymnasium import spaces as gymn_spaces
from gymnasium import ObservationWrapper
from collections import deque

class AddChannelWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        if len(shp) == 2:
            h, w = shp
        elif len(shp) == 3:
            h, w = shp[:2]
        else:
            raise ValueError(f"Unexpected obs shape: {shp}")
        self.observation_space = gymn_spaces.Box(low=0, high=255, shape=(1, h, w), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[-1] == 1:
            obs = obs[..., 0]
        elif obs.ndim == 3 and obs.shape[-1] == 3:
            r = obs[..., 0].astype(np.float32)
            g = obs[..., 1].astype(np.float32)
            b = obs[..., 2].astype(np.float32)
            obs = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
        return np.expand_dims(obs, 0)

class FlappyRewardWrapper(gymn.Wrapper):
    def __init__(self, env, survive_bonus=0.01, gap_weight=0.02, death_penalty=1.0):
        super().__init__(env)
        self.survive_bonus = float(survive_bonus)
        self.gap_weight = float(gap_weight)
        self.death_penalty = float(death_penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self.survive_bonus
        bird_y = info.get("player_y")
        gap_y  = info.get("pipe_gap_y")
        H      = getattr(self.env, "height", info.get("screen_height", 256)) or 256
        if bird_y is not None and gap_y is not None and H:
            proximity = 1.0 - abs(float(bird_y) - float(gap_y)) / float(H)
            reward += self.gap_weight * proximity
        if terminated:
            reward -= self.death_penalty
        return obs, reward, terminated, truncated, info

class ChannelFrameStack(ObservationWrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        c, h, w = env.observation_space.shape
        self.k = int(k)
        self.frames = deque(maxlen=self.k)
        self.observation_space = gymn_spaces.Box(low=0, high=255, shape=(self.k, h, w), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        f = obs[0]
        for _ in range(self.k):
            self.frames.append(f)
        return self._get_obs(), info

    def observation(self, obs):
        self.frames.append(obs[0])
        return self._get_obs()

    def _get_obs(self):
        return np.stack(self.frames, axis=0)
