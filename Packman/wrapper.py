import numpy as np
import gymnasium as gymn
from gymnasium import ObservationWrapper, spaces as gymn_spaces

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
        self.observation_space = gymn_spaces.Box(
            low=0, high=255, shape=(1, h, w), dtype=np.uint8
        )

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[-1] == 1:
            obs = obs[..., 0]
        elif obs.ndim == 3 and obs.shape[-1] == 3:
            r = obs[..., 0].astype(np.float32)
            g = obs[..., 1].astype(np.float32)
            b = obs[..., 2].astype(np.float32)
            obs = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
        return np.expand_dims(obs, 0)

class PacmanRewardWrapper(gymn.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
