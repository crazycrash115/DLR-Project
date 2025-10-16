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

    def __init__(self, env, gamma=0.99, gap_weight=0.5, extra_pipe_bonus=0.0):
        super().__init__(env)
        self.gamma = float(gamma)
        self.gap_weight = float(gap_weight)
        self.extra_pipe_bonus = float(extra_pipe_bonus)
        self._prev_phi = None
        self._prev_score = 0

    def _phi_from_info(self, info):
        bird_y = info.get("player_y")
        gap_y  = info.get("pipe_gap_y")
        H      = getattr(self.env, "height", info.get("screen_height", 256)) or 256
        if bird_y is None or gap_y is None or not H:
            return None
        return -abs(float(bird_y) - float(gap_y)) / float(H)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi_from_info(info)
        sc = info.get("score", 0)
        self._prev_score = int(sc) if isinstance(sc, (int, float)) else 0
        return obs, info

    def step(self, action):
        obs, env_r, terminated, truncated, info = self.env.step(action)

        # Base env reward
        reward = float(env_r)

        # Optional extra bonus for pipe(s)
        cur_score = info.get("score", self._prev_score)
        cur_score = int(cur_score) if isinstance(cur_score, (int, float)) else self._prev_score
        delta = max(0, cur_score - self._prev_score)
        if delta > 0 and self.extra_pipe_bonus != 0.0:
            reward += self.extra_pipe_bonus * delta
        self._prev_score = cur_score

        # Potential-based shaping
        phi_next = self._phi_from_info(info)
        if self._prev_phi is not None and phi_next is not None:
            shaping = self.gamma * phi_next - self._prev_phi
            reward += self.gap_weight * float(shaping)
        self._prev_phi = phi_next

        return obs, float(reward), terminated, truncated, info

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
