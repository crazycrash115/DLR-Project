import gym
import numpy as np
from gym import Wrapper, ObservationWrapper, spaces as gym_spaces
from gymnasium import spaces as gymn_spaces

class GymnasiumToGymV21(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

class GymnasiumActionFix(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.action_space, gymn_spaces.Discrete):
            self.action_space = gym_spaces.Discrete(env.action_space.n)

class AddChannelWrapper(ObservationWrapper):
    """
    Ensure final obs is channel-first (1, H, W) no matter whether
    input is (H,W)  or (H,W,1).
    """
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[:2]      # ignore trailing dim
        self.observation_space = gym_spaces.Box(
            low=0, high=255,
            shape=(1, h, w),         # (C, H, W)
            dtype=np.uint8
        )

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[-1] == 1:     # (H,W,1) → (H,W)
            obs = obs[:, :, 0]
        return np.expand_dims(obs, 0)                # (1,H,W)

#unused
class FlappyRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        bird_y   = info.get("player_y", 0)
        pipe_mid = info.get("pipe_gap_y", bird_y)
        stability = 1.0 - abs(bird_y - pipe_mid) / self.env.height
        reward += 0.01 * stability
        return obs, reward, done, info
