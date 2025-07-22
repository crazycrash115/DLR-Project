import gym
import numpy as np

class PongHitRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(PongHitRewardWrapper, self).__init__(env)
        self.prev_obs = None
        self.prev_ball_x = None
        self.hit_bonus = 0.5 #tweak

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        self.prev_ball_x = self._get_ball_x(self.prev_obs)
        return self.prev_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        ball_x = self._get_ball_x(obs)

        # Detect hit by checking direction change
        if self.prev_ball_x is not None:
            if (self.prev_ball_x < ball_x < 80) or (self.prev_ball_x > ball_x > 80):
                reward += self.hit_bonus

        self.prev_obs = obs
        self.prev_ball_x = ball_x
        return obs, reward, done, info

    def _get_ball_x(self, obs):
        # Ball is white (236,236,236) 
        ball_coords = np.argwhere(obs == 236)
        if ball_coords.size == 0:
            return 0  # Not visible
        return np.mean(ball_coords[:, 1])  # Average x pos

