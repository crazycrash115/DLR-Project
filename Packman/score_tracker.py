import gymnasium as gymn

class RawScoreTracker(gymn.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._score = 0.0

    def reset(self, **kwargs):
        self._score = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._score += float(reward) 
        info["ale_score"] = self._score
        if terminated or truncated:
            info["episode_ale_score"] = self._score
        return obs, reward, terminated, truncated, info
