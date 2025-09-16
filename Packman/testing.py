
import os
import numpy as np
import gymnasium as gymn
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import PPO
from wrapper import AddChannelWrapper
from collections import deque
from gymnasium import spaces
import ale_py
np.set_printoptions(suppress=True)

MODEL_PATH = "CNN_pacman_latest.zip"
N_EPISODES = 10
RENDER = True



class ChannelFirstFrameStack(gymn.Wrapper):
    """Stack last n frames along channel axis. Expects (C,H,W) uint8 in, returns (n*C,H,W)."""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        c, h, w = env.observation_space.shape
        self.frames = deque(maxlen=n_stack)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c * n_stack, h, w), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), r, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

def make_test_env():
    env = gymn.make(
        "ALE/MsPacman-v5",
        render_mode="human" if RENDER else None,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=False,
        frame_skip=4,
        screen_size=84,
    )
    env = AddChannelWrapper(env)                 # (1,84,84)
    env = ChannelFirstFrameStack(env, n_stack=4) # (4,84,84)
    return env

def main():
    env = make_test_env()

    assert os.path.exists(MODEL_PATH), f"model not found: {MODEL_PATH}"
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
    print(f"Loaded model: {MODEL_PATH}")

    episode_rewards = []
    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += float(reward)
        episode_rewards.append(ep_reward)
        print(f"Episode {ep:02d} - reward: {ep_reward:.1f}")

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)
    print(f"\nfinished {N_EPISODES} episodes")
    print(f"mean reward: {mean_r:.2f} ± {std_r:.2f}")

    env.close()

if __name__ == "__main__":
    main()
