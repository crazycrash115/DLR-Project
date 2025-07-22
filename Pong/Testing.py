# === test.py ===
import os
import numpy as np
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

np.set_printoptions(suppress=True)

MODEL_PATH = "CNN_pong_latest.zip"
N_EPISODES = 10
RENDER = True

def make_test_env():
    env = gym.make("PongNoFrameskip-v4", render_mode="human" if RENDER else None)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, num_stack=4)
    return env
from stable_baselines3.common.vec_env import DummyVecEnv

def make_test_env():
    def _init():
        env = gym.make("PongNoFrameskip-v4", render_mode="human" if RENDER else None)
        env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
        env = FrameStack(env, num_stack=4)
        return env

    return DummyVecEnv([_init])

env = make_test_env()
assert os.path.exists(MODEL_PATH), f"model not found: {MODEL_PATH}"
model = PPO.load(MODEL_PATH, env=env, device="cpu")
print(f"Loaded model: {MODEL_PATH}")

episode_rewards = []
for ep in range(1, N_EPISODES + 1):
    obs = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        ep_reward += reward[0]  
        done = done[0]          

    episode_rewards.append(ep_reward)
    print(f"Episode {ep:02d} ─ reward: {ep_reward:.1f}")

mean_r = np.mean(episode_rewards)
std_r = np.std(episode_rewards)
print(f"\n=== finished {N_EPISODES} episodes ===")
print(f"mean reward: {mean_r:.2f} ± {std_r:.2f}")

env.close()
