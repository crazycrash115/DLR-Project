import os
import numpy as np
import gymnasium as gymn
import flappy_bird_gymnasium
import cv2

from stable_baselines3 import PPO
from gymnasium.wrappers import AddRenderObservation, ResizeObservation
from wrapper import *

MODEL_PATH = "./CNN_flappy_latest"
N_EPISODES = 10
RENDER = True

def make_env():
    env = gymn.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = AddRenderObservation(env, render_only=True)
    env = ResizeObservation(env, (84, 84))
    env = AddChannelWrapper(env)                    # (1,84,84)
    env = ChannelFrameStack(env, k=4)               # (4,84,84)
    env = FlappyRewardWrapper(env,                  # same reward as training
                               survive_bonus=0.05,
                               pipe_reward=1.0,
                               gap_weight=0.01,
                               death_penalty=1.0)
    return env

env = make_env()
print("Obs space:", env.observation_space)  # (4, 84, 84)
print("Act space:", env.action_space)

assert os.path.isfile(f"{MODEL_PATH}.zip"), "model not found"
model = PPO.load(MODEL_PATH, env=env, device="cpu")
print(f"Loaded model: {MODEL_PATH}")

episode_rewards = []
for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        if RENDER:
            frame = env.render()
            if frame is not None:
                cv2.imshow("FlappyBird (test)", frame[:, :, ::-1])
                cv2.waitKey(1)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        ep_reward += float(reward)
    episode_rewards.append(ep_reward)
    print(f"Episode {ep:02d} — reward: {ep_reward:.1f}")

env.close()
if RENDER:
    cv2.destroyAllWindows()
