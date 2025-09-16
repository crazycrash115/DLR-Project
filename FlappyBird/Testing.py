import os
import numpy as np
import gymnasium as gymn
import flappy_bird_gymnasium

from stable_baselines3 import PPO
from gymnasium.wrappers import ResizeObservation
from wrapper import AddChannelWrapper

np.set_printoptions(suppress=True)

MODEL_PATH = "CNN_flappy_latest.zip"  # Change if needed
N_EPISODES = 10
RENDER     = True

def make_test_env():
    env = gymn.make(
        "FlappyBird-v0",
        render_mode="human" if RENDER else None,
        use_lidar=False
    )
    env = ResizeObservation(env, (84, 84))
    env = AddChannelWrapper(env)
    return env

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
    print(f"Episode {ep:02d} — reward: {ep_reward:.1f}")

mean_r = np.mean(episode_rewards)
std_r  = np.std(episode_rewards)
print(f"\n=== finished {N_EPISODES} episodes ===")
print(f"mean reward: {mean_r:.2f} ± {std_r:.2f}")

env.close()
