import os
import numpy as np
import gym                                    
import gymnasium as gymn                      
import flappy_bird_gymnasium                 
from wrapper import GymnasiumToGymV21, GymnasiumActionFix, AddChannelWrapper
from gym.wrappers import ResizeObservation
from stable_baselines3 import PPO

np.set_printoptions(suppress=True)

# === config ===
MODEL_PATH   = "CNN_flappy_latest.zip"        # Change 
N_EPISODES   = 10
RENDER       = True                          

# === build single test env ===
def make_test_env():
    env = gymn.make("FlappyBird-v0",
                    render_mode="human" if RENDER else None,
                    use_lidar=False)

    env = GymnasiumToGymV21(env)
    env = GymnasiumActionFix(env)
    env = ResizeObservation(env, (84, 84))
    env = AddChannelWrapper(env)
    return env

env = make_test_env()

# === load model ===
assert os.path.exists(MODEL_PATH), f"model not found: {MODEL_PATH}"
model = PPO.load(MODEL_PATH, env=env, device="cpu")
print(f"Loaded model: {MODEL_PATH}")

# === evaluation loop ===
episode_rewards = []

for ep in range(1, N_EPISODES + 1):
    obs = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward

    episode_rewards.append(ep_reward)
    print(f"Episode {ep:02d} ─ reward: {ep_reward:.1f}")

mean_r = np.mean(episode_rewards)
std_r  = np.std(episode_rewards)
print(f"\n=== finished {N_EPISODES} episodes ===")
print(f"mean reward: {mean_r:.2f} ± {std_r:.2f}")

env.close()