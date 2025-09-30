import os
import numpy as np
import gymnasium as gymn
from stable_baselines3 import PPO
from wrapper import Controls, FrameStack, ScaleWrapper, PacmanRewardWrapper

np.set_printoptions(suppress=True)

MODEL_PATH = "MLP_pacman_latest.zip"
N_EPISODES = 10
RENDER = True

def make_test_env():
    import ale_py
    env = gymn.make(
        "ALE/MsPacman-v5",
        obs_type="ram",
        render_mode="human" if RENDER else None,
        frameskip=4,
        repeat_action_probability=0.0,
    )
    env = Controls(env)
    env = FrameStack(env, num_stack=4)
    env = ScaleWrapper(env, scale=True)
    env = PacmanRewardWrapper(
            env,
            base_scale=0.05,
            survive_bonus=0.01,
            no_score_patience=40,
            no_score_penalty=0.01,           # set >0 only if you really want anti-stall
            death_penalty=10.0,
        )
    return env

env = make_test_env()
print("Obs space:", env.observation_space)  
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
        if RENDER:
            env.render()
        done = terminated or truncated
        ep_reward += float(reward)
    episode_rewards.append(ep_reward)
    print(f"Episode {ep:02d} — reward: {ep_reward:.1f}")

mean_r = np.mean(episode_rewards)
std_r  = np.std(episode_rewards)
print(f"\n=== finished {N_EPISODES} episodes ===")
print(f"mean reward: {mean_r:.2f} ± {std_r:.2f}")
env.close()
