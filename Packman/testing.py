import os
import numpy as np
import gymnasium as gymn
from stable_baselines3 import PPO

from wrapper import Controls, FrameStack, ScaleWrapper, PacmanRewardWrapper

MODEL_PATH   = "./MLP_pacman_latest"
N_EPISODES   = 10
RENDER       = True

def make_env():
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
        no_score_penalty=0.01,
        death_penalty=10.0,
    )
    return env

env = make_env()
print("Obs space:", env.observation_space)
print("Act space:", env.action_space)

assert os.path.isfile(f"{MODEL_PATH}.zip") or os.path.isfile(MODEL_PATH), "model not found"
model = PPO.load(MODEL_PATH, env=env, device="cpu")
print(f"Loaded model: {MODEL_PATH}")

episode_rewards = []
for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        if RENDER:
            env.render()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        ep_reward += float(reward)

    episode_rewards.append(ep_reward)
    print(f"Episode {ep:02d} — reward: {ep_reward:.1f}")

env.close()
