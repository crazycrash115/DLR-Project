import os
import numpy as np
import gymnasium as gymn
from wrapper import *

while True:
    ALG = input("Would you like PPO or A2C? ").strip().upper()
    if ALG == "PPO":
        from stable_baselines3 import PPO as ALG_CLASS
        break
    elif ALG == "A2C":
        from stable_baselines3 import A2C as ALG_CLASS
        break
    else:
        print("Invalid choice. Please type 'PPO' or 'A2C'.")

while True:
    MODE = input("Use 'Base' or 'Surv' model? ").strip().capitalize()
    if MODE in ("Base", "Surv"):
        break
    else:
        print("Invalid choice. Please type 'Base' or 'Surv'.")

NAME = f"{ALG}_pacman_MLP_latest_{MODE}"
MODEL_PATH = f"./{NAME}"
N_EPISODES = 10
RENDER = True

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
        survive_bonus=0.0,
        no_score_patience=999,
        no_score_penalty=0.0,
        death_penalty=3.0,
        pos_boost=0.0,
        combo_window=0,
        combo_step_bonus=0.0,
        power_trigger_min=float("inf"),
        power_steps=0,
        power_step_bonus=0.0,
        ghost_threshold=float("inf"),
        ghost_mult=0.0,
        max_steps=None,
    )
    return env

env = make_env()
print("Obs space:", env.observation_space)
print("Act space:", env.action_space)

candidate_paths = [MODEL_PATH, f"{MODEL_PATH}.zip"]
model_file = next((p for p in candidate_paths if os.path.isfile(p)), None)
if model_file is None:
    raise FileNotFoundError(f"Model not found. Tried: {candidate_paths}")

model = ALG_CLASS.load(model_file, env=env, device="cpu")
print(f"Loaded model: {model_file}")

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
