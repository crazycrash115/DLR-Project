import os
import numpy as np
import gymnasium as gym
import gym_snake
from gymnasium.wrappers import TimeLimit

from wrapper import GymV21toGymnasium, SnakeRewardWrapper, SnakeActionListWrapper
from observation import SnakeObservationWrapper

N_EPISODES = 10
RENDER = True

def make_env():
    try:
        from gym_snake.envs.snake_env import SnakeEnv
    except ImportError:
        from gym_snake.envs.snake import SnakeEnv
    env = SnakeEnv()
    env = GymV21toGymnasium(env)
    env.n_foods = 1
    env.random_init = True
    env = SnakeRewardWrapper(env)
    env = SnakeObservationWrapper(env)
    env = SnakeActionListWrapper(env)
    env = TimeLimit(env, max_episode_steps=2000)
    return env

if __name__ == "__main__":
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

    NAME = f"{ALG}_snake_MLP"
    MODEL_PATH = f"./{NAME}_latest"

    env = make_env()
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)

    assert os.path.isfile(f"{MODEL_PATH}.zip") or os.path.isfile(MODEL_PATH), "model not found"
    model = ALG_CLASS.load(MODEL_PATH, env=env, device="cpu")
    print(f"Loaded model: {MODEL_PATH}")

    episode_rewards = []
    for ep in range(1, N_EPISODES + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            if RENDER:
                env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
        episode_rewards.append(ep_reward)
        print(f"Episode {ep:02d} — reward: {ep_reward:.1f}")

    env.close()
