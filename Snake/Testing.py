import gymnasium as gym
import gym_snake
from stable_baselines3 import PPO
from wrapper import SnakeRewardWrapper 
from wrapper import GymV21toGymnasium
from observation import SnakeObservationWrapper

def main():
    # === Load and wrap env ===
    try:
        from gym_snake.envs.snake_env import SnakeEnv
    except ImportError:
        from gym_snake.envs.snake import SnakeEnv

    env = SnakeEnv()
    env = GymV21toGymnasium(env)
    env.n_foods = 1
    env.random_init = True
    env = SnakeRewardWrapper(env)      ### reward wrapper
    env = SnakeObservationWrapper(env) ### dont forget to remove if swapping

    # === Load trained model ===
    model = PPO.load("./MLP_snake_latest")  # REMEMBA TO CHANG PATH

    obs, info = env.reset(seed=0)
    done = False
    total_reward = 0

    while True:
        env.render() # show the game

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done:
            print(f"Episode finished. Score: {total_reward}")
            total_reward = 0
            obs, info = env.reset()

if __name__ == "__main__":
    main()
