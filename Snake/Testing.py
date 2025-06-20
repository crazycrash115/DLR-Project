import gym
import gym_snake
from stable_baselines3 import PPO
from reward import SnakeRewardWrapper
from observation import SnakeObservationWrapper
def main():
    # === Load and wrap env ===
    env = gym.make("snake-v0")
    env.n_foods = 1
    env.random_init = True
    env = SnakeRewardWrapper(env)      ### reward wrapper
    env = SnakeObservationWrapper(env) ### dont forget to remove if swapping


    # === Load trained model ===
    model = PPO.load("./MLP_snake_latest")  # REMEMBA TO CHANG PATH

    obs = env.reset()
    done = False
    total_reward = 0

    while True:
        env.render() # show the game

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step([action]) 
        total_reward += reward

        if done:
            print(f"Episode finished. Score: {total_reward}")
            total_reward = 0
            obs = env.reset()

if __name__ == "__main__":
    main()
