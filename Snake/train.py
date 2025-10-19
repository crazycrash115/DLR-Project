import os
import gymnasium as gym
import gym_snake
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from callbacks import AutoSaveCallback
from wrapper import GymV21toGymnasium, SnakeActionListWrapper, SnakeRewardWrapper
from observation import SnakeObservationWrapper
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_mean = -float("inf")
recent_rewards = []
SAVE_PATH = None

def update_best(info_list, model, save_path):
    global best_mean, recent_rewards
    for info in info_list:
        ep = info.get("episode")
        if ep is not None:
            recent_rewards.append(float(ep["r"]))
            if len(recent_rewards) > 200:
                recent_rewards.pop(0)
            m = sum(recent_rewards) / len(recent_rewards)
            if m > best_mean:
                best_mean = m
                model.save(save_path)

class GlobalBestCallback(BaseCallback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
    def _on_step(self):
        update_best(self.locals.get("infos", []), self.model, self.save_path)
        return True

def make_env():
    def _init():
        
        env = SnakeEnv()
        env = GymV21toGymnasium(env)
        env = SnakeActionListWrapper(env)

        env.n_foods = 1
        env.random_init = True
        env = SnakeRewardWrapper(env)
        env = SnakeObservationWrapper(env)
        return env
    return _init

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

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

    NUM_ENVS = 16
    LOG_DIR = f"./{ALG.lower()}_snake_logs"
    NAME = f"{ALG}_snake_MLP"
    MODEL_LATEST = f"./{NAME}_latest"
    FINAL_PATH = f"./snake/{NAME}_final"
    SAVE_PATH = f"./snake/{NAME}_best"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("./snake", exist_ok=True)
    np.set_printoptions(suppress=True)

    venv = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    venv = VecMonitor(venv, LOG_DIR)

    if os.path.exists(f"{MODEL_LATEST}.zip"):
        model = ALG_CLASS.load(MODEL_LATEST, env=venv, device="auto")
        print("Resumed from latest autosave")
    else:
        model = ALG_CLASS(
            "MlpPolicy",
            venv,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )
        print("Starting training from scratch")

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix=NAME,
    )
    autosave_cb = AutoSaveCallback(save_path=MODEL_LATEST, save_freq=2048, verbose=1)
    best_cb = GlobalBestCallback(save_path=SAVE_PATH)

    model.learn(total_timesteps=100_024_000, callback=[checkpoint_cb, autosave_cb, best_cb])

    model.save(FINAL_PATH)
    venv.close()
