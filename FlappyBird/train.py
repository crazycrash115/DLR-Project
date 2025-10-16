import os
import numpy as np
import gymnasium as gymn
import flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation, ResizeObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from callbacks import AutoSaveCallback
from wrapper import *

np.set_printoptions(suppress=True)

def make_env():
    def _init():
        env = gymn.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
        env = AddRenderObservation(env, render_only=True)
        env = ResizeObservation(env, (84, 84))
        env = AddChannelWrapper(env)             
        env = FlappyRewardWrapper(env, 
                                  gamma=0.99, 
                                  gap_weight=0.5, 
                                  extra_pipe_bonus=0.0)

        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    NUM_ENVS = 4  # keep small on CPU/Windows
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    env = VecFrameStack(env, n_stack=4, channels_order="first")  # (4,84,84)

    CHECKPOINT_DIR = "./checkpoints"
    LATEST_MODEL_PATH = "CNN_flappy_latest"
    FINAL_MODEL_PATH  = "CNN_flappy_final"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("Resumed from latest autosave")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_flappy_logs",
        )
        print("Starting training from scratch")

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="flappy_CNN",
    )
    autosave_callback = AutoSaveCallback(
        save_path=LATEST_MODEL_PATH,
        save_freq=2048,
        verbose=1,
    )

    model.learn(
        total_timesteps=1_000_000_000,
        callback=[checkpoint_callback, autosave_callback],
    )

    model.save(FINAL_MODEL_PATH)
    print("Final Flappy model saved.")
