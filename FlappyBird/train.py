import os
import numpy as np
import gym                 # classic gym 0.21 (SB3 1.7 needs it)
import flappy_bird_gymnasium                 
from gym.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import AutoSaveCallback
from wrapper import *
np.set_printoptions(suppress=True)

# === Parallel Environment Setup ===
def make_env():
    def _init():

        import gymnasium as gymn
        from wrapper import GymnasiumToGymV21, GymnasiumActionFix, AddChannelWrapper, FlappyRewardWrapper
        from stable_baselines3.common.monitor import Monitor
        from gym.wrappers import ResizeObservation
        env = gymn.make("FlappyBird-v0", render_mode=None, use_lidar=False)

        # 2) compatibility for SB3 1.x
        env = GymnasiumToGymV21(env)
        env = GymnasiumActionFix(env)
        # env = FlappyRewardWrapper(env)   # put back after shaping rewards are added

        # 3) image preprocessing
        env = ResizeObservation(env, (84, 84))
        env = AddChannelWrapper(env)       #change if not using CNN

        # 4) monitoring
        env = Monitor(env)
        return env
    return _init

# === (Required for **WINDOWS** Multiprocessing) ===
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # === Env Creation ===
    NUM_ENVS = 16
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")

    # === Paths ===
    CHECKPOINT_DIR = "./checkpoints"
    LATEST_MODEL_PATH = "CNN_flappy_latest" #REMEMBER TO CHANGE 
    FINAL_MODEL_PATH = "CNN_flappy_final" #REMEMBER TO CHANGE 
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === Load or Create Model ===
    model = None
    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("Resumed from latest autosave")
    else:
        model = PPO(
            "CnnPolicy", #REMEMBER TO CHANGE 
            env,
            verbose=1,
            n_steps=1024,
            tensorboard_log="./ppo_flappy_logs",
            policy_kwargs=dict(normalize_images=False) # Needed for adding the channel bit
        )
        print("Starting training from scratch")

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="flappy_CNN" #REMEMBER TO CHANGE 
    )

    autosave_callback = AutoSaveCallback(
        save_path=LATEST_MODEL_PATH,
        save_freq=2048,
        verbose=1
    )

    # === Train ===
    model.learn(
        total_timesteps=1_000_000_000,
        callback=[checkpoint_callback, autosave_callback]
    )

    # === Save Final Model ===
    model.save(FINAL_MODEL_PATH)
    print("Final Flappy model saved.")
