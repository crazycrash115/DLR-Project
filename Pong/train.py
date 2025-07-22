
"""
pip install gym[atari] autorom
AutoROM --accept-license
"""
import os
import numpy as np
import gym
from wrapper import PongHitRewardWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import AutoSaveCallback
from gym.wrappers import AtariPreprocessing, FrameStack
np.set_printoptions(suppress=True)

# === Parallel Environment Setup ===
def make_env():
    def _init():
        env = gym.make("PongNoFrameskip-v4")
        #env = PongHitRewardWrapper(env)
        env = Monitor(env)  
        env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
        env = FrameStack(env, num_stack=4)
        return env
    return _init

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    NUM_ENVS = 16 # probably could do way more since this is pong lol
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")

    CHECKPOINT_DIR = "./checkpoints"
    LATEST_MODEL_PATH = "CNN_pong_latest"
    FINAL_MODEL_PATH = "CNN_pong_final"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = None
    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("Resumed from latest autosave")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=1024,
            tensorboard_log="./ppo_pong_logs",
            policy_kwargs=dict(normalize_images=False)
        )
        print("Starting training from scratch")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CHECKPOINT_DIR,
        name_prefix="pong_CNN"
    )

    autosave_callback = AutoSaveCallback(
        save_path=LATEST_MODEL_PATH,
        save_freq=2048,
        verbose=1
    )

    model.learn(
        total_timesteps=10_000_000,
        callback=[checkpoint_callback, autosave_callback]
    )

    model.save(FINAL_MODEL_PATH)
    print("Final Pong model saved.")
