import os
import numpy as np
import gymnasium as gymn                   # Gymnasium only
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from callbacks import AutoSaveCallback
from wrapper import AddChannelWrapper      # channel-first (1,84,84)
from gymnasium.wrappers import AtariPreprocessing

np.set_printoptions(suppress=True)

# === Parallel Environment Setup ===
def make_env():
    def _init():
        import ale_py           
        env = gymn.make(
            "ALE/MsPacman-v5",
            render_mode=None,
            frameskip=1,
            repeat_action_probability=0.0,
        )


        # Atari preprocessing: 84x84 grayscale uint8 + frame_skip=4
        env = AtariPreprocessing(
            env,
            grayscale_obs=True,
            scale_obs=False,
            frame_skip=4,
            screen_size=84
        )

        env = AddChannelWrapper(env)  # (1,84,84)
        env = Monitor(env)
        return env
    return _init

# === (Required for WINDOWS multiprocessing) ===
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # === Env Creation ===
    NUM_ENVS = 16
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    env = VecFrameStack(env, n_stack=4, channels_order="first")  # (4,84,84)

    # === Paths ===
    CHECKPOINT_DIR = "./checkpoints"
    LATEST_MODEL_PATH = "CNN_pacman_latest"
    FINAL_MODEL_PATH  = "CNN_pacman_final"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === Load or Create Model ===
    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("Resumed from latest autosave")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=1024,
            tensorboard_log="./ppo_pacman_logs",
            policy_kwargs=dict(normalize_images=False)
        )
        print("Starting training from scratch")

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="pacman_CNN"
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
    print("Final Pac-Man model saved.")
