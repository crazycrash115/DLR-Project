
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from callbacks import AutoSaveCallback
from gym.wrappers import TimeLimit
from env import CaveGameEnv  
import os
np.set_printoptions(suppress=True)

# === Parallel Environment Setup ===
def make_env():
    def _init():
        import os
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no video 
        os.environ["SDL_AUDIODRIVER"] = "dummy" # mute 

        #nuke pygame.mixer so it can't come back
        import pygame

        class _Noop:
            def __getattr__(self, _): return self
            def __call__(self, *a, **k): return self

        try:
            pygame.mixer.quit()
        except Exception:
            pass
        pygame.mixer = _Noop()   

        from env import CaveGameEnv
        env = CaveGameEnv()
        env = TimeLimit(env, max_episode_steps=2000)   # No idea if this is a good time or not will test
        return Monitor(env)
    return _init

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # === Env Creation ===
    NUM_ENVS = 4  # go higher if stable (so far not really)
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    env = VecMonitor(env)   

    # === Paths ===
    CHECKPOINT_DIR = "./checkpoints"
    LATEST_MODEL_PATH = "CNN_CaveGame_latest"
    FINAL_MODEL_PATH = "CNN_CaveGame_final"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === Load or Create Model ===
    model = None
    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("Resumed from latest autosave")
    else:
        # --- FOR CNNPolicy ---
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=1024,
            tensorboard_log="./ppo_CaveGame_logs",
            policy_kwargs=dict(normalize_images=False)  
        )
        print("Starting training from scratch")

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="CaveGame_CNN"
    )

    autosave_callback = AutoSaveCallback(
        save_path=LATEST_MODEL_PATH,
        save_freq=2048,
        verbose=1
    )

    # === Train ===
    model.learn(
        total_timesteps=1_000_000_000,
        callback=[checkpoint_callback, autosave_callback],
        log_interval=1
    ) 

    # === Save Final Model ===
    model.save(FINAL_MODEL_PATH)
    print("Final CaveGame model saved.")
