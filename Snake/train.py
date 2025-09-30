import os
import gymnasium as gym
import gym_snake

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from wrapper import GymV21toGymnasium, SnakeActionListWrapper, SnakeRewardWrapper
from observation import SnakeObservationWrapper
from callbacks import AutoSaveCallback  

# === Paths ===
CHECKPOINT_DIR = "./checkpoints"
MODEL_PATH     = "./MLP_snake_latest"
LOG_DIR        = "./logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Parallel Env  ===
def make_env():
    def _init():
        try:
            from gym_snake.envs.snake_env import SnakeEnv
        except ImportError:
            from gym_snake.envs.snake import SnakeEnv

        env = SnakeEnv()                    
        env = GymV21toGymnasium(env)
        env = SnakeActionListWrapper(env)     

        # base config
        env.n_foods = 1
        env.random_init = True

        env = SnakeRewardWrapper(env)
        env = SnakeObservationWrapper(env)
        return env
    return _init

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    NUM_ENVS = 16  

  
    venv = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    venv = VecMonitor(venv, LOG_DIR)

    # Load or create model 
    if os.path.exists(f"{MODEL_PATH}.zip"):
        model = PPO.load(MODEL_PATH, env=venv, device="auto")
        print("Resumed from latest autosave")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )
        print("Starting training from scratch")

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,  
        save_path=CHECKPOINT_DIR,
        name_prefix="MLP_snake",
    )
    autosave_cb = AutoSaveCallback(save_path=MODEL_PATH, save_freq=2048, verbose=1)

    # Train
    model.learn(total_timesteps=100_024_000, callback=[checkpoint_cb, autosave_cb])

    # Save
    os.makedirs("./snake", exist_ok=True)
    model.save("./snake/MLP_snake_final")

    venv.close()
