import os
import numpy as np
import gymnasium as gymn
import multiprocessing

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from wrapper import Controls, FrameStack, ScaleWrapper, PacmanRewardWrapper, OneLifeWrapper
from callbacks import AutoSaveCallback

def make_env():
    def _init():
        import ale_py
        env = gymn.make(
            "ALE/MsPacman-v5",
            obs_type="ram",
            render_mode=None,
            frameskip=4,
            repeat_action_probability=0.0,
        )
        env = Controls(env)
        env = FrameStack(env, num_stack=4)
        env = ScaleWrapper(env, scale=True)
        env = PacmanRewardWrapper(
            env,
            base_scale=0.1,
            survive_bonus=0.01,
            no_score_patience=20,
            no_score_penalty=0.1,
            death_penalty=5.0,
        )
        return Monitor(env)
    return _init

def build_model(algo_class, env, log_dir, latest_path, alg_name):
    if os.path.exists(f"{latest_path}.zip"):
        print(f"Resuming {alg_name} from {latest_path}.zip")
        return algo_class.load(latest_path, env=env, device="auto")
    return algo_class("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    while True:
        ALG = input("Would you like PPO or A2C? ").strip().upper()
        if ALG == "PPO":
            print("Using PPO.")
            from stable_baselines3 import PPO as ALG_CLASS
            break
        elif ALG == "A2C":
            print("Using A2C.")
            from stable_baselines3 import A2C as ALG_CLASS
            break
        else:
            print("Invalid choice. Please type 'PPO' or 'A2C'.")

    NUM_ENVS = 16
    LOG_DIR = f"./{ALG.lower()}_pacman_logs"
    CHECKPOINT_DIR = "./checkpoints"
    NAME = f"{ALG}_pacman_MLP"
    LATEST = f"{NAME}_latest"
    FINAL  = f"{NAME}_final"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    np.set_printoptions(suppress=True)

    vec = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    vec = VecMonitor(vec, LOG_DIR)

    model = build_model(ALG_CLASS, vec, LOG_DIR, LATEST, ALG)

    ckpt = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix=f"pacman_{ALG}_MLP")
    autosave = AutoSaveCallback(save_path=LATEST, save_freq=2048, verbose=1)

    model.learn(total_timesteps=50_000_000, callback=[ckpt, autosave])
    model.save(FINAL)
    print(f"Final {ALG} Pac-Man model saved to {FINAL}.")
