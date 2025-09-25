import os
import numpy as np
import gymnasium as gymn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from wrapper import Controls, FrameStack, ScaleWrapper, PacmanRewardWrapper
from callbacks import AutoSaveCallback

np.set_printoptions(suppress=True)

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./ppo_pacman_logs"
LATEST = "MLP_pacman_latest"
FINAL  = "MLP_pacman_final"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env():
    def _init():
        import ale_py  # ensure ale-py is importable in each subproc
        env = gymn.make(
            "ALE/MsPacman-v5",
            obs_type="ram",                 # **THIS** is the RAM flag
            render_mode=None,
            frameskip=4,
            repeat_action_probability=0.0,
        )
        env = Controls(env)                 # Discrete(4)
        env = FrameStack(env, num_stack=4)  # (128,) -> (512,)
        env = ScaleWrapper(env, scale=True) # float32 [0,1]
        env = PacmanRewardWrapper(
            env,
            base_scale=0.02,
            survive_bonus=0.02,
            no_score_patience=60,
            no_score_penalty=0.0,           # set >0 only if you really want anti-stall
            death_penalty=10.0,
        )
        return Monitor(env)
    return _init

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Windows tip: start with fewer envs if you see spawn/multiproc flakiness.
    NUM_ENVS = 8
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)], start_method="spawn")
    env = VecMonitor(env, LOG_DIR)

    if os.path.exists(f"{LATEST}.zip"):
        model = PPO.load(LATEST, env=env, device="auto")
        print("Resumed from latest autosave")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )
        print("Starting training from scratch (MLP/RAM)")

    ckpt = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="pacman_MLP")
    autosave = AutoSaveCallback(save_path=LATEST, save_freq=2048, verbose=1)

    model.learn(total_timesteps=50_000_000, callback=[ckpt, autosave])
    model.save(FINAL)
    print("Final Pac-Man model saved.")
