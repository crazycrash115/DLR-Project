import os
import numpy as np
import gymnasium as gymn
import multiprocessing
from score_tracker import RawScoreTracker

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from wrapper import *
from callbacks import AutoSaveCallback
import csv

class EpisodeScoreCSVCallback(BaseCallback):
    def __init__(self, log_dir):
        super().__init__()
        self.path = os.path.join(log_dir, "episode_scores.csv")
        self.header_written = False
    def _on_training_start(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["t", "r", "l", "episode_ale_score"])
            self.header_written = True
    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                with open(self.path, "a", newline="") as f:
                    csv.writer(f).writerow([int(self.num_timesteps), ep["r"], ep["l"], info.get("episode_ale_score")])
        return True

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
        env = RawScoreTracker(env)
        env = Controls(env)
        env = FrameStack(env, num_stack=4)
        env = ScaleWrapper(env, scale=True)
        env = PacmanRewardWrapper(
            env,
            base_scale=0.05,
            survive_bonus=0.0,
            no_score_patience=999,   
            no_score_penalty=0.0,   
            death_penalty=3.0,
            pos_boost=0.0,
            combo_window=0,
            combo_step_bonus=0.0,
            power_trigger_min=float("inf"),
            power_steps=0,
            power_step_bonus=0.0,
            ghost_threshold=float("inf"),
            ghost_mult=0.0,
            max_steps=None,
        )
        return env
    return _init

def build_model(algo_class, env, log_dir, latest_path, alg_name):
    if os.path.exists(f"{latest_path}.zip"):
        print(f"Resuming {alg_name} from {latest_path}.zip")
        return algo_class.load(latest_path, env=env, device="auto")
    return algo_class(
        "MlpPolicy",
        env,
        device="auto",
        verbose=1,
        tensorboard_log=log_dir,
    )

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
    score_csv = EpisodeScoreCSVCallback(LOG_DIR)

    model.learn(total_timesteps=50_000_000, callback=[ckpt, autosave, score_csv])
    model.save(FINAL)
    print(f"Final {ALG} Pac-Man model saved to {FINAL}.")
