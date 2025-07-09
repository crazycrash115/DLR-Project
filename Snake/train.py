import gym
import gym_snake
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from reward import SnakeRewardWrapper
from observation import SnakeObservationWrapper
from callbacks import AutoSaveCallback  
import os
import re

# === Paths ===
CHECKPOINT_DIR = "./checkpoints"
MODEL_PATH = "./MLP_snake_latest"
LOG_DIR = "./logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Env Setup ===
def make_env():
    env = gym.make("snake-v0")
    env.n_foods = 1
    env.random_init = True
    env = SnakeRewardWrapper(env)
    env = SnakeObservationWrapper(env) #
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])  # Vectorized env
env = VecMonitor(env)          # Monitor for vec env

# === Load or Create Model ===
model = None
latest_ckpt = None

if os.path.exists(f"{MODEL_PATH}.zip"):
    model = PPO.load(MODEL_PATH, env)
    print("Resumed from latest autosave")

elif any(f.endswith('.zip') and 'MLP_snake' in f for f in os.listdir(CHECKPOINT_DIR)):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip') and 'MLP_snake' in f]
    checkpoints.sort(key=lambda x: int(re.search(r'_(\d+)_steps', x).group(1)))
    latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    model = PPO.load(latest_ckpt, env)
    print(f"Resumed from checkpoint: {latest_ckpt}")

else:
    model = PPO(
        "MlpPolicy",  #REMEMBER TO CHANGE 
        env, 
        verbose=1, 
        n_steps=1024, 
        tensorboard_log=LOG_DIR)
    print("Starting training from scratch")

# === Callbacks ===
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix="MLP_snake"
)

autosave_callback = AutoSaveCallback(
    save_path=MODEL_PATH,
    save_freq=2048,
    verbose=1
)

# === Train ===
model.learn(total_timesteps=1_024_000, callback=[checkpoint_callback, autosave_callback])
model.save("./snake/MLP_snake_final")
