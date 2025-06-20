import os
import gym_super_mario_bros
from wrappers import MarioRewardWrapper, ActionRepeatWrapper 
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np
import gym
import re
from gym.wrappers import ResizeObservation #resizes the screen for less ram use 
from callbacks import AutoSaveCallback

np.set_printoptions(suppress=True)  # disables scientific notation

CUSTOM_MOVEMENT = [
    ['NOOP'],           # No action
    ['right'],          # Walk right
    ['right', 'A'],     # Jump while walking right
    ['A'],              # Jump in place
    ['left'],           # Walk left
    ['left', 'A'],      # Jump while walking left
    ['B'],              # Run in place
    ['right', 'B'],     # Run right
    ['right', 'B', 'A'],# Run + Jump right 
    ['left', 'B'],      # Run left
    ['left', 'B', 'A'], # Run + Jump left
]

# === Parallel Environment Setup ===
def make_env():
    def _init():
        env = gym_super_mario_bros.make('SuperMarioBros-v3') #the level (i think)
        env._max_lives = 1  # force 1 life (this is unofficial and might not work consistently)

        env = JoypadSpace(env, CUSTOM_MOVEMENT) # my custom movemnet
        env = ActionRepeatWrapper(env, repeat=2) # this is the repeater wrapper
        env = MarioRewardWrapper(env) # applies the wrapper
        env = ResizeObservation(env, (84, 84))  #Resize frame to 84x84
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    num_envs = 16
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # === Paths ===
    CHECKPOINT_DIR = "./checkpoints"
    FINAL_MODEL_PATH = "ppo_mario_final"
    LATEST_MODEL_PATH = "ppo_mario_latest"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === Load Model ===
    model = None
    latest_ckpt = None
    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("Resumed from latest autosave")
    elif any(f.endswith('.zip') and 'mario_ppo' in f for f in os.listdir(CHECKPOINT_DIR)):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip') and 'mario_ppo' in f]
        checkpoints.sort(key=lambda x: int(re.search(r'_(\d+)_steps', x).group(1)))
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        model = PPO.load(latest_ckpt, env)
        print(f"Resumed from checkpoint: {latest_ckpt}")
    else:
        model = PPO("CnnPolicy", env, verbose=1, n_steps=1024, tensorboard_log="./ppo_mario_logs")
        print("Starting training from scratch")

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CHECKPOINT_DIR,
        name_prefix="mario_ppo"
    )
    autosave_callback = AutoSaveCallback(LATEST_MODEL_PATH, save_freq=2048, verbose=1)

    from stable_baselines3.common.logger import configure

    # === Train ===
    model.learn(total_timesteps=1_000_000_000, callback=[checkpoint_callback, autosave_callback])

    # === Save final model ===
    model.save(FINAL_MODEL_PATH)
    print("Final model saved.")
