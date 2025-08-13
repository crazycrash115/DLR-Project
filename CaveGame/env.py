import gym
import numpy as np
import pygame
import cv2  
from gym import spaces
from src.game.game import Game

class CaveGameEnv(gym.Env):
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_mode((1280, 720))  # game resolution

        self.game = Game("QWERTY")
        self.done = False

        # --- reward shaping state/params (added) ---
        self.screen_h = 720
        self.prev_y = None          # last frame's y (pixels)
        self.best_y = None          # best (min) y achieved so far
        self.eps_px = 2             # deadband to ignore 1px jitters
        self.up_bonus_scale = 2.0   # tiny per-step nudge for moving up
        self.progress_scale = 50.0  # one-time bonus when beating best height

        # --- FOR CNNPolicy ---
        self.observation_space = spaces.Box(0, 255, shape=(1, 84, 84), dtype=np.uint8)

        # --- FOR MlpPolicy ---
        # Uncomment below and comment the line above if switching to MlpPolicy
        # self.observation_space = spaces.Box(low=0, high=10000, shape=(3,), dtype=np.float32)

        # === ACTION SPACE ===
        self.action_space = spaces.Discrete(5)  # 0: nothing 1: left 2: right 3: fire 4: (un?)fire

    def reset(self):
        self.game = Game("QWERTY")
        self.done = False
        # init shaping trackers (added)
        y0 = self.game._player.rect.centery
        self.prev_y = y0
        self.best_y = y0
        return self._get_obs()

    def step(self, action):
        self._apply_action(action)

        self.game._global_timer.update()
        self.game._update()

        obs = self._get_obs()
        reward = self._get_reward()
        self.done = not self.game._player.alive

        return obs, reward, self.done, {}

    def _apply_action(self, action):
        vel = np.array([0.0, 0.0])
        if action == 1:
            vel += np.array([-5.0, 0])
        elif action == 2:
            vel += np.array([5.0, 0])
        elif action == 3:
            self.game._grapple.fire(self.game._cave_map, self.game._camera.offset)
        elif action == 4:
            self.game._grapple.unfire()
        self.game._player.update(vel, self.game._cave_map, self.game._global_timer.delta)

    def _get_obs(self):
        # === OBS RETURN ===
        # --- FOR CNNPolicy ---
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, axis=0)  
        return frame.astype(np.uint8)

        # --- FOR MlpPolicy ---
        # pos = self.game._player.world_position
        # return np.array([pos[0], pos[1], self.game._lava.y], dtype=np.float32)

    def _get_reward(self):
        if not self.game._player.alive:
            return -100.0

        y = float(self.game._player.rect.centery)

        if y <= 0:
            # reached the top – big payout (no extra shaping)
            self.prev_y = y
            if self.best_y is None or y < self.best_y:
                self.best_y = y
            return 100.0

        # base step reward
        reward = 1.0

        # small per-step nudge for moving upward (no punish for falling)
        if self.prev_y is not None:
            delta_up = max(0.0, self.prev_y - y)  # pixels moved up this step
            reward += self.up_bonus_scale * (delta_up / self.screen_h)

        # progress bonus only when beating previous best height (ignores tiny jumps or at least should)
        if self.best_y is None:
            self.best_y = y
        if y + self.eps_px < self.best_y:
            progress_px = (self.best_y - y) - self.eps_px
            if progress_px > 0:
                reward += self.progress_scale * (progress_px / self.screen_h)
            self.best_y = y  # update best-so-far

        self.prev_y = y
        return reward

    def render(self, mode="human"):
        self.game._display()

    def close(self):
        pygame.quit()
