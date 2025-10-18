import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Controls(gym.ActionWrapper):
    """
    Discrete(4): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Mapped to ALE's action indices via get_action_meanings().
    """
    def __init__(self, env):
        super().__init__(env)
        meanings = self.unwrapped.get_action_meanings()
        want = {"UP": None, "DOWN": None, "LEFT": None, "RIGHT": None}
        for i, m in enumerate(meanings):
            mm = m.upper()
            if mm in want and want[mm] is None:
                want[mm] = i
        missing = [k for k, v in want.items() if v is None]
        if missing:
            raise RuntimeError(f"Missing actions {missing} in {meanings}")
        self._map = np.array(
            [want["UP"], want["DOWN"], want["LEFT"], want["RIGHT"]],
            dtype=np.int64
        )
        self.action_space = spaces.Discrete(4)

    def action(self, a):
        return int(self._map[int(a)])
    
class RawScoreTracker(gym.Wrapper):
    """
    Accumulates the *raw* environment reward (ALE score deltas) per episode
    and exposes it via info["ale_score"] each step, and
    info["episode_ale_score"] on terminal/truncated steps.
    """
    def __init__(self, env):
        super().__init__(env)
        self._score = 0.0

    def reset(self, **kwargs):
        self._score = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # In ALE, reward == change in game score for the step
        self._score += float(reward)
        info["ale_score"] = self._score
        if terminated or truncated:
            info["episode_ale_score"] = self._score
        return obs, reward, terminated, truncated, info


class ScaleWrapper(gym.ObservationWrapper):
    """(N,) uint8 RAM -> float32 in [0,1] (keeps shape)."""
    def __init__(self, env, scale=True):
        super().__init__(env)
        self.scale = bool(scale)
        n = int(env.observation_space.shape[0])
        low, high = (0.0, 1.0) if self.scale else (0, 255)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(n,),
            dtype=np.float32 if self.scale else np.uint8
        )

    def observation(self, obs):
        arr = np.asarray(obs)
        return (arr.astype(np.float32) / 255.0) if self.scale else arr


class FrameStack(gym.ObservationWrapper):
    """Stack k RAM vectors: (N,) -> (N*k,)."""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.k = int(num_stack)
        n = int(env.observation_space.shape[0])
        self._buf = np.zeros((self.k, n), dtype=env.observation_space.dtype)
        low = np.repeat(env.observation_space.low, self.k)
        high = np.repeat(env.observation_space.high, self.k)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf[...] = obs
        return self._buf.reshape(-1), info

    def observation(self, obs):
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = obs
        return self._buf.reshape(-1)


class PacmanRewardWrapper(gym.Wrapper):
    """
    r = base_scale*raw
        + survive_bonus
        + pos_boost (when raw_r>0)
        + combo_bonus (clustered scoring; only if combo_window>0 and combo_step_bonus>0)
        + power-mode bonuses (only if power_steps>0 and power_trigger_min finite)
        - no_score_penalty (after patience)
        - death_penalty on life loss
    Adds info['raw_return'] at episode end.
    """
    def __init__(self, env,
                 base_scale=0.02,
                 survive_bonus=0.02,
                 no_score_patience=60,
                 no_score_penalty=0.05,
                 death_penalty=10.0,
                 # positive score shaping
                 pos_boost=0.02,
                 # combo shaping (set window<=0 or bonus<=0 to disable)
                 combo_window=20,
                 combo_step_bonus=0.01,
                 # power mode heuristic (set power_steps<=0 or trigger huge to disable)
                 power_trigger_min=40.0,   # big dot/fruit likely
                 power_steps=180,          # adjust for frameskip
                 power_step_bonus=0.003,
                 ghost_threshold=150.0,    # large spike ≈ ghost eat
                 ghost_mult=0.02,
                 max_steps=None):
        super().__init__(env)
        self.base_scale = float(base_scale)
        self.survive_bonus = float(survive_bonus)
        self.no_score_patience = int(no_score_patience)
        self.no_score_penalty = float(no_score_penalty)
        self.death_penalty = float(death_penalty)

        self.pos_boost = float(pos_boost)

        self.combo_window = int(combo_window)
        self.combo_step_bonus = float(combo_step_bonus)

        self.power_trigger_min = float(power_trigger_min)
        self.power_steps = int(power_steps)
        self.power_step_bonus = float(power_step_bonus)
        self.ghost_threshold = float(ghost_threshold)
        self.ghost_mult = float(ghost_mult)

        self.max_steps = max_steps

        # state
        self._raw_return = 0.0
        self._no_score = 0
        self._prev_lives = None
        self._steps = 0
        self._power_left = 0
        self._last_score_step = -10**9
        self._combo_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._raw_return = 0.0
        self._no_score = 0
        self._steps = 0
        self._power_left = 0
        self._combo_count = 0
        self._last_score_step = -10**9
        self._prev_lives = info.get("lives", None)
        return obs, info

    def _combo_enabled(self):
        return (self.combo_window > 0) and (self.combo_step_bonus > 0.0)

    def _power_enabled(self):
        return (self.power_steps > 0) and np.isfinite(self.power_trigger_min)

    def step(self, action):
        obs, raw_r, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        raw_r = float(raw_r)

        # Base + survival
        r = self.base_scale * raw_r + self.survive_bonus
        self._raw_return += raw_r

        # Positive score shaping + combos
        if raw_r > 0.0:
            r += self.pos_boost
            if self._combo_enabled():
                if (self._steps - self._last_score_step) <= self.combo_window:
                    self._combo_count += 1
                else:
                    self._combo_count = 1
                r += self._combo_count * self.combo_step_bonus
            self._last_score_step = self._steps
            self._no_score = 0
        else:
            self._no_score += 1

        # Power mode heuristic
        if self._power_enabled():
            if raw_r >= self.power_trigger_min:
                self._power_left = max(self._power_left, self.power_steps)
            if self._power_left > 0:
                r += self.power_step_bonus
                if raw_r >= self.ghost_threshold:
                    r += self.ghost_mult
                self._power_left -= 1
        else:
            self._power_left = 0  # make sure it's off

        # Anti-stall (only if configured)
        if (self.no_score_penalty > 0.0) and (self._no_score > self.no_score_patience):
            r -= self.no_score_penalty

        # Death via lives delta
        lives = info.get("lives", self._prev_lives)
        if (self._prev_lives is not None) and (lives is not None) and (lives < self._prev_lives):
            r -= self.death_penalty
            self._no_score = 0
            self._combo_count = 0
            self._power_left = 0
        self._prev_lives = lives

        # Optional cap
        if (self.max_steps is not None) and (self._steps >= self.max_steps):
            truncated = True

        if terminated or truncated:
            info = dict(info)
            info["raw_return"] = self._raw_return

        return obs, r, terminated, truncated, info