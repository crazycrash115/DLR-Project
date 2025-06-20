import gym 

# === Reward Wrapper ===
class MarioRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_frames=3000):
        super(MarioRewardWrapper, self).__init__(env)
        self.prev_x = 0
        self.prev_score = 0
        self.no_move_counter = 0
        self.frame_count = 0
        self.max_frames = max_frames
        self.max_x = 0
        self.prev_life = 2

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x = 0
        self.prev_score = 0
        self.no_move_counter = 0
        self.frame_count = 0
        self.max_x = 0
        self.checkpoint_hit = False
        self.prev_life = 2
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_count += 1

        new_x = info.get("x_pos", 0)
        new_score = info.get("score", 0)
        flag_get = info.get("flag_get", False)

        shaped_reward = 0
        
        #pros for moving 
        if new_x > self.max_x:
            shaped_reward += new_x - self.max_x
            self.max_x = new_x
            self.no_move_counter = 0

        #Adds kind of like a sudo checkpoint (not truly a checkpoint)
        if new_x > 1800 and not self.checkpoint_hit:  
            shaped_reward += 20
            self.checkpoint_hit = True

        #added bonus for score
        if new_score - self.prev_score > 10:
               shaped_reward += (new_score - self.prev_score) * 0.005

        #level complete + reward for finishing earlier
        if flag_get:
            shaped_reward += 50
            shaped_reward += max(0, 15 - self.frame_count // 100)# Max 15 bonus, less if slower


        #if dead or time run out
        if done and not flag_get:
            shaped_reward -= 2

        #closes training if it isnt moving 
        if new_x <= self.max_x:
            self.no_move_counter += 1
            if self.no_move_counter > 100:
                shaped_reward -= 1
                done = True
            #    print("Episode ended early: stuck too long.")

        else:
            self.no_move_counter = 0

        #If its going too long it ends
        if self.frame_count >= self.max_frames:
            done = True
        #    print("Episode ended early: frame limit reached.")

        #Force end if life count drops
        if 'life' in info and info['life'] < self.prev_life:
        #    print("Life lost! Forcing reset to 1-1.")
            shaped_reward -= 2
            done = True

        self.prev_life = info.get('life', 2)
        self.prev_x = new_x
        self.prev_score = new_score

        return obs, shaped_reward, done, info

# === Action Repeat ===
class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=2):
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
