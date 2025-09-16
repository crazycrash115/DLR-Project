from stable_baselines3.common.callbacks import BaseCallback

class AutoSaveCallback(BaseCallback):
    def __init__(self, save_path, save_freq=1000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"Autosaved model to {self.save_path} at step {self.n_calls}")
        return True