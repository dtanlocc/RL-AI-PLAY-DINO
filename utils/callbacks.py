# adaptive_dino_rl/utils/callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np
import os


class LoggingCallback(BaseCallback):
    def __init__(self, log_path: str, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.rewards_window = deque(maxlen=100)
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_training_start(self):
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        try:
            self.log_file = open(self.log_path, "w+")
            header = "Episode,Timesteps,Episode Reward,Avg Reward (Last 100),Episode Length\n"
            self.log_file.write(header)
            self.log_file.flush()
        except IOError as e:
            print(f"Loi khi mo file log: {e}")
            self.log_file = None  # Đặt là None nếu không mở được

        if self.verbose > 0:
            print("--- BẮT ĐẦU HUẤN LUYỆN ---")
            print(f"Log sẽ được lưu tại: {self.log_path}")
            print("-" * 70)
            print(f"{'Episode':<7} | {'Timesteps':<9} | {'Ep Reward':<10} | {'Avg Reward':<12} | {'Ep Length':<9}")
            print("-" * 70)

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_count += 1
            self.rewards_window.append(self.current_episode_reward)
            avg_reward = np.mean(self.rewards_window)

            if self.verbose > 0:
                log_str = (
                    f"{self.episode_count:<7} | {self.num_timesteps:<9} | {self.current_episode_reward:<10.2f} | "
                    f"{avg_reward:<12.2f} | {self.current_episode_length:<9}"
                )
                print(log_str)

            if self.log_file:
                file_line = (
                    f"{self.episode_count},{self.num_timesteps},{self.current_episode_reward:.2f},"
                    f"{avg_reward:.2f},{self.current_episode_length}\n"
                )
                self.log_file.write(file_line)
                self.log_file.flush()

            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        return True

    def _on_training_end(self):
        if self.verbose > 0:
            print("-" * 70)
            print("--- KẾT THÚC HUẤN LUYỆN ---")
        if self.log_file:
            self.log_file.close()