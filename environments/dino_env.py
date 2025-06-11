# adaptive_dino_rl/environments/dino_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
import time


class DinoJumpAdaptiveEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config, render_mode=None):
        super(DinoJumpAdaptiveEnv, self).__init__()
        self.config = config
        self.monitor_region = tuple(map(int, config["MONITOR_REGION"]))
        self.game_over_region = tuple(map(int, config["GAME_OVER_REGION"]))

        self.gameover_pixel_threshold = config["GAMEOVER_PIXEL_THRESHOLD_NORMALIZED"]
        self.obstacle_threshold = config["OBSTACLE_PIXEL_THRESHOLD_NORMALIZED"]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        mon_x, mon_y, mon_w, mon_h = self.monitor_region
        self.mode_check_region = (
            int(mon_x + mon_w + config["MODE_CHECK_REGION_OFFSET_X"]),
            int(mon_y + config["MODE_CHECK_REGION_OFFSET_Y"]),
            int(config["MODE_CHECK_REGION_WIDTH"]),
            int(config["MODE_CHECK_REGION_HEIGHT"])
        )
        self.is_dark_mode = False
        self.dark_mode_brightness_threshold = config["DARK_MODE_BRIGHTNESS_THRESHOLD"]
        self.current_episode_steps = 0
        self.render_mode = render_mode
        self.window = None
        # print("Moi truong DinoJumpAdaptiveEnv da duoc khoi tao.")

    def _update_game_mode(self):
        try:
            mode_check_screen = pyautogui.screenshot(region=self.mode_check_region).convert('L')
            avg_brightness = np.mean(np.array(mode_check_screen))
            self.is_dark_mode = avg_brightness < self.dark_mode_brightness_threshold
        except (pyautogui.PyAutoGUIException, OSError):
            # print("Warning: Khong the chup man hinh de kiem tra che do. Giu nguyen che do truoc do.")
            pass

    def _get_raw_screen_gray(self, region_tuple):
        try:
            screen = pyautogui.screenshot(region=region_tuple).convert('L')
            return np.array(screen, dtype=np.uint8)
        except (pyautogui.PyAutoGUIException, OSError) as e:
            # print(f"Error: Khong the chup man hinh vung {region_tuple}: {e}")
            h, w = region_tuple[3], region_tuple[2]
            return np.zeros((int(h), int(w)), dtype=np.uint8)

    def _get_observation(self):
        self._update_game_mode()
        obs_np_raw = self._get_raw_screen_gray(self.monitor_region)

        obs_np_normalized = obs_np_raw.copy()
        if self.is_dark_mode:
            obs_np_normalized = 255 - obs_np_raw

        obs_image = cv2.resize(obs_np_normalized, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs_image, axis=-1)

    def _is_done(self):
        go_screen_raw = self._get_raw_screen_gray(self.game_over_region)
        if go_screen_raw.size == 0: return True

        go_screen_normalized = go_screen_raw.copy()
        if self.is_dark_mode:
            go_screen_normalized = 255 - go_screen_raw

        return np.sum(go_screen_normalized < 128) > self.gameover_pixel_threshold

    def _detect_obstacle_on_normalized_obs(self, normalized_observation_84x84):
        detection_zone = normalized_observation_84x84[:, 40:self.observation_space.shape[1]]
        return np.sum(detection_zone < 128) > self.obstacle_threshold

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_episode_steps = 0
        try:
            pyautogui.press('space')
            time.sleep(0.5)
        except pyautogui.FailSafeException:
            pass

        current_obs = self._get_observation()
        info = {}

        retry_count = 0
        while self._is_done():
            if retry_count > 10:
                print("ERROR: Khong the reset moi truong sau 10 lan thu!")
                info["needs_reset_again"] = True
                return current_obs, info
            try:
                pyautogui.press('space')
                time.sleep(0.5)
            except pyautogui.FailSafeException:
                time.sleep(1)
            current_obs = self._get_observation()
            retry_count += 1
        return self._get_observation(), info

    def step(self, action):
        if action == 1:
            try:
                pyautogui.press('up')
            except pyautogui.FailSafeException:
                pass

        time.sleep(1 / 30)

        observation = self._get_observation()
        terminated = self._is_done()

        if terminated:
            reward = -25.0
        else:
            reward = 0.5

        self.current_episode_steps += 1

        if not terminated:
            obstacle_near = self._detect_obstacle_on_normalized_obs(np.squeeze(observation))
            if not obstacle_near and action == 1:
                reward -= 0.5

        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        pass  # Không cần làm gì đặc biệt khi đóng env này