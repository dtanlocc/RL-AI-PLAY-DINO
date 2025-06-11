# adaptive_dino_rl/utils/vision_debug.py
import cv2
import numpy as np
import pyautogui
import time
from config.game_config import COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_YELLOW


def get_current_mode_for_debug(config):
    mon_x, mon_y, mon_w, mon_h = map(int, config["MONITOR_REGION"])
    try:
        mode_check_region = (
            mon_x + mon_w + int(config["MODE_CHECK_REGION_OFFSET_X"]),
            mon_y + int(config["MODE_CHECK_REGION_OFFSET_Y"]),
            int(config["MODE_CHECK_REGION_WIDTH"]),
            int(config["MODE_CHECK_REGION_HEIGHT"])
        )
        brightness_threshold = config["DARK_MODE_BRIGHTNESS_THRESHOLD"]
        scr = pyautogui.screenshot(region=mode_check_region).convert('L')
        return np.mean(np.array(scr)) < brightness_threshold
    except (pyautogui.PyAutoGUIException, OSError):
        return False


def detect_obstacle_for_vision_debug(config, current_is_dark_mode):
    try:
        obs_raw_np = np.array(pyautogui.screenshot(region=tuple(map(int, config["MONITOR_REGION"]))).convert('L'))
        obs_normalized_np = obs_raw_np.copy()
        if current_is_dark_mode:
            obs_normalized_np = 255 - obs_raw_np

        obs_resized = cv2.resize(obs_normalized_np, (84, 84), interpolation=cv2.INTER_AREA)
        detection_zone = obs_resized[:, 40:84]
        threshold = config["OBSTACLE_PIXEL_THRESHOLD_NORMALIZED"]
        return np.sum(detection_zone < 128) > threshold
    except (pyautogui.PyAutoGUIException, OSError):
        return False


def debug_vision_thread(config, stop_event):
    print("--- Luong Debug Vision da bat dau (nhan 'q' tren cua so de thoat HOAC Ctrl+C trong console) ---")
    monitor_region = tuple(map(int, config["MONITOR_REGION"]))
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while not stop_event.is_set():
            current_is_dark_mode = get_current_mode_for_debug(config)
            obstacle_present = detect_obstacle_for_vision_debug(config, current_is_dark_mode)
            monitor_box_color = COLOR_RED if obstacle_present else COLOR_GREEN

            try:
                screen_pil = pyautogui.screenshot()
            except (pyautogui.PyAutoGUIException, OSError):
                time.sleep(0.05)
                continue

            frame_to_draw = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2BGR)

            # Vẽ các vùng
            m_x, m_y, m_w, m_h = monitor_region
            mode_text = "Dark" if current_is_dark_mode else "Light"
            obs_text = "OBS!" if obstacle_present else ""
            cv2.rectangle(frame_to_draw, (m_x, m_y), (m_x + m_w, m_y + m_h), monitor_box_color, 2)
            cv2.putText(frame_to_draw, f'MONITOR ({mode_text}) {obs_text}', (m_x, m_y - 10), font, 0.4,
                        monitor_box_color, 1)

            # ... (Thêm các hình vẽ khác nếu cần: GAME_OVER, MODE_CHECK)

            # Hiển thị
            center_x, center_y = m_x + m_w // 2, m_y + m_h // 2
            display_w, display_h = 800, 450
            start_x, start_y = max(0, center_x - display_w // 2), max(0, center_y - display_h // 2)
            cv2.imshow("Vung Quan Sat (Debug)", frame_to_draw[start_y:start_y + display_h, start_x:start_x + display_w])

            # AI's Vision
            obs_raw_ai = np.array(pyautogui.screenshot(region=monitor_region).convert('L'))
            obs_norm_ai = obs_raw_ai.copy()
            if current_is_dark_mode:
                obs_norm_ai = 255 - obs_norm_ai
            obs_disp_resized = cv2.resize(obs_norm_ai, (300, 300), interpolation=cv2.INTER_NEAREST)

            debug_obs_with_zone = cv2.cvtColor(obs_disp_resized, cv2.COLOR_GRAY2BGR)
            scale_factor = 300 / 84.0
            zone_x_start = int(40 * scale_factor)
            zone_x_end = int(84 * scale_factor)
            cv2.rectangle(debug_obs_with_zone, (zone_x_start, 0), (zone_x_end - 1, 299), COLOR_YELLOW, 1)
            cv2.imshow("AI's Vision (Normalized + Zone)", debug_obs_with_zone)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                stop_event.set()
                break
    except Exception as e:
        print(f"Loi trong luong debug_vision_thread: {e}")
    finally:
        cv2.destroyAllWindows()
        print("--- Luong Debug Vision da ket thuc ---")


def debug_go_normalized(config):
    print("--- CHẾ ĐỘ CHẨN ĐOÁN PHÁT HIỆN GAME OVER (TRÊN ẢNH ĐÃ CHUẨN HÓA) ---")
    current_mode_is_dark = get_current_mode_for_debug(config)
    print(f"Chế độ game hiện tại: {'Dark' if current_mode_is_dark else 'Light'}")

    def get_pixel_count(region, is_dark):
        screen_raw = np.array(pyautogui.screenshot(region=region).convert('L'))
        screen_norm = screen_raw.copy()
        if is_dark:
            screen_norm = 255 - screen_raw
        return np.sum(screen_norm < 128)

    go_region = tuple(map(int, config["GAME_OVER_REGION"]))
    input(f"Mở game, để CHẠY, sau đó nhấn Enter...")
    p_run = get_pixel_count(go_region, current_mode_is_dark)
    print(f"Pixel count (đã chuẩn hóa) khi đang chạy: {p_run}")
    input(f"Bây giờ, để GAME OVER, sau đó nhấn Enter...")
    p_over = get_pixel_count(go_region, current_mode_is_dark)
    print(f"Pixel count (đã chuẩn hóa) khi game over: {p_over}")

    current_thresh = config['GAMEOVER_PIXEL_THRESHOLD_NORMALIZED']
    print(f"\nNgưỡng hiện tại: {current_thresh}")
    if p_over > p_run + 5:
        sugg = int((p_run + p_over) / 2)
        print(f"Gợi ý: Đặt GAMEOVER_PIXEL_THRESHOLD_NORMALIZED thành khoảng {sugg}")
    else:
        print("Lỗi: Pixel count khi game over không đủ lớn hơn khi chạy.")