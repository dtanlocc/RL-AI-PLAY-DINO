# adaptive_dino_rl/main.py
import os
import argparse
import threading
import time
import pyautogui
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from config.game_config import CONFIG
from environments.dino_env import DinoJumpAdaptiveEnv
from utils.callbacks import LoggingCallback
from utils.vision_debug import debug_vision_thread, debug_go_normalized

stop_debug_vision_event = threading.Event()


def train_agent(config, show_vision=True, transfer_learn_path=None):
    global stop_debug_vision_event
    stop_debug_vision_event.clear()
    vision_thread = None

    if show_vision:
        vision_thread = threading.Thread(target=debug_vision_thread, args=(config, stop_debug_vision_event),
                                         daemon=True)
        vision_thread.start()

    env_lambda = lambda: DinoJumpAdaptiveEnv(config)
    env = DummyVecEnv([env_lambda])
    env = VecFrameStack(env, n_stack=config["N_STACK"], channels_order='last')

    model_checkpoint_prefix = os.path.splitext(os.path.basename(config["MODEL_FILENAME"]))[0]
    callback_list = [
        CheckpointCallback(save_freq=20480, save_path=config["MODEL_DIR"], name_prefix=model_checkpoint_prefix),
        LoggingCallback(log_path=os.path.join(config["LOG_DIR"], config["TRAINING_LOG_FILE"]), verbose=1)
    ]
    callback = CallbackList(callback_list)

    model = None
    reset_num_timesteps = True
    if transfer_learn_path and os.path.exists(transfer_learn_path):
        print(f"--- TIẾP TỤC HUẤN LUYỆN TỪ MODEL: {transfer_learn_path} ---")
        try:
            model = PPO.load(transfer_learn_path, print_system_info=True)
            new_logger = configure(folder=config["LOG_DIR"], format_strings=["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            model.set_env(env)
            model.n_steps = config["N_STEPS"]
            reset_num_timesteps = False
            print("--- Model đã tải và logger đã được thiết lập lại. ---")
        except Exception as e:
            print(f"Lỗi khi tải model cũ: {e}. Huấn luyện từ đầu.")
            traceback.print_exc()
            model = None

    if model is None:
        if transfer_learn_path:
            print(f"--- HUẤN LUYỆN TỪ ĐẦU DO LỖI. ---")
        else:
            print(f"--- HUẤN LUYỆN MODEL MỚI TỪ ĐẦU. ---")
        model = PPO("CnnPolicy", env, verbose=0, tensorboard_log=config["LOG_DIR"],
                    n_steps=config["N_STEPS"], batch_size=256, n_epochs=15, gamma=0.99)

    print("\nVui long chuyen sang cua so game Dino trong 5 giay...")
    time.sleep(5)
    try:
        model.learn(total_timesteps=config["TOTAL_TIMESTEPS"], callback=callback,
                    reset_num_timesteps=reset_num_timesteps)
    except (KeyboardInterrupt, pyautogui.FailSafeException) as e:
        print(f"\nĐã dừng huấn luyện bởi người dùng: {e}")
    except Exception as e:
        print(f"\nLỗi trong quá trình huấn luyện: {e}")
        traceback.print_exc()
    finally:
        if show_vision and vision_thread and vision_thread.is_alive():
            stop_debug_vision_event.set()
            vision_thread.join(timeout=2)

        if model:
            model_path = os.path.join(config["MODEL_DIR"], config["MODEL_FILENAME"])
            model.save(model_path)
            print(f"\nĐã lưu model vào '{model_path}'.")

    if env: env.close()


def play_agent(config, show_vision=True):
    global stop_debug_vision_event
    stop_debug_vision_event.clear()
    vision_thread = None

    if show_vision:
        vision_thread = threading.Thread(target=debug_vision_thread, args=(config, stop_debug_vision_event),
                                         daemon=True)
        vision_thread.start()

    model_path = os.path.join(config["MODEL_DIR_PLAY"], config["MODEL_PLAY"])
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy model tại '{model_path}'.")
        return

    env = DummyVecEnv([lambda: DinoJumpAdaptiveEnv(config)])
    env = VecFrameStack(env, n_stack=config["N_STACK"], channels_order='last')

    try:
        model = PPO.load(model_path, env=env)
        print(f"Đã tải model từ '{model_path}'. Bắt đầu chơi trong 5 giây...")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}");
        env.close();
        return

    time.sleep(5)
    obs = env.reset()
    try:
        while not stop_debug_vision_event.is_set():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                print("Game Over. Đang khởi động lại...")
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nĐã dừng chơi thử.")
    finally:
        if show_vision and vision_thread and vision_thread.is_alive():
            stop_debug_vision_event.set()
            vision_thread.join(timeout=2)
    if env: env.close()


def main():
    parser = argparse.ArgumentParser(description="AI chơi game Dino (Adaptive).")
    parser.add_argument("--mode", type=str,
                        choices=["train", "play", "debug_game_over_normalized", "debug_vision", "train_continue"],
                        default="train")
    parser.add_argument("--novision", action="store_true", help="Chạy không có debug vision.")
    parser.add_argument("--old_model_path", type=str, default=CONFIG.get("OLD_MODEL_DEFAULT_PATH"),
                        help="Đường dẫn model cũ để học tiếp.")
    args = parser.parse_args()

    paths_to_create = [CONFIG["LOG_DIR"], CONFIG["MODEL_DIR"], CONFIG["MODEL_DIR_PLAY"]]
    if args.old_model_path:
        paths_to_create.append(os.path.dirname(args.old_model_path))

    for path in paths_to_create:
        if path: os.makedirs(path, exist_ok=True)

    if args.mode == "debug_game_over_normalized":
        debug_go_normalized(CONFIG)
    elif args.mode == "debug_vision":
        print("Chạy debug vision. Nhấn 'q' trong cửa sổ AI's Vision hoặc Ctrl+C trong console để thoát.")
        debug_thread = threading.Thread(target=debug_vision_thread, args=(CONFIG, stop_debug_vision_event), daemon=True)
        debug_thread.start()
        try:
            while debug_thread.is_alive():
                debug_thread.join(0.1)
        except KeyboardInterrupt:
            print("\nĐang dừng debug vision...")
            stop_debug_vision_event.set()
            debug_thread.join(2)
        print("Debug vision đã kết thúc.")
    elif args.mode == "train":
        train_agent(CONFIG, show_vision=not args.novision, transfer_learn_path=None)
    elif args.mode == "train_continue":
        if not args.old_model_path or not os.path.exists(args.old_model_path):
            print(f"Lỗi: Đường dẫn model cũ '{args.old_model_path}' không hợp lệ hoặc không tồn tại.")
        else:
            train_agent(CONFIG, show_vision=not args.novision, transfer_learn_path=args.old_model_path)
    elif args.mode == "play":
        play_agent(CONFIG, show_vision=not args.novision)


if __name__ == "__main__":
    main()