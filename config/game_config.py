# adaptive_dino_rl/config/game_config.py

# ==============================================================================
# CẤU HÌNH TRUNG TÂM - CẦN BẠN KIỂM TRA VÀ TINH CHỈNH KỸ LƯỠNG
# ==============================================================================
CONFIG = {
    "MONITOR_REGION": (231, 223, 328, 64),
    "GAME_OVER_REGION": (396, 201, 9, 9),

    "GAMEOVER_PIXEL_THRESHOLD_NORMALIZED": 24,
   "OBSTACLE_PIXEL_THRESHOLD_NORMALIZED": 50,

    # --- Cấu hình phát hiện chế độ ---
    "MODE_CHECK_REGION_OFFSET_X": -50,
    "MODE_CHECK_REGION_OFFSET_Y": 5,
    "MODE_CHECK_REGION_WIDTH": 40,
    "MODE_CHECK_REGION_HEIGHT": 20,
    "DARK_MODE_BRIGHTNESS_THRESHOLD": 128,

    # --- Cấu hình cho phiên bản hiện tại (v3) ---
    "LOG_DIR": "logs/adaptive_dino_logs_v3",
    "MODEL_DIR": "models/adaptive_dino_models_v3",
    "MODEL_FILENAME": "dino_adaptive_expert_v3.zip",
    "TRAINING_LOG_FILE": "training_log_adaptive_v3.txt",

    # --- Cấu hình cho việc chơi lại một model cụ thể ---
    "MODEL_DIR_PLAY": "models/adaptive_dino_models_v3",  # Ví dụ: chơi model từ thư mục v2
    "MODEL_PLAY": "dino_adaptive_expert_v3_484042_steps.zip",  # Ví dụ: checkpoint cụ thể

    # --- Đường dẫn mặc định cho model cũ để transfer learning ---
    "OLD_MODEL_DEFAULT_PATH": "models/jumps_models/dino_jump_expert.zip",

    # --- Siêu tham số huấn luyện ---
    "N_STACK": 4,
    "N_STEPS": 4096,
    "TOTAL_TIMESTEPS": 500_000
}

# Hằng số màu sắc có thể dùng chung
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)