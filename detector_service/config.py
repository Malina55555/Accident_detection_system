# detector_service/config.py
import os

# ==========================================================
# VIDEO SOURCE
# ==========================================================
CAMERA_ID = 0                    # ID веб-камеры
FPS_TARGET = 30                  # Целевой FPS для оценки буфера

# ==========================================================
# DETECTOR SETTINGS
# ==========================================================
# Пути к моделям
YOLO_MODEL_PATH = None # вставьте путь к весам модели yolo, н-р: r"C:\Users\User\Diplom\detector_service\models\yolo_accident.pt"
CONF_THRESHOLD = 0.7             # Порог уверенности YOLO

# ==========================================================
# STATE MACHINE SETTINGS
# ==========================================================
BUFFER_SEC = 10                  # Глубина буфера в секундах
CONFIRM_FRAMES = 3               # Сколько кадров подряд для подтверждения ДТП
IOU_STATIC_THRESHOLD = 0.90      # IOU для определения статичности
CENTER_SHIFT_THRESHOLD = 10      # Смещение центра в пикселях
STATIC_DURATION_SEC = 2.0        # Сколько секунд ДТП должно быть статичным
LOST_TIMEOUT_SEC = 3.0           # Таймаут потери объекта
WAIT_AFTER_ACCIDENT_SEC = 20.0   # Cooldown после ДТП
REFERENCE_UPDATE_SEC = 1.0       # Как часто обновлять референсный bbox

# ==========================================================
# SERVICE URLs
# ==========================================================
VERIFICATION_SERVICE_URL = "http://localhost:5002/verify"
FRONTEND_SERVICE_URL = "http://localhost:5000"

# ==========================================================
# SERVER SETTINGS
# ==========================================================
DETECTOR_HOST = "0.0.0.0"
DETECTOR_PORT = 5001