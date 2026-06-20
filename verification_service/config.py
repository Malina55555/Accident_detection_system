# verification_service/config.py

# ==========================================================
# SERVER SETTINGS
# ==========================================================
VERIFICATION_HOST = "0.0.0.0"
VERIFICATION_PORT = 5002

# ==========================================================
# MODEL SETTINGS
# ==========================================================
# RTDETR для повреждений
RTDETR_MODEL_PATH = None #заменить на путь к весам модели, н-р: r"C:\Users\User\Diplom\verification_service\models\rtdetr_damage.pt" 
RTDETR_CONF_THRESHOLD = 0.7
USE_REAL_RTDETR = False  #True  # False - заглушка, True - реальная модель

# VLM1 (Gemma)
VLM1_MODEL_NAME = "google/gemma-4-e2b-it"
USE_REAL_VLM1 = False  # False - заглушка, True - реальная модель  Прим-е: заглушка использует реальный вывод модели, но из другого устройства

# ==========================================================
# REPORT SETTINGS
# ==========================================================
REPORT_SAVE_PATH = None # заменить на директорию с отчтами, н-р: r"C:\Users\User\Diplom\reports" 
FONTS_PATH = None #хаменить на путь к русскому шрифту, н-р: r"C:\Users\User\Diplom\verification_service\fonts\times.ttf"  # Путь к шрифту для PDF

# ==========================================================
# SERVICE URLs
# ==========================================================
FRONTEND_SERVICE_URL = "http://localhost:5000"