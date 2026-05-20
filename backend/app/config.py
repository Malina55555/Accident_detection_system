from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === Другие настройки ===
    wait_for_next_accident_minimum: int = 20  #0  #00   # условно говоря следующая авария скорее всего не случится за эти секунды
    check_wait_sec: float = 1.2

    # === Вход ===
    input_source_type: str = "video"  #video or webcam or rtsp

    webcam_index: int = 0

    video_path: str = r"C:\Users\User\Downloads\v21.mov"  #"videos/test.mp4"

    rtsp_url: str = ""  #"rtsp://admin:password@192.168.0.10:554/stream"

    # === Детектор ===
    detector_model: str = "yolo" #"rtdetr"  #

    yolo_model_path: str = r"C:\Users\User\PycharmProjects\Diplom\backend\app\models\yolo\best.pt"
    rtdetr_model_path: str = r"C:\Users\User\PycharmProjects\Diplom\backend\app\models\rtdetr\best.pt"
    accident_threshold: float = 0.7

    # === Описатель аварии ===

    gemma_model_name: str = "google/gemma-4-e2b-it"  #"google/gemma-4"

    use_gemma_stub: bool = True  # False
    gemma_stub_sleep_sec: int = 5

    # === Описатель повреждений ===

    use_qwen_stub: bool = True  # False
    qwen_stub_sleep_sec: int = 5

    qwen_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"  # "Qwen/Qwen3-VL"
    qwen_path: str = r"C:\Users\User\PycharmProjects\Diplom\backend\app\models\qwen3"

    # === Отчёты ===

    report_save_path: str = r"C:\Users\User\PycharmProjects\Diplom\reports"


settings = Settings()
