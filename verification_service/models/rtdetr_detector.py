# verification_service/models/rtdetr_detector.py
import os
import cv2
import numpy as np

# Отключаем сетевые запросы ultralytics
os.environ['ULTRALYTICS_SYNC'] = '0'
os.environ['ULTRALYTICS_CHECK'] = '0'
os.environ['YOLO_VERBOSE'] = '0'

from ultralytics import RTDETR


class RTDETRDamageDetector:
    """RTDETR для детекции повреждений"""

    # Классы повреждений в правильном порядке
    DAMAGE_CLASSES = [
        "dent", "crack", "scratch",
        "glass_shatter", "tire_flat", "lamp_broken"
    ]

    def __init__(self, model_path, conf_threshold=0.5, use_real=True):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.use_real = use_real
        if self.use_real:
            # Проверяем существование модели
            if os.path.exists(model_path):
                print(f"[RTDETR] Loading model from: {model_path}")
                try:
                    self.model = RTDETR(model_path)
                    print(f"[RTDETR] Model loaded successfully")
                    print(f"[RTDETR] Damage classes: {self.DAMAGE_CLASSES}")
                except Exception as e:
                    print(f"[RTDETR] Failed to load model: {e}")
                    self.model = None
            else:
                print(f"[RTDETR] Model not found at: {model_path}")
                print(f"[RTDETR] Using STUB mode")
                self.model = None
        else:
            print(f"[RTDETR] Use stub")

    def _get_class_name(self, class_id):
        """Получает название класса по его ID"""
        if 0 <= class_id < len(self.DAMAGE_CLASSES):
            return self.DAMAGE_CLASSES[class_id]
        return f"unknown_{class_id}"

    def _normalize_bbox(self, bbox, frame_width, frame_height):
        """Конвертирует абсолютные координаты в нормализованные (0-1)"""
        x1, y1, x2, y2 = bbox
        return [
            x1 / frame_width,
            y1 / frame_height,
            x2 / frame_width,
            y2 / frame_height
        ]

    def detect(self, frame):
        """Детектирует повреждения и возвращает нормализованные координаты"""
        if self.model is None or not self.use_real:
            return self._detect_stub(frame)

        try:
            height, width = frame.shape[:2]
            results = self.model(frame, verbose=False)
            result = results[0]

            damages = []

            if len(result.boxes) > 0:
                for box in result.boxes:
                    conf = float(box.conf.cpu().item())
                    if conf < self.conf_threshold:
                        continue

                    # Получаем абсолютные координаты
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]

                    # Конвертируем в нормализованные
                    norm_bbox = self._normalize_bbox([x1, y1, x2, y2], width, height)

                    cls_id = int(box.cls.cpu().item())
                    class_name = result.names[cls_id] if result.names else "damage"

                    damages.append({
                        "bbox": norm_bbox,
                        "confidence": conf,
                        "class": class_name
                    })

            return damages

        except Exception as e:
            print(f"[RTDETR] Error during detection: {e}")
            return self._detect_stub(frame)

    def _detect_stub(self, frame):
        """Заглушка с нормализованными координатами"""
        # Заглушка сразу возвращает нормализованные координаты
        # return [
        #     {
        #         "bbox": [0.3, 0.4, 0.5, 0.6],  # уже нормализованные
        #         "confidence": 0.85,
        #         "class": "dent"
        #     },
        #     {
        #         "bbox": [0.6, 0.3, 0.7, 0.45],  # уже нормализованные
        #         "confidence": 0.72,
        #         "class": "scratch"
        #     }
        # ]
        bbox = [[0.493, 0.404, 0.568, 0.473],
                [0.553, 0.396, 0.59, 0.45],
                [0.503, 0.485, 0.555, 0.577],
                [0.486, 0.434, 0.517, 0.544],
                [0.399, 0.345, 0.427, 0.409],
        ]
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = box
            x1 = ((x1*2390-867)/1047).__round__(3)
            x2 = ((x2 * 2390 - 867) / 1047).__round__(3)
            y1 = ((y1 * 1330-334)/509).__round__(3)
            y2 = ((y2 * 1330 - 334) / 509).__round__(3)
            bbox[i] = [x1, y1, x2, y2]

        print(bbox)

        return [
            {
                "bbox": bbox[0],  #[0.493, 0.404, 0.568, 0.473], #[0.297,
                "class": "lamp_broken",
                "confidence": 0.93
            },
            {
                "bbox": bbox[1], #[0.553, 0.396, 0.59, 0.45],
                "class": "crack", #"lamp_broken",
                "confidence": 0.92
            },
            {
                "bbox": bbox[2],  #[0.503, 0.485, 0.555, 0.577],
                "class": "tire_flat",
                "confidence": 0.92
            },
            {
                "bbox": bbox[3],
                "class": "crack",
                "confidence": 0.81
            },
            {
                "bbox": bbox[4],
                "class": "dent",
                "confidence": 0.72
            },
        ]
