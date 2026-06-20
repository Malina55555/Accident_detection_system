# detector_service/state_machine.py
import os
import ssl
import cv2
import time
import base64
import threading
import requests
from collections import deque
from dataclasses import dataclass
from ultralytics import YOLO

from config import *

os.environ['ULTRALYTICS_SYNC'] = '0'
os.environ['ULTRALYTICS_CHECK'] = '0'
os.environ['YOLO_VERBOSE'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class Detection:
    detected: bool
    bbox: tuple | None
    center: tuple | None
    inference_time: float = 0.0
    confidence: float = 0.0


@dataclass
class FrameRecord:
    ts: float
    frame: any
    bbox: tuple | None
    confidence: float = 0.0


def center_distance(c1, c2):
    if c1 is None or c2 is None:
        return float('inf')
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


class YOLODetector:
    def __init__(self, model_path, conf_threshold):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

        if os.path.exists(model_path):
            print(f"[YOLO] Loading model from: {model_path}")
            self.model = YOLO(model_path)
            print(f"[YOLO] Model loaded")
        else:
            print(f"[YOLO] Model not found, using STUB")
            self.model = None

    def detect(self, frame):
        start_time = time.time()

        if self.model is None:
            h, w = frame.shape[:2]
            return Detection(
                detected=True,
                bbox=(w * 0.3, h * 0.3, w * 0.7, h * 0.7),
                center=(w * 0.5, h * 0.5),
                inference_time=0.01,
                confidence=0.5
            )

        results = self.model(frame, verbose=False)
        result = results[0]

        if len(result.boxes) == 0:
            return Detection(detected=False, bbox=None, center=None, inference_time=time.time() - start_time, confidence=0.5)

        best_conf = -1
        best_box = None
        best_center = None

        for box in result.boxes:
            conf = float(box.conf.cpu().item())
            if conf < self.conf_threshold:
                continue

            if conf > best_conf:
                best_conf = conf
                xc, yc, w, h = box.xywh.cpu().numpy()[0]
                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2
                best_box = (float(x1), float(y1), float(x2), float(y2))
                best_center = (float(xc), float(yc))

        return Detection(
            detected=best_box is not None,
            bbox=best_box,
            center=best_center,
            inference_time=time.time() - start_time,
            confidence=best_conf

        )


class AccidentStateMachine:
    def __init__(self):
        self.detector = None
        self.buffer = deque(maxlen=int(FPS_TARGET * BUFFER_SEC))

        self.state = "WAIT_ACCIDENT"
        self.true_counter = 0
        self.t0 = None
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        self.frame4 = None
        self.frame5 = None
        self.last_seen_ts = None
        self.static_start_ts = None
        self.reference_bbox = None
        self.reference_center = None
        self.reference_ts = None
        self.cooldown_start = None

        self.cap = None
        self.running = False
        self.source_type = None

        # Для видеопотока на фронтенд (только последний кадр, без кодирования)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.current_bbox = None  # Добавить: текущий bbox для отображения
        self.bbox_lock = threading.Lock()

    def init_detector(self):
        print(f"[INIT] Loading YOLO model from: {YOLO_MODEL_PATH}")
        self.detector = YOLODetector(YOLO_MODEL_PATH, CONF_THRESHOLD)

    def get_frame_copy(self, target_ts):
        best_item = None
        best_dt = float('inf')
        for item in self.buffer:
            dt = abs(item.ts - target_ts)
            if dt < best_dt:
                best_dt = dt
                best_item = item
        if best_item is None:
            return None
        return FrameRecord(ts=best_item.ts, frame=best_item.frame.copy(), bbox=best_item.bbox, confidence=best_item.confidence)

    def get_current_frame_copy(self, frame, bbox, ts, confidence):
        return FrameRecord(ts=ts, frame=frame.copy(), bbox=bbox, confidence=confidence)

    def reset_event(self):
        self.true_counter = 0
        self.t0 = None
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        self.frame4 = None
        self.frame5 = None
        self.last_seen_ts = None
        self.static_start_ts = None
        self.reference_bbox = None
        self.reference_center = None
        self.reference_ts = None

    def send_to_verification(self, frames):
        print(f"[STATE] Sending verification request...")
        frames_data = []
        timestamps = []
        bboxes = []
        confidences = []

        for fr in frames:
            if fr is None:
                frames_data.append(None)
                timestamps.append(None)
                bboxes.append(None)
                confidences.append(None)
            else:
                frames_data.append(frame_to_base64(fr.frame))
                timestamps.append(fr.ts)
                bboxes.append(fr.bbox)
                confidences.append(fr.confidence)

        try:
            response = requests.post(VERIFICATION_SERVICE_URL, json={
                "frames": frames_data,
                "timestamps": timestamps,
                "bboxes": bboxes,
                "confidences": confidences,
                "source": self.source_type
            }, timeout=30)

            if response.status_code == 200:
                verdict = response.json().get("verdict", False)
                if verdict:
                    self.state = "APPROVED_COOLDOWN"
                    self.cooldown_start = time.time()
                else:
                    self.state = "WAIT_ACCIDENT"
                    self.reset_event()
            else:
                self.state = "WAIT_ACCIDENT"
                self.reset_event()
        except Exception as e:
            print(f"[ERROR] {e}")
            self.state = "WAIT_ACCIDENT"
            self.reset_event()

    def finish_track_accident(self, final_ts):
        """Завершает TRACK_ACCIDENT и переходит в WAIT_VERDICT"""

        # frame5 - последний кадр (стабильный или последний доступный)
        self.frame5 = self.get_frame_copy(final_ts)

        # Если frame5 нет (ситуация конца видео), используем frame3
        if self.frame5 is None and self.frame3 is not None:
            print("[WARNING] frame5 not available, using frame3 as fallback")
            self.frame5 = self.frame3

        # Если frame5 не найден, используем последний доступный кадр из буфера
        if self.frame5 is None and len(self.buffer) > 0:
            self.frame5 = self.buffer[-1]  # последний кадр в буфере
            print("[WARNING] Using last buffer frame as frame5")

        # frame1 и frame2 - кадры до ДТП
        self.frame1 = self.get_frame_copy(self.t0 - 1.0) if self.t0 else None
        self.frame2 = self.get_frame_copy(self.t0 - 0.5) if self.t0 else None

        # frame4 - промежуточный между frame3 и frame5
        if self.frame3 and self.frame5:
            # Если frame3 и frame5 есть - считаем промежуточный
            if self.frame5.ts > self.frame3.ts:
                mid_ts = self.frame3.ts + (self.frame5.ts - self.frame3.ts) / 10
                self.frame4 = self.get_frame_copy(mid_ts)
            else:
                # На случай, если frame5 раньше frame3 (не должно быть, но на всякий случай)
                self.frame4 = self.frame3
        else:
            self.frame4 = self.frame3 or self.frame5

        # Собираем 5 кадров (заменяем None на доступные)
        five_frames = (
            self.frame1 or self.frame2 or self.frame3,  # frame1
            self.frame2 or self.frame1 or self.frame3,  # frame2
            self.frame3,  # frame3
            self.frame4,  # frame4
            self.frame5  # frame5
        )

        # Логируем, какие кадры реально есть
        available = sum(1 for f in five_frames if f is not None)
        print(f"[STATE] Collected {available}/5 frames for verification")

        self.state = "WAIT_VERDICT"

        # Отправляем на верификацию
        verify_thread = threading.Thread(target=self.send_to_verification, args=(five_frames,))
        verify_thread.daemon = True
        verify_thread.start()

    def process_frame(self, frame):
        ts = time.time()

        detection = self.detector.detect(frame)
        #print(f"[DETECTOR]: {detection.detected}")  #debug

        # Сохраняем кадр для видеопотока (без кодирования, просто ссылка)
        with self.frame_lock:
            self.current_frame = frame.copy()

        with self.bbox_lock:
            if detection.detected and detection.bbox:
                self.current_bbox = detection.bbox
            else:
                self.current_bbox = None

        self.buffer.append(FrameRecord(ts=ts, frame=frame.copy(), bbox=detection.bbox, confidence=detection.confidence))

        if self.state == "WAIT_ACCIDENT":
            if detection.detected:
                self.true_counter += 1
            else:
                self.true_counter = 0

            if self.true_counter >= CONFIRM_FRAMES:
                self.t0 = ts
                self.frame1 = self.get_frame_copy(self.t0 - 1.0)
                self.frame2 = self.get_frame_copy(self.t0 - 0.5)
                self.frame3 = self.get_current_frame_copy(frame, detection.bbox, ts, detection.confidence)
                self.last_seen_ts = ts
                self.reference_bbox = detection.bbox
                self.reference_center = detection.center
                self.reference_ts = ts
                self.static_start_ts = None
                self.state = "TRACK_ACCIDENT"

        elif self.state == "TRACK_ACCIDENT":
            if detection.detected:
                self.last_seen_ts = ts
                current_iou = iou(self.reference_bbox, detection.bbox)
                current_shift = center_distance(self.reference_center, detection.center)
                is_static = (current_iou > IOU_STATIC_THRESHOLD and current_shift < CENTER_SHIFT_THRESHOLD)

                if is_static:
                    if self.static_start_ts is None:
                        self.static_start_ts = ts
                    elif ts - self.static_start_ts >= STATIC_DURATION_SEC:
                        self.finish_track_accident(ts)
                else:
                    self.static_start_ts = None

                if ts - self.reference_ts >= REFERENCE_UPDATE_SEC:
                    self.reference_bbox = detection.bbox
                    self.reference_center = detection.center
                    self.reference_ts = ts
            else:
                if self.last_seen_ts and ts - self.last_seen_ts >= LOST_TIMEOUT_SEC:
                    self.finish_track_accident(self.last_seen_ts)

        elif self.state == "APPROVED_COOLDOWN":
            if ts - self.cooldown_start >= WAIT_AFTER_ACCIDENT_SEC:
                self.reset_event()
                self.state = "WAIT_ACCIDENT"

        return detection.inference_time

    def get_current_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def get_current_bbox(self):
        """Возвращает текущий bbox"""
        if not hasattr(self, 'bbox_lock'):
            return None
        with self.bbox_lock:
            if self.current_bbox is not None:
                return self.current_bbox.copy() if hasattr(self.current_bbox, 'copy') else self.current_bbox
            return None

    def run_webcam(self):
        """Запуск обработки с веб-камеры (синхронизация по времени)"""
        self.source_type = "webcam"
        self.init_detector()
        self.reset_event()
        self.cap = cv2.VideoCapture(CAMERA_ID)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Получаем FPS камеры
        camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:
            camera_fps = 30  # стандартное значение

        frame_time = 1.0 / camera_fps  # время между кадрами
        print(f"[WEBCAM] Camera FPS: {camera_fps}, frame time: {frame_time:.3f}s")

        self.running = True
        last_frame_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = time.time()

            # Ждём, пока не наступит время следующего кадра
            expected_time = last_frame_time + frame_time
            if current_time < expected_time:
                time.sleep(expected_time - current_time)

            # Обрабатываем кадр
            inference_time = self.process_frame(frame)

            # Обновляем время последнего кадра
            last_frame_time = time.time()

            # Если обработка заняла больше времени, чем интервал кадров,
            # пропускаем следующий кадр, чтобы не накапливать отставание
            if inference_time > frame_time:
                #print(f"[WEBCAM] Warning: inference time ({inference_time:.3f}s) > frame time ({frame_time:.3f}s)")
                # Пропускаем следующий кадр
                self.cap.grab()
                last_frame_time = time.time()

        self.cap.release()
        print("[WEBCAM] Stopped")

    def run_video(self, video_path):
        """Запуск обработки видеофайла (синхронизация с FPS видео)"""
        self.source_type = "video"
        self.init_detector()
        self.reset_event()
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Получаем FPS видео
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30

        frame_time = 1.0 / video_fps
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VIDEO] FPS: {video_fps}, frame time: {frame_time:.3f}s, total frames: {total_frames}")

        self.running = True
        start_time = time.time()
        frame_idx = 0
        #end_of_video = False  # Флаг конца видео

        while self.running and frame_idx < total_frames:
            # Читаем кадр
            ret, frame = self.cap.read()
            if not ret:
                print(f"[VIDEO] End of video reached. Current state: {self.state}")
                # Конец видео
                if self.state == "TRACK_ACCIDENT":
                    # Принудительное завершение ДТП
                    print("[VIDEO] End of video - forcing accident completion")
                    self.finish_track_accident(self.last_seen_ts or time.time())
                    time.sleep(0.5)
                elif self.state == "WAIT_VERDICT":
                    print("[VIDEO] End of video while waiting for verdict - continuing...")
                    wait_start = time.time()
                    while self.state == "WAIT_VERDICT" and time.time() - wait_start < 10:
                        time.sleep(0.5)
                    print("[VIDEO] Verdict wait finished")
                    # Ожидаем вердикт в фоне
                    #pass
                self.stop()
                break

            # Обрабатываем кадр
            inference_time = self.process_frame(frame)

            frame_idx += 1

            # Рассчитываем, сколько времени должно пройти до следующего кадра
            expected_time = start_time + (frame_idx) * frame_time
            current_time = time.time()

            if current_time < expected_time:
                # Ждём до следующего кадра
                time.sleep(expected_time - current_time)
            elif current_time > expected_time + frame_time:
                # Мы отстаём - пропускаем кадры, чтобы догнать
                frames_behind = int((current_time - expected_time) / frame_time)
                skip_frames = min(frames_behind, 1)  # Пропускаем не более 10 кадров

                if skip_frames > 0:
                    #print(f"[VIDEO] Skipping {skip_frames} frames to catch up")
                    for _ in range(skip_frames):
                        ret = self.cap.grab()
                        frame_idx += 1
                        if not ret:
                            break

            if self.should_stop():
                print("[VIDEO] Stop condition met")
                break

        self.cap.release()
        print("[VIDEO] Stopped")

    def run(self, source, video_path=None):
        if source == "webcam":
            self.run_webcam()
        elif source == "video" and video_path:
            self.run_video(video_path)

    def stop(self):
        self.running = False

    def should_stop(self):
        """Проверяет, нужно ли остановить обработку"""
        # Если видео закончилось и мы не в активном состоянии ожидания

        # Если мы в WAIT_VERDICT, подождём немного
        if self.state == "WAIT_VERDICT" or self.state == "TRACK_ACCIDENT":
            return False

        # # Если в APPROVED_COOLDOWN, подождём
        # if self.state == "APPROVED_COOLDOWN":
        #     return False
        #
        if not self.cap or not self.cap.isOpened():
            return True

        return not self.running
