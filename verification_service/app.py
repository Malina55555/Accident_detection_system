# verification_service/app.py
import os
import sys
import cv2
import base64
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image

# Отключаем сетевые запросы ДО импорта всего остального
os.environ['ULTRALYTICS_SYNC'] = '0'
os.environ['ULTRALYTICS_CHECK'] = '0'
os.environ['YOLO_VERBOSE'] = '0'

from config import *
from models.rtdetr_detector import RTDETRDamageDetector
from models.vlm1_gemma import VLM1Gemma
# from models.vlm2_qwen import VLM2Qwen
from services.report_service import ReportService

# ==========================================================
# INITIALIZATION
# ==========================================================

app = Flask(__name__)

print("[INIT] Initializing Verification Service...")

# Инициализация RTDETR для повреждений (реальная модель)
print(f"[INIT] Loading damage detection model from: {RTDETR_MODEL_PATH}")
damage_detector = RTDETRDamageDetector(RTDETR_MODEL_PATH, RTDETR_CONF_THRESHOLD, USE_REAL_RTDETR)

# Инициализация VLM1 и VLM2
vlm1 = VLM1Gemma(VLM1_MODEL_NAME, USE_REAL_VLM1)
# vlm2 = VLM2Qwen(QWEN_MODEL_PATH, USE_REAL_VLM2)
report_service = ReportService()

print("[INIT] Verification Service ready\n")


# ==========================================================
# HELPERS
# ==========================================================

def base64_to_frame(b64_string):
    """Конвертирует base64 строку в кадр OpenCV"""
    if b64_string is None:
        return None

    try:
        img_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"[ERROR] Failed to convert base64 to frame: {e}")
        return None


# ==========================================================
# API ENDPOINTS
# ==========================================================

@app.route('/verify', methods=['POST'])
def verify():
    """Верификация ДТП и анализ повреждений"""
    data = request.json

    frames_base64 = data.get('frames', [])
    timestamps = data.get('timestamps', [])
    bboxes = data.get('bboxes', [])
    confidences = data.get('confidences', [])
    avg_confidence = 0.0
    if len(confidences) != 0 and confidences[0] is not None:
        for c in confidences:
            if c is not None:
                avg_confidence += c
        avg_confidence /= len(confidences)
    if avg_confidence == 0.0:
        avg_confidence = 0.5

    print(f"\n[VERIFY] Received {len(frames_base64)} frames for verification")

    # ======================================================
    # Шаг 1: VLM1 - верификация ДТП (пока заглушка)
    # ======================================================
    vlm1_result = vlm1.verify_accident(frames_base64, timestamps, bboxes)
    verdict = vlm1_result["verdict"]
    gemma_description = vlm1_result["description"]

    print(f"[VERIFY] VLM1 verdict: {verdict}")

    # Если ДТП не подтверждено, возвращаем сразу
    if not verdict:
        return jsonify({
            "verdict": False,
            "message": "Accident not confirmed by VLM1"
        })

    # ======================================================
    # Шаг 2: Берём последний кадр для анализа повреждений
    # ======================================================
    last_frame_b64 = frames_base64[-1] if frames_base64 else None

    if last_frame_b64 is None:
        return jsonify({
            "verdict": True,
            "error": "No frame for damage analysis"
        }), 400

    # Конвертируем в OpenCV формат
    last_frame = base64_to_frame(last_frame_b64)

    if last_frame is None:
        return jsonify({
            "verdict": True,
            "error": "Failed to convert frame"
        }), 400

    # ======================================================
    # Шаг 2.5: Обрезаем кадр по последнему bbox (если есть)
    # ======================================================
    cropped_frame = last_frame.copy()  # по умолчанию используем весь кадр
    cropped_bbox = None

    # Получаем bbox для последнего кадра
    if bboxes and len(bboxes) > 0:
        last_bbox = bboxes[-1] if bboxes[-1] is not None else None

        if last_bbox is not None and len(last_bbox) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in last_bbox]

            # Проверяем, что координаты в пределах изображения
            h, w = last_frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Обрезаем кадр
            cropped_frame = last_frame[y1:y2, x1:x2]
            cropped_bbox = (x1, y1, x2, y2)

            print(f"[VERIFY] Cropped frame to bbox: {cropped_bbox}, size: {cropped_frame.shape}")

        else:
            print(f"[VERIFY] No valid bbox for last frame, using full frame")
    else:
        print(f"[VERIFY] No bboxes provided, using full frame")

    # ======================================================
    # Шаг 3: RTDETR - детекция повреждений
    # ======================================================

    damages = damage_detector.detect(cropped_frame)  #last_frame)
    # if cropped_bbox is not None:
    #     x_offset, y_offset, _, _ = cropped_bbox
    #     h, w = last_frame.shape[:2]
    #     for damage in damages:
    #         if 'bbox' in damage:
    #             # Сдвигаем bbox повреждения на смещение обрезки
    #             damage['bbox'][0] += x_offset/w  # x1
    #             damage['bbox'][1] += y_offset/h  # y1
    #             damage['bbox'][2] += x_offset/w  # x2
    #             damage['bbox'][3] += y_offset/h  # y2
    #     print(f"[VERIFY] Adjusted {len(damages)} damage bboxes to original coordinates")
    # #
    # print(f"[VERIFY] Detected {len(damages)} damages")
    # Если кадр был обрезан, корректируем координаты обратно к оригинальному изображению
    if cropped_bbox is not None:
        h_orig, w_orig = last_frame.shape[:2]
        x_offset, y_offset, x2, y2 = cropped_bbox
        crop_w = x2 - x_offset  # ширина обрезанной области
        crop_h = y2 - y_offset  # высота обрезанной области

        for damage in damages:
            if 'bbox' in damage:
                # Нормализованные координаты повреждения относительно обрезанного кадра
                # Преобразуем обратно к координатам оригинального кадра
                # Сначала денормализуем относительно crop, затем добавляем смещение и снова нормализуем
                x1_rel, y1_rel, x2_rel, y2_rel = damage['bbox']

                # Переводим в абсолютные пиксельные координаты обрезанного кадра
                x1_abs_crop = x1_rel * crop_w
                y1_abs_crop = y1_rel * crop_h
                x2_abs_crop = x2_rel * crop_w
                y2_abs_crop = y2_rel * crop_h

                # Переводим в абсолютные координаты оригинального кадра
                x1_abs_orig = x1_abs_crop + x_offset
                y1_abs_orig = y1_abs_crop + y_offset
                x2_abs_orig = x2_abs_crop + x_offset
                y2_abs_orig = y2_abs_crop + y_offset

                # Нормализуем обратно относительно оригинального кадра
                h_orig, w_orig = last_frame.shape[:2]
                damage['bbox'][0] = x1_abs_orig / w_orig  # x1
                damage['bbox'][1] = y1_abs_orig / h_orig  # y1
                damage['bbox'][2] = x2_abs_orig / w_orig  # x2
                damage['bbox'][3] = y2_abs_orig / h_orig  # y2

        print(
            f"[VERIFY] Adjusted {len(damages)} damage bboxes from cropped ({crop_w}x{crop_h}) to original frame ({w_orig}x{h_orig})")
    else:
        h_orig, w_orig = last_frame.shape[:2]
        print(f"[VERIFY] No cropping, damages already in original coordinates ({w_orig}x{h_orig})")

    # ======================================================
    # Шаг 4: VLM2 - описание повреждений
    # ======================================================
    vlm2_result = vlm1.describe_damages(last_frame_b64, damages)  # vlm2.describe_damages(last_frame_b64, damages)
    # damages2 = vlm2_result["damages"]

    # ======================================================
    # Шаг 5: Формирование PDF отчёта с bbox
    # ======================================================

    # Подготавливаем 5 кадров с bbox ДТП для отчёта
    accident_frames_with_bbox = []
    for i, b64_frame in enumerate(frames_base64):
        frame = base64_to_frame(b64_frame)
        bbox = bboxes[i] if i < len(bboxes) else None
        accident_frames_with_bbox.append((frame, bbox))

    # Кадр с повреждениями и его bbox
    damage_frame_with_bbox = (last_frame, damages)  #damages2

    # Данные детектора
    detector_verdict = {
        "confidence": avg_confidence,  # Берётся средняя уверенность из detector_service
        "source": data.get('source', 'unknown')
    }

    # Генерируем PDF
    pdf_path = report_service.generate_report(
        detector_verdict=detector_verdict,
        gemma_description=gemma_description,
        qwen_output=vlm2_result,
        accident_frames_with_bbox=accident_frames_with_bbox,
        damage_frame_with_bbox=damage_frame_with_bbox
    )

    print(f"[VERIFY] PDF generated: {pdf_path}\n")

    return jsonify({
        "verdict": True,
        "pdf_path": pdf_path,
        "damages_count": len(damages),
        "severity_score": vlm2_result.get("severity_score", 0)
    })


@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности сервиса"""
    return jsonify({
        "status": "healthy",
        "damage_model": "loaded" if damage_detector.model is not None else "stub",
        "vlm1_mode": "real" if USE_REAL_VLM1 else "stub",
        "vlm2_mode": "real" if USE_REAL_VLM2 else "stub"
    })


if __name__ == "__main__":
    print(f"[START] Running on http://{VERIFICATION_HOST}:{VERIFICATION_PORT}")
    app.run(host=VERIFICATION_HOST, port=VERIFICATION_PORT, debug=False)
