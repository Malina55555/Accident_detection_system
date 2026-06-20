# detector_service/app.py
import threading
from flask import Flask, request, jsonify, Response
import cv2
import time
from state_machine import AccidentStateMachine
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
state_machine = None
processing_thread = None


@app.route('/start', methods=['POST'])
def start():
    global state_machine, processing_thread

    data = request.json
    source = data.get('source', 'webcam')
    video_path = data.get('video_path', None)

    if state_machine and state_machine.running:
        return jsonify({"status": "error", "message": "Already running"}), 400

    try:
        state_machine = AccidentStateMachine()
        processing_thread = threading.Thread(
            target=state_machine.run,
            args=(source,),
            kwargs={"video_path": video_path}
        )
        processing_thread.daemon = True
        processing_thread.start()

        return jsonify({"status": "success", "message": f"Started {source}"})
    except Exception as e:
        logger.error(f"Start failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop():
    global state_machine
    if state_machine:
        state_machine.stop()
        # Даём время на завершение потока
        time.sleep(0.5)
        state_machine = None
    return jsonify({"status": "success", "message": "Stopped"})


@app.route('/status', methods=['GET'])
def status():
    """Статус детектора (быстрый ответ без блокировок)"""
    global state_machine

    # Быстрая проверка без блокировок
    if state_machine is None:
        return jsonify({
            "service_available": True,
            "running": False,
            "state": None,
            "source": None
        })

    # Получаем статус с защитой от блокировок
    try:
        running = state_machine.running
        state = state_machine.state if hasattr(state_machine, 'state') else None
        source = state_machine.source_type if hasattr(state_machine, 'source_type') else None
    except:
        running = False
        state = None
        source = None

    return jsonify({
        "service_available": True,
        "running": running,
        "state": state,
        "source": source
    })


@app.route('/video_feed')
def video_feed():
    """Видеопоток с отображением bbox"""

    def generate():
        global state_machine
        last_frame_time = 0
        frame_interval = 0.066  # ~15 FPS
        empty_frame_count = 0

        while True:
            try:
                # Проверяем существование state_machine
                if state_machine is None:
                    time.sleep(0.1)
                    continue

                # Проверяем, запущен ли детектор
                if not state_machine.running:
                    time.sleep(0.1)
                    continue

                # Получаем кадр
                frame = None
                try:
                    frame = state_machine.get_current_frame()
                except Exception as e:
                    print(f"[ERROR] Failed to get frame: {e}")
                    time.sleep(0.1)
                    continue

                if frame is not None:
                    empty_frame_count = 0
                    current_time = time.time()

                    if current_time - last_frame_time >= frame_interval:
                        # Рисуем bbox если есть
                        try:
                            bbox = state_machine.get_current_bbox() if hasattr(state_machine,
                                                                               'get_current_bbox') else None
                            if bbox is not None:
                                frame_with_bbox = frame.copy()
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            else:
                                frame_with_bbox = frame
                        except Exception as e:
                            frame_with_bbox = frame

                        # Кодируем в JPEG
                        _, buffer = cv2.imencode('.jpg', frame_with_bbox, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        last_frame_time = current_time
                else:
                    empty_frame_count += 1
                    if empty_frame_count > 30:
                        # Слишком долго нет кадров
                        time.sleep(0.5)
                    else:
                        time.sleep(0.05)

            except Exception as e:
                print(f"[ERROR] Video feed error: {e}")
                time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host=DETECTOR_HOST, port=DETECTOR_PORT, debug=False, threaded=True)