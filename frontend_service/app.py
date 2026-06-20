# frontend_service/app.py
import os
import glob
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import *

# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Папка для сохранения загруженных видео
UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_detector_status():
    """Получает статус детектора из detector_service"""
    try:
        response = requests.get(
            f"{DETECTOR_SERVICE_URL}/status",
            timeout=2
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"service_available": False, "running": False}
    except:
        return {"service_available": False, "running": False}


def get_reports_list():
    """Получает список всех PDF отчётов"""
    reports_dir = Path(REPORTS_DIR)

    if not reports_dir.exists():
        return []

    pdf_files = glob.glob(str(reports_dir / "*.pdf"))

    reports = []
    for pdf_path in pdf_files:
        path = Path(pdf_path)
        stat = path.stat()

        reports.append({
            "filename": path.name,
            "name": path.name.replace("report_", "").replace(".pdf", ""),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size": stat.st_size
        })

    reports.sort(key=lambda x: x["date"], reverse=True)
    return reports


# ==========================================================
# ROUTES
# ==========================================================

@app.route('/')
def index():
    """Главная страница"""
    status = get_detector_status()
    reports = get_reports_list()

    return render_template(
        'index.html',
        status=status,
        reports=reports
    )


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Запуск детекции с веб-камеры"""
    try:
        response = requests.post(
            f"{DETECTOR_SERVICE_URL}/start",
            json={"source": "webcam"},
            timeout=5
        )

        if response.status_code == 200:
            return jsonify({"success": True, "message": "Webcam started"})
        else:
            return jsonify({"success": False, "message": response.text}), 500

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/start_video', methods=['POST'])
def start_video():
    """Запуск детекции с видеофайла"""
    try:
        # Проверяем, есть ли файл в запросе
        if 'video_file' not in request.files:
            return jsonify({"success": False, "message": "No video file provided"}), 400

        file = request.files['video_file']

        if file.filename == '':
            return jsonify({"success": False, "message": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

        # Сохраняем файл
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        video_path = UPLOAD_FOLDER / saved_filename
        file.save(str(video_path))

        print(f"[FRONTEND] Video saved: {video_path}")

        # Отправляем запрос в detector_service
        response = requests.post(
            f"{DETECTOR_SERVICE_URL}/start",
            json={"source": "video", "video_path": str(video_path)},
            timeout=5
        )

        if response.status_code == 200:
            return jsonify({"success": True, "message": "Video processing started", "video_path": str(video_path)})
        else:
            return jsonify({"success": False, "message": response.text}), 500

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop():
    """Остановка детекции"""
    try:
        response = requests.post(
            f"{DETECTOR_SERVICE_URL}/stop",
            timeout=5
        )

        if response.status_code == 200:
            return jsonify({"success": True, "message": "Detector stopped"})
        else:
            return jsonify({"success": False, "message": response.text}), 500

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/reports/<filename>')
def download_report(filename):
    """Скачивание PDF отчёта"""
    reports_dir = Path(REPORTS_DIR)

    safe_path = reports_dir / filename
    if not safe_path.exists() or not safe_path.is_file():
        return jsonify({"error": "Report not found"}), 404

    return send_from_directory(
        reports_dir,
        filename,
        as_attachment=True
    )


@app.route('/reports/list', methods=['GET'])
def reports_list():
    """API для получения списка отчётов"""
    reports = get_reports_list()
    return jsonify(reports)


# frontend_service/app.py - исправленный video_feed
# frontend_service/app.py - исправленный video_feed

@app.route('/video_feed')
def video_feed():
    """Прокси видеопотока из detector_service"""
    import requests
    from flask import Response
    import time

    def generate():
        while True:
            try:
                # Проверяем статус детектора с увеличенным таймаутом
                try:
                    status_response = requests.get(
                        f"{DETECTOR_SERVICE_URL}/status",
                        timeout=5  # Увеличили с 2 до 5 секунд
                    )
                    if status_response.status_code == 200:
                        status = status_response.json()

                        if status.get('running', False):
                            # Детектор запущен - проксируем поток
                            url = f"{DETECTOR_SERVICE_URL}/video_feed"
                            try:
                                with requests.get(url, stream=True, timeout=(3, 30)) as r:
                                    if r.status_code == 200:
                                        for chunk in r.iter_content(chunk_size=8192):
                                            if chunk:
                                                yield chunk
                                            else:
                                                time.sleep(0.01)
                                    else:
                                        time.sleep(0.5)
                            except requests.exceptions.Timeout:
                                # Таймаут соединения - просто ждём
                                time.sleep(0.5)
                            except requests.exceptions.ChunkedEncodingError:
                                # Нормальная ситуация при остановке потока
                                time.sleep(0.5)
                    else:
                        time.sleep(0.5)
                except requests.exceptions.Timeout:
                    # Таймаут статуса - просто ждём
                    time.sleep(0.5)
                except requests.exceptions.ConnectionError:
                    # Сервис недоступен
                    time.sleep(1)

            except Exception as e:
                # Любая другая ошибка
                print(f"[ERROR] Video feed proxy error: {e}")
                time.sleep(1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status', methods=['GET'])
def status():
    """API статуса для AJAX обновлений"""
    return jsonify(get_detector_status())


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    print(f"[FRONTEND] Starting on http://{FRONTEND_HOST}:{FRONTEND_PORT}")
    print(f"[FRONTEND] Detector service URL: {DETECTOR_SERVICE_URL}")
    print(f"[FRONTEND] Reports directory: {REPORTS_DIR}")

    app.run(
        host=FRONTEND_HOST,
        port=FRONTEND_PORT,
        debug=False
    )