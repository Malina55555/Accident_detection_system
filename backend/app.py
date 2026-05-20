from pathlib import Path

from flask import Flask
from flask import render_template
from flask import request
from flask import send_file
from flask import redirect
from flask import url_for
from flask import flash

from werkzeug.utils import secure_filename

from app.pipeline.stream_processor import (
    StreamProcessor
)

from app.config import settings


app = Flask(__name__)

app.secret_key = "secret-key"

processor = StreamProcessor()

UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"

Path(UPLOAD_FOLDER).mkdir(
    exist_ok=True
)

Path(REPORTS_FOLDER).mkdir(
    exist_ok=True
)


@app.route("/")
def index():

    reports = sorted(
        Path(REPORTS_FOLDER).glob("*.pdf"),
        reverse=True
    )

    report_names = [
        r.name
        for r in reports
    ]

    return render_template(
        "index.html",
        reports=report_names,
        current_source=
            settings.input_source_type
    )


@app.route(
    "/start_stream",
    methods=["POST"]
)
def start_stream():

    source_type = request.form.get(
        "source_type"
    )

    #
    # update runtime config
    #

    settings.input_source_type = (
        source_type
    )

    #
    # processing
    #

    processor.process()

    flash(
        f"Stream processing started "
        f"({source_type})"
    )

    return redirect(
        url_for("index")
    )


@app.route(
    "/upload",
    methods=["POST"]
)
def upload_video():

    if "video" not in request.files:

        flash("No file uploaded")

        return redirect(
            url_for("index")
        )

    file = request.files["video"]

    if file.filename == "":

        flash("Empty filename")

        return redirect(
            url_for("index")
        )

    filename = secure_filename(
        file.filename
    )

    file_path = (
        Path(UPLOAD_FOLDER)
        / filename
    )

    file.save(file_path)

    #
    # switch source
    #

    settings.input_source_type = (
        "video"
    )

    settings.video_path = str(
        file_path
    )

    #
    # processing
    #

    processor.process()

    flash(
        f"Video processed: "
        f"{filename}"
    )

    return redirect(
        url_for("index")
    )


@app.route("/reports/<filename>")
def download_report(filename):

    path = (
        Path(REPORTS_FOLDER)
        / filename
    )

    return send_file(
        path,
        as_attachment=True
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

# from pathlib import Path
#
# from flask import Flask
# from flask import render_template
# from flask import request
# from flask import send_file
# from flask import redirect
# from flask import url_for
#
# from werkzeug.utils import secure_filename
#
# from app.pipeline.stream_processor import StreamProcessor
#
#
# app = Flask(__name__)
#
# processor = StreamProcessor()
#
# UPLOAD_FOLDER = "uploads"
# REPORTS_FOLDER = "reports"
#
# Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
# Path(REPORTS_FOLDER).mkdir(exist_ok=True)
#
#
# @app.route("/")
# def index():
#
#     return render_template("index.html")
#
#
# @app.route("/upload", methods=["POST"])
# def upload_video():
#
#     if "video" not in request.files:
#         return "No file uploaded"
#
#     file = request.files["video"]
#
#     if file.filename == "":
#         return "Empty filename"
#
#     filename = secure_filename(file.filename)
#
#     file_path = Path(UPLOAD_FOLDER) / filename
#
#     file.save(file_path)
#
#     #
#     # processing
#     #
#
#     processor.process()
#
#     return redirect(url_for("index"))
#
#
# @app.route("/reports/<filename>")
# def download_report(filename):
#
#     path = Path(REPORTS_FOLDER) / filename
#
#     return send_file(path, as_attachment=True)
#
#
# if __name__ == "__main__":
#
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=True
#     )