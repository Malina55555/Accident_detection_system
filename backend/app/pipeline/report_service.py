import uuid
import cv2

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage, #Image,
    Table,
    TableStyle,
)

from app.config import settings


#
# регистрация русского шрифта
#

pdfmetrics.registerFont(
    TTFont(
        "TimesNewRoman",
        "fonts/times.ttf"
    )
)

damage_dict = {
    "dent": "вмятина",
    "crack": "трещина",
    "scratch": "царапина",
    "lamp_broken": "разбитая лампа",
    "tire_flat": "спущенная шина",
    "glass_shatter": "разбитое стекло",
}



class ReportService:

    def __init__(self):

        self.output_dir = Path(
            settings.report_save_path
        )

        self.output_dir.mkdir(
            exist_ok=True,
            parents=True
        )

        #
        # временные изображения
        #

        self.images_dir = (
            self.output_dir / "images"
        )

        self.images_dir.mkdir(
            exist_ok=True,
            parents=True
        )

    #
    # draw bbox for qwen damages
    #

    def draw_damage_boxes(
            self,
            frame,
            damages
    ):

        annotated = frame.copy()

        height, width = annotated.shape[:2]

        for damage in damages:
            bbox = damage["bbox"]

            #
            # normalized -> pixel
            #

            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            damage_type = damage.get(
                "damage_type",
                "damage"
            )
            damage_type = damage_dict[damage_type]

            confidence = damage.get(
                "confidence",
                0.0
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            label = (
                f"{damage_type}: "
                f"{confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        return annotated

    #
    # save image helper
    #

    def save_image(self, image, filename):

        if image is None:
            raise ValueError(
                "save_image received None image"
            )

        path = (self.images_dir / filename).resolve()

        success = cv2.imwrite(str(path), image)

        if not success:
            raise RuntimeError(
                f"Failed to save image: {path}"
            )

        print(
            f"[INFO] Image saved: {path}"
        )

        return str(path)

    #
    # main report generation
    #

    def generate_report(
            self,
            detector_output,
            gemma_description,
            qwen_output,
            frames_for_gemma=None,
            qwen_frame=None,
    ):

        report_id = str(uuid.uuid4())

        pdf_path = (
                self.output_dir
                / f"report_{report_id}.pdf"
        )

        #
        # absolute path
        #

        pdf_path = pdf_path.resolve()

        print(
            f"[INFO] Creating PDF report: "
            f"{pdf_path}"
        )

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        #
        # fonts
        #

        styles["BodyText"].fontName = (
            "TimesNewRoman"
        )

        styles["Title"].fontName = (
            "TimesNewRoman"
        )

        styles["Heading1"].fontName = (
            "TimesNewRoman"
        )

        elements = []

        #
        # title
        #

        elements.append(
            Paragraph(
                "Отчёт о ДТП",
                styles["Title"]
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        #
        # detector section
        #

        elements.append(
            Paragraph(
                "Детекция ДТП",
                styles["Heading1"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        #
        # detector frames
        #

        if (
                frames_for_gemma is not None
                and len(frames_for_gemma) > 0
        ):

            print(
                f"[INFO] Saving "
                f"{len(frames_for_gemma)} "
                f"detector frames"
            )

            image_paths = []

            for idx, frame in enumerate(
                    frames_for_gemma
            ):

                if frame is None:
                    print(
                        f"[WARNING] Frame {idx} "
                        f"is None"
                    )

                    continue

                frame_path = self.save_image(
                    frame,
                    (
                        f"{report_id}"
                        f"_detector_{idx}.png"
                    )
                )

                image_paths.append(
                    frame_path
                )

            #
            # images row
            #

            image_row = []

            for path in image_paths:
                print(
                    f"[INFO] Adding image "
                    f"to PDF: {path}"
                )

                image_row.append(

                    RLImage(
                        str(path),
                        width=170,
                        height=120
                    )
                )

            if len(image_row) > 0:
                table = Table(
                    [image_row]
                )

                table.setStyle(
                    TableStyle([
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            1,
                            colors.black
                        ),

                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER"
                        ),
                    ])
                )

                elements.append(
                    table
                )

                elements.append(
                    Spacer(1, 15)
                )

        #
        # confidence
        #

        elements.append(
            Paragraph(
                (
                    "Уверенность "
                    "детектора: "
                    f"{detector_output['confidence']:.3f}"
                ),
                styles["BodyText"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        #
        # gemma
        #

        elements.append(
            Paragraph(
                (
                    "<b>Описание ДТП "
                    "от Gemma:</b>"
                ),
                styles["BodyText"]
            )
        )

        elements.append(
            Spacer(1, 5)
        )

        elements.append(
            Paragraph(
                gemma_description,
                styles["BodyText"]
            )
        )

        elements.append(
            Spacer(1, 25)
        )

        #
        # qwen section
        #

        elements.append(
            Paragraph(
                "Анализ повреждений",
                styles["Heading1"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        #
        # damage image
        #

        if qwen_frame is not None:
            print(
                "[INFO] Drawing damage boxes"
            )

            damages = qwen_output.get(
                "damages",
                []
            )

            annotated_damage_frame = (
                self.draw_damage_boxes(
                    qwen_frame,
                    damages
                )
            )

            damage_image_path = (
                self.save_image(
                    annotated_damage_frame,
                    (
                        f"{report_id}"
                        f"_damage.png"
                    )
                )
            )

            elements.append(

                RLImage(
                    str(damage_image_path),
                    width=400,
                    height=300
                )
            )

            elements.append(
                Spacer(1, 15)
            )

        #
        # damages list
        #

        damages = qwen_output.get(
            "damages",
            []
        )

        if len(damages) == 0:

            # elements.append(
            #     Paragraph(
            #         (
            #             "<b>Найденные "
            #             "повреждения:</b>"
            #         ),
            #         styles["BodyText"]
            #     )
            # )
            #
            # elements.append(
            #     Spacer(1, 5)
            # )
            #
            # for damage in damages:
            #     damage_type = damage.get(
            #         "damage_type",
            #         "unknown"
            #     )
            #
            #     confidence = damage.get(
            #         "confidence",
            #         0.0
            #     )
            #
            #     bbox = damage.get(
            #         "bbox",
            #         []
            #     )
            #
            #     text = (
            #         f"- {damage_type} "
            #         f"(confidence="
            #         f"{confidence:.2f}) "
            #         f"bbox={bbox}"
            #     )
            #
            #     elements.append(
            #         Paragraph(
            #             text,
            #             styles["BodyText"]
            #         )
            #     )
        #
        # else:

            elements.append(
                Paragraph(
                    "Повреждений "
                    "не обнаружено.",
                    styles["BodyText"]
                )
            )

        elements.append(
            Spacer(1, 15)
        )

        #
        # severity
        #

        severity_score = qwen_output.get(
            "severity_score",
            "unknown"
        )

        elements.append(
            Paragraph(
                (
                    "<b>Оценка "
                    "серьёзности:</b> "
                    f"{severity_score}/10"
                ),
                styles["BodyText"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        #
        # complex description
        #

        complex_description = (
            qwen_output.get(
                "complex_description",
                "Описание отсутствует."
            )
        )

        elements.append(
            Paragraph(
                (
                    "<b>Общее описание "
                    "повреждений:</b>"
                ),
                styles["BodyText"]
            )
        )

        elements.append(
            Spacer(1, 5)
        )

        elements.append(
            Paragraph(
                complex_description,
                styles["BodyText"]
            )
        )

        #
        # build pdf
        #

        print(
            "[INFO] Building PDF..."
        )

        doc.build(elements)

        print(
            f"[INFO] PDF saved: "
            f"{pdf_path}"
        )

        return str(pdf_path)
