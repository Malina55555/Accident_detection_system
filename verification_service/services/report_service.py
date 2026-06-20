# verification_service/services/report_service.py
import uuid
import cv2
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle
)

from config import *

# Регистрация русского шрифта
try:
    pdfmetrics.registerFont(TTFont("TimesNewRoman", FONTS_PATH))
except:
    print("[WARNING] Font not found, using default")
    # Используем стандартный шрифт
    pdfmetrics.registerFont(TTFont("TimesNewRoman", "times.ttf"))


class ReportService:
    def __init__(self):
        self.output_dir = Path(REPORT_SAVE_PATH)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True, parents=True)

    def draw_damage_boxes(self, frame, damages):
        """
        Рисует bbox повреждений на кадре (для RTDETR)
        damages содержат нормализованные координаты (0-1)
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        for damage in damages:
            #damage - bbox [x1, y1, x2, y2] нормализованные, confidence, class (damage_type)
            bbox = damage["bbox"]
            # Координаты нормализованные (0-1) -> конвертируем в пиксели
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            damage_type = damage.get("class", "damage")
            # damage_type = damage.get("damage_type_en", "damage")

            confidence = damage.get("confidence", 0.0)

            # Надпись на английском
            label = f"{damage_type.upper()}: {confidence:.2f}"

            # Рисуем прямоугольник
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Рисуем фон для текста
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated,
                          (x1, y1 - label_size[1] - 5),
                          (x1 + label_size[0] + 5, y1),
                          (0, 0, 255),
                          -1)

            # Рисуем текст
            cv2.putText(annotated, label,
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated

    def draw_accident_boxes(self, frame, bbox):
        """
        Рисует bbox ДТП на кадре (для YOLO/RTDETR из detector_service)

        Args:
            frame: кадр OpenCV
            bbox: [x1, y1, x2, y2] или None # координаты в пикселях
        """
        if bbox is None:
            return frame.copy()

        annotated = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)

        # Рисуем прямоугольник (зелёный для ДТП)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Добавляем надпись "ACCIDENT"
        cv2.putText(annotated, "ACCIDENT",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

    def save_image(self, image, filename):
        """Сохраняет изображение и возвращает путь"""
        if image is None:
            raise ValueError("save_image received None image")

        path = (self.images_dir / filename).resolve()
        success = cv2.imwrite(str(path), image)

        if not success:
            raise RuntimeError(f"Failed to save image: {path}")

        print(f"[INFO] Image saved: {path}")
        return str(path)

    def generate_report(self, detector_verdict, gemma_description,
                        qwen_output, accident_frames_with_bbox, damage_frame_with_bbox):
        """
        Генерирует PDF отчёт

        Args:
            detector_verdict: dict с результатами детектора (confidence, source)
            gemma_description: str описание от Gemma
            qwen_output: dict с результатами Qwen (damages, severity_score, complex_description)
            accident_frames_with_bbox: list of (frame, bbox) для 5 кадров с ДТП
            damage_frame_with_bbox: (frame, damages) кадр с повреждениями
        """
        report_id = str(uuid.uuid4())
        pdf_path = self.output_dir / f"report_{report_id}.pdf"
        pdf_path = pdf_path.resolve()

        print(f"[INFO] Creating PDF report: {pdf_path}")

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()

        # Настройка шрифтов
        styles["BodyText"].fontName = "TimesNewRoman"
        styles["Title"].fontName = "TimesNewRoman"
        styles["Heading1"].fontName = "TimesNewRoman"
        elements = []
        # Заголовок
        elements.append(Paragraph("Отчёт о ДТП", styles["Title"]))
        elements.append(Spacer(1, 20))

        # ==========================================================
        # Секция детекции ДТП
        # ==========================================================
        elements.append(Paragraph("1. Детекция ДТП", styles["Heading1"]))
        elements.append(Spacer(1, 10))

        # 5 кадров с bbox ДТП
        if accident_frames_with_bbox:
            image_paths = []

            for idx, (frame, bbox) in enumerate(accident_frames_with_bbox):
                if frame is not None:
                    # Рисуем bbox ДТП на кадре
                    annotated_frame = self.draw_accident_boxes(frame, bbox)
                    frame_path = self.save_image(
                        annotated_frame,
                        f"{report_id}_accident_{idx + 1}.png"
                    )
                    image_paths.append(frame_path)

            # Отображаем кадры в таблице (2 ряда: 3 + 2)
            if image_paths:
                # Первый ряд (первые 3 кадра)
                row1 = []
                for i in range(min(3, len(image_paths))): #0,1,2
                    row1.append(RLImage(image_paths[i], width=170, height=120))

                table1 = Table([row1])
                table1.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                elements.append(table1)
                elements.append(Spacer(1, 10))

                # Второй ряд (следующие 2 кадра)
                if len(image_paths) > 3:
                    row2 = []
                    for i in range(3, min(5, len(image_paths))): #3,4
                        row2.append(RLImage(image_paths[i], width=170, height=120))

                    # # Добавляем пустые ячейки для выравнивания
                    # while len(row2) < 3:
                    #     row2.append(Spacer(170, 120))

                    table2 = Table([row2])
                    table2.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]))
                    elements.append(table2)
                    elements.append(Spacer(1, 15))

        # Информация о детекторе
        elements.append(Paragraph(
            f"<b>Уверенность детектора:</b> {int(detector_verdict.get('confidence', 0.5)*10)}/10",
            styles["BodyText"]
        ))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(
            f"<b>Источник:</b> {detector_verdict.get('source', 'unknown')}",
            styles["BodyText"]
        ))
        elements.append(Spacer(1, 10))

        # Описание ДТП от Gemma
        elements.append(Paragraph("<b>Описание ДТП:</b>", styles["BodyText"]))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(gemma_description, styles["BodyText"]))
        elements.append(Spacer(1, 25))

        # ==========================================================
        # Секция анализа повреждений
        # ==========================================================
        elements.append(Paragraph("2. Анализ повреждений", styles["Heading1"]))
        elements.append(Spacer(1, 10))

        # Кадр с повреждениями
        if damage_frame_with_bbox:
            damage_frame, damages = damage_frame_with_bbox
            if damage_frame is not None:
                # Рисуем bbox повреждений на кадре
                annotated_damage_frame = self.draw_damage_boxes(damage_frame, damages)
                damage_image_path = self.save_image(
                    annotated_damage_frame,
                    f"{report_id}_damage.png"
                )

                # Добавляем изображение в PDF
                img = RLImage(damage_image_path, width=400, height=300)
                elements.append(img)
                elements.append(Spacer(1, 15))

        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Пояснение к обозначениям:</b>", styles["BodyText"]))
        elements.append(Spacer(1, 5))

        legend_text = """
                <font size="9">
                • DENT — вмятина<br/>
                • CRACK — трещина<br/>
                • SCRATCH — царапина<br/>
                • GLASS_SHATTER — разбитое стекло<br/>
                • TIRE_FLAT — спущенная шина<br/>
                • LAMP_BROKEN — разбитая лампа
                </font>
                """
        elements.append(Paragraph(legend_text, styles["BodyText"]))

        elements.append(Spacer(1, 15))

        # Оценка серьёзности
        severity_score = qwen_output.get("severity_score", "N/A")
        elements.append(Paragraph(
            f"<b>Оценка серьёзности:</b> {severity_score}/10",
            styles["BodyText"]
        ))
        elements.append(Spacer(1, 10))

        # Общее описание от Qwen
        complex_description = qwen_output.get("complex_description", "Описание отсутствует.")
        elements.append(Paragraph("<b>Общее описание повреждений:</b>", styles["BodyText"]))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(complex_description, styles["BodyText"]))

        # Сборка PDF
        print("[INFO] Building PDF...")
        doc.build(elements)
        print(f"[INFO] PDF saved: {pdf_path}")

        return str(pdf_path)
