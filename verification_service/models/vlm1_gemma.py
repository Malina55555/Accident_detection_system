# verification_service/models/vlm1_gemma.py
import base64
import cv2
from io import BytesIO
from PIL import Image
import torch


class VLM1Gemma:
    """VLM1 для верификации ДТП (Gemma)"""

    def __init__(self, model_name, use_real=False):
        self.model_name = model_name
        self.use_real = use_real
        self.model = None
        self.processor = None

        if use_real:
            self._load_model()

    def _load_model(self):
        """Загрузка реальной модели Gemma"""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            print(f"[VLM1] Loading Gemma model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name)
            print("[VLM1] Gemma model loaded successfully")
        except Exception as e:
            print(f"[VLM1] Failed to load Gemma model: {e}")
            print("[VLM1] Falling back to stub mode")
            self.use_real = False

    def verify_accident(self, frames_base64, timestamps, bboxes):
        """
        Проверяет, действительно ли на кадрах ДТП

        Args:
            frames_base64: list of base64 encoded frames
            timestamps: list of timestamps
            bboxes: list of bboxes

        Returns:
            dict: {"verdict": bool, "description": str}
        """
        if not self.use_real:
            return self._verify_stub(frames_base64, timestamps, bboxes)

        # Реальная логика с Gemma
        try:
            import torch
            from PIL import Image
            from io import BytesIO

            # base64 -> PIL
            images = []
            for b64 in frames_base64:
                if b64:
                    images.append(
                        Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    )

            # Формируем промпт
            prompt = """
                        Проанализируй 5 последовательных кадров, определённых моделью YOLO как авария. 
                        Проследи за передвижениями участников ДТП и определи каким образом произошла авария.
                        Определи - действительно ли на изображениях развернулась авария.

                        Отвачай только в формате JSON без дополнительных пометок и markdown:

                        {
                            "verdict": true,
                            "confidence": 0.85,
                            "description": "Описание динамики и участников аварии на русском языке"
                        }
                        """
            messages = [
                {
                    "role": "user",
                    "content": (
                            [{"type": "image"} for _ in images]
                            + [{"type": "text", "text": prompt}]
                    ),
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
            )

            # если модель на GPU
            inputs = {
                k: v.to(self.model.device)
                if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )

            try:
                # Парсим JSON из ответа
                import re, json
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    if result.get("verdict", False) and result.get("description", ""):
                        return {"verdict": result.get("verdict", False), "description": result.get("description", "")}
                return {"verdict": True, "description": response}
            except Exception as e:
                print(f"[VLM1] Error: {e}")
                return self._verify_stub()

            # Пока возвращаем заглушку
            #return {"verdict": True, "description": "Real accident detected"}

        except Exception as e:
            print(f"[VLM1] Error during inference: {e}")
            return {"verdict": False, "description": f"Error: {e}"}

    def _verify_stub(self, frames_base64=None, timestamps=None, bboxes=None):
        """Заглушка VLM1"""
        print("[VLM1 STUB] Verifying accident...")

        return {
            "verdict": True,
            "confidence": 0.95,
            "description":  "Красный автомобиль, двигавшийся по дороге, столкнулся с серебристым седаном, который ехал по дороге. Оба автомобиля находятся на мокрой дороге, что может влиять на управляемость. Похоже, произошло столкновение между двумя транспортными средствами.",
            "severity_score": 4
        }

    def describe_damages(self, frame_base64, damages):
        """
        Описывает повреждения на кадре

        Args:
            frame_base64: base64 encoded frame
            damages: list of damages from RTDETR

        Returns:
            dict: {
                "damages": list,
                "severity_score": int,
                "complex_description": str
            }
        """
        if not self.use_real:
            return self._describe_stub(frame_base64, damages)

        try:
            from PIL import Image
            from io import BytesIO
            # base64 -> PIL

            img = Image.open(BytesIO(base64.b64decode(frame_base64))).convert("RGB")

            damages_info = ""
            if damages:
                damages_info = f"Pre-detected damages: {damages}\n"

            prompt = f"""
                {damages_info}
                Analyze car damage in this image.

                Return JSON:
                {{
                    "damages": [{{"bbox": [x1,y1,x2,y2], "damage_type": "dent/crack/scratch/glass_shatter/tire_flat/lamp_broken", "confidence": 0.0-1.0}}],
                    "severity_score": 0-10,
                    "complex_description": "detailed damage description in Russian"
                }}
                """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                          #  "image": img,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=text,
                images=img,
                return_tensors="pt",
            )

            # если модель на GPU
            inputs = {
                k: v.to(self.model.device)
                if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )

            damage_dict = {
                "dent": "вмятина",
                "crack": "трещина",
                "scratch": "царапина",
                "glass_shatter": "разбитое стекло",
                "tire_flat": "спущенная шина",
                "lamp_broken": "разбитая лампа"
            }

            try:
                # Парсим JSON из ответа
                import re, json
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    #return result
                    formatted_damages = []
                    max_severity = 0
                    result.get("damages")

                    for d in damages:
                        damage_type_en = d.get("class", "damage")
                        damage_type_ru = damage_dict.get(damage_type_en, damage_type_en)
                        # Нормализуем bbox (если они абсолютные)
                        bbox = d["bbox"]
                        norm_bbox = bbox  # нормализованные координаты
                        confidence = d["confidence"]
                        formatted_damages.append({
                            "bbox": norm_bbox,
                            "damage_type": damage_type_ru,
                            "damage_type_en": damage_type_en,
                            "confidence": confidence
                        })

                        # Оценка серьёзности на основе уверенности и типа повреждения
                        severity = int(confidence * 10)
                        if damage_type_en in ["glass_shatter", "lamp_broken"]:
                            severity = min(10, severity + 2)
                        max_severity = max(max_severity, severity)

                    comlex_description = result.get("complex_description")
                    return {
                        "damages": formatted_damages,
                        "severity_score": max_severity,
                        "complex_description": comlex_description
                    }
                return {"damages": [], "severity_score": 5, "complex_description": response}
            except Exception as e:
                print(f"[VLM1] Error: {e}")
                return self._verify_stub()
        except Exception as e:
            print(f"[VLM1] Error during inference: {e}")
            return self._describe_stub(frame_base64, damages)

    def _describe_stub(self, frame_base64, damages):
        """Заглушка VLM2 с правильными названиями повреждений"""
        print("[VLM2 STUB] Describing damages...")

        # Конвертируем damage class в понятные русские названия
        damage_dict = {
            "dent": "вмятина",
            "crack": "трещина",
            "scratch": "царапина",
            "glass_shatter": "разбитое стекло",
            "tire_flat": "спущенная шина",
            "lamp_broken": "разбитая лампа"
        }

        formatted_damages = []
        max_severity = 0

        for d in damages:
            damage_type_en = d.get("class", "damage")
            damage_type_ru = damage_dict.get(damage_type_en, damage_type_en)

            # Нормализуем bbox (если они абсолютные)
            bbox = d["bbox"]
            # if max(bbox) > 1.0:  # Если координаты абсолютные
            #     # Сохраняем как есть, ReportService сам обработает
            #     norm_bbox = bbox
            # else:
            norm_bbox = bbox  # нормализованные координаты

            confidence = d["confidence"]

            formatted_damages.append({
                "bbox": norm_bbox,
                "damage_type": damage_type_ru,
                "damage_type_en": damage_type_en,
                "confidence": confidence
            })

        complex_description = ""
        # Формируем описание
        if len(damages) == 0:
            complex_description = "Повреждений не обнаружено. Автомобиль(и) на изображении не имеют видимых дефектов."
        else:
            damage_list = ", ".join([d["damage_type"] for d in formatted_damages])
            complex_description = f"На автомобиле(ях) обнаружены следующие повреждения: {damage_list}. "
            complex_description += f"Серьёзность повреждений оценивается как {max_severity} из 10."

        return {

            "severity_score": 4,  #max_severity,
            "complex_description":
                #"Обе машины серьезно повреждены: красная машина с разбитыми фарами и лампой, а также с поврежденной левой стороной, включая лампу и колесо, а также смятые боковые части кузова. Белый автомобиль также имеет серьезные повреждения, включая смятые боковые части кузова и поврежденную левую сторону. "
                "Желтый автомобиль в центре имеет повреждения на передней части, включая, по-видимому, повреждение бампера или передней части. Красный автомобиль справа имеет заметные повреждения на передней части, включая вмятины и повреждения кузова. Вокруг машин видны осколки стекла и остатки разбитых деталей.",
        }

