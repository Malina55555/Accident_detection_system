import json
import time

import torch

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from app.config import settings


class QwenService:

    def __init__(self):

        self.use_stub = settings.use_qwen_stub

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if not self.use_stub:

            print(
                "[INFO] Loading Qwen model..."
            )

            self.processor = (
                AutoProcessor.from_pretrained(
                    settings.qwen_path
                )
            )

            self.model = (
                Qwen3VLForConditionalGeneration
                .from_pretrained(
                    settings.qwen_path,
                    torch_dtype=torch.float16
                    if self.device == "cuda"
                    else torch.float32,
                    device_map="auto"
                )
            )

            print(
                "[INFO] Qwen model loaded"
            )

    def analyze_damage(
        self,
        frame
    ):

        #
        # STUB
        #

        if self.use_stub:

            print(
                "[INFO] Stub damage analysis started"
            )

            time.sleep(
                settings.qwen_stub_sleep_sec
            )

            return {
              "damages": [
                {
                  "bbox": [0.481, 0.455, 0.539, 0.543],
                  "damage_type": "dent",
                  "confidence": 0.87
                },
                {
                  "bbox": [0.557, 0.385, 0.607, 0.445],
                  "damage_type": "crack",
                  "confidence": 0.87
                },
                {
                  "bbox": [0.36, 0.285, 0.393, 0.325],
                  "damage_type": "lamp_broken",
                  "confidence": 0.87
                },
                {
                  "bbox": [0.357, 0.301, 0.38, 0.34],
                  "damage_type": "lamp_broken",
                  "confidence": 0.87
                }
              ],
              "complex_description": "Повреждения: 1. Вмятина на передней части фиолетового автомобиля. 2. Разбитое лобовое стекло фиолетового автомобиля. 3. Разбитая лампа на фиолетовом автомобиле. 4. Разбитая лампа на серебристом автомобиле.",
              "severity_score": 1
            }

        #
        # REAL MODEL
        #

        print(
            "[INFO] Qwen damage analysis started"
        )

        #
        # IMPORTANT:
        # forcing strict JSON output
        #

        prompt = """
You are a vehicle damage assessment system.
Analyze the provided image and identify visible vehicle damage instances.

IMPORTANT CONSTRAINTS:
- One continuous damaged area = ONE bounding box only
- Minimum bounding box area: at least 0.005 (normalized)
- Do NOT detect: reflections, shadows, dirt, water drops, or normal wear
- Be conservative: false positives are worse than false negatives

For EACH detected damage, output:
- Bounding box coordinates (normalized 0.000-1.000, 3 decimal places)
- Damage type classification
- Confidence score (float between 0.00 and 1.00, 2 decimal places)

Damage types: dent, crack, scratch, glass_shatter, tire_flat, lamp_broken

Output format (JSON array only):
{
  "damages": [
    {
      "bbox": [x_min, y_min, x_max, y_max],
      "damage_type": "string",
      "confidence": 0.00
    }
  ],
  "complex_description": "string",
  "severity_score": 1
}

If no damage detected, return:
{
  "damages": [],
  "complex_description": "Повреждений не обнаружено",
  "severity_score": 1
}

CRITICAL RULES:
- Do NOT copy any example values
- Do NOT include markdown, code blocks, or explanations
- Output ONLY the JSON array
- severity_score must be integer from 1 to 10
- confidence must be from 0.0 to 1.0

Return ONLY JSON.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": frame
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        #
        # chat template
        #

        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        #
        # processor
        #

        inputs = self.processor(
            text=[text],
            images=[frame],
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        #
        # generation
        #

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        #
        # decode
        #

        output_text = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
        )

        print(
            "[INFO] Raw Qwen output:"
        )

        print(output_text)

        #
        # extract JSON
        #

        try:

            #
            # sometimes models return extra text
            #

            json_start = output_text.find("{")

            json_end = output_text.rfind("}")

            if (
                json_start == -1
                or json_end == -1
            ):
                raise ValueError(
                    "JSON not found"
                )

            json_text = output_text[
                json_start:json_end + 1
            ]

            parsed = json.loads(json_text)

            #
            # validation
            #

            if "damages" not in parsed:
                parsed["damages"] = []

            if (
                "complex_description"
                not in parsed
            ):
                parsed[
                    "complex_description"
                ] = (
                    "Описание отсутствует"
                )

            if (
                "severity_score"
                not in parsed
            ):
                parsed[
                    "severity_score"
                ] = 1

            #
            # bbox normalization
            #

            for damage in parsed["damages"]:

                bbox = damage.get(
                    "bbox",
                    []
                )

                if len(bbox) == 4:

                    image_height, image_width = (
                        frame.shape[:2]
                    )

                    if any(v > 1.0 for v in bbox):
                        x1, y1, x2, y2 = bbox

                        bbox = [

                            float(x1) / image_width,
                            float(y1) / image_height,

                            float(x2) / image_width,
                            float(y2) / image_height,
                        ]

                    bbox = [

                        max(0.0, min(1.0, float(v)))

                        for v in bbox
                    ]

                    x1, y1, x2, y2 = bbox

                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    damage["bbox"] = [
                        x1, y1, x2, y2
                    ]

                damage["confidence"] = float(
                    damage.get(
                        "confidence",
                        0.0
                    )
                )

            print(
                "[INFO] Parsed Qwen JSON successfully"
            )

            return parsed

        except Exception as e:

            print(
                "[ERROR] Failed to parse "
                f"Qwen output: {e}"
            )

            #
            # fallback
            #

            return {

                "damages": [],

                "complex_description":
                    (
                        "Не удалось корректно "
                        "распознать ответ модели."
                    ),

                "severity_score": 1
            }