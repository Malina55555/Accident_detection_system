import time
import cv2

from ultralytics import YOLO
from ultralytics import RTDETR

from app.config import settings


class AccidentDetector:

    def __init__(self):

        if settings.detector_model == "yolo":

            print(
                f"[INFO] Loading YOLO model: "
                f"{settings.yolo_model_path}"
            )

            self.model = YOLO(
                settings.yolo_model_path
            )

        else:

            print(
                f"[INFO] Loading RT-DETR model: "
                f"{settings.rtdetr_model_path}"
            )

            self.model = RTDETR(
                settings.rtdetr_model_path
            )

    def infer(
        self,
        frame
    ):

        start = time.time()

        #
        # inference
        #

        results = self.model.predict(
            frame,
            verbose=False
        )

        inference_time = (
            time.time() - start
        )

        #
        # defaults
        #

        detected = False

        best_confidence = 0.0

        all_boxes = []

        annotated_frame = frame.copy()

        #
        # parse detections
        #

        for result in results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(
                    box.conf[0]
                )

                #
                # threshold filtering
                #

                if (
                    confidence
                    < settings.accident_threshold
                ):
                    continue

                detected = True

                #
                # best confidence
                #

                if confidence > best_confidence:
                    best_confidence = confidence

                #
                # bbox
                #

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                #
                # class id
                #

                class_id = int(
                    box.cls[0]
                )

                #
                # detection info
                #

                detection_info = {

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "confidence":
                        confidence,

                    "class_id":
                        class_id
                }

                all_boxes.append(
                    detection_info
                )

                #
                # draw bbox
                #

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                #
                # label
                #

                label = (
                    f"ACCIDENT "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 0)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        #
        # logging
        #

        if detected:

            print(
                f"[INFO] Accident detected "
                f"(conf={best_confidence:.3f})"
            )

        #
        # return
        #

        return {

            "detected":
                detected,

            "confidence":
                best_confidence,

            "inference_time":
                inference_time,

            "results":
                results,

            #
            # all detections
            #

            "boxes":
                all_boxes,

            #
            # frame with bbox
            #

            "annotated_frame":
                annotated_frame
        }