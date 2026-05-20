import cv2
import time
from collections import deque

from app.pipeline.detector import AccidentDetector
from app.pipeline.gemma_service import GemmaService
from app.pipeline.qwen_service import QwenService
from app.pipeline.report_service import ReportService

from app.config import settings


class StreamProcessor:
    def __init__(self):
        self.detector = AccidentDetector()
        self.gemma = GemmaService()
        self.qwen = QwenService()
        self.report_service = ReportService()

        self.last_accident_time = 0
        self.last_detected_accident_time = 0

    def process(self):
        if settings.input_source_type == "video":
            self._process_video()

        elif settings.input_source_type == "webcam":
            self._process_webcam()

        elif settings.input_source_type == "rtsp":
            self._process_rtsp()
        else:
            raise ValueError(f"Unsupported source type: {settings.input_source_type}")

    def _handle_frame(
            self,
            frame,
            frame_buffer,
            cap=None
    ):

        #
        # first frame inference
        #

        detector_output = self.detector.infer(
            frame
        )

        frame_buffer.append(frame)

        cooldown_sec = (
            settings.wait_for_next_accident_minimum
        )

        now = time.time()

        #
        # anti-spam cooldown
        #

        if (
                not detector_output["detected"]
                or now - self.last_accident_time
                <= cooldown_sec
        ):
            return detector_output

        print(
            "[INFO] First accident frame detected. "
            "Starting temporal validation..."
        )

        #
        # validation buffers
        #

        validation_frames = [frame]

        validation_detector_outputs = [
            detector_output
        ]

        detected_count = 1

        #
        # wait between validation frames
        #

        wait_sec = settings.check_wait_sec

        #
        # need 2 more detected frames
        #

        for i in range(2):

            if cap is None:
                print(
                    "[WARNING] cap is None"
                )

                return detector_output

            print(
                f"[INFO] Waiting "
                f"{wait_sec} sec before "
                f"validation frame {i + 1}"
            )

            #
            # WAIT
            #

            if settings.input_source_type == "video":

                #
                # for video:
                # skip frames according to fps
                #

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                frames_to_skip = int(
                    wait_sec * fps
                )

                for _ in range(frames_to_skip):
                    cap.grab()

            else:

                #
                # for webcam / rtsp:
                # real waiting + frame grabbing
                #

                target_time = (
                        time.time() + wait_sec
                )

                while time.time() < target_time:
                    cap.grab()

            #
            # read validation frame
            #

            ret, next_frame = cap.read()

            if not ret:
                print(
                    "[WARNING] Failed to read "
                    "validation frame"
                )

                return detector_output

            frame_buffer.append(next_frame)

            #
            # detector inference
            #

            next_detector_output = (
                self.detector.infer(
                    next_frame
                )
            )

            #
            # validation failed
            #

            if (
                    not next_detector_output[
                        "detected"
                    ]
            ):
                print(
                    f"[INFO] Validation frame "
                    f"{i + 1}: NOT detected"
                )

                return detector_output

            #
            # validation success
            #

            print(
                f"[INFO] Validation frame "
                f"{i + 1}: detected"
            )

            detected_count += 1

            validation_frames.append(
                next_frame
            )

            validation_detector_outputs.append(
                next_detector_output
            )

        #
        # all 3 frames detected
        #

        if detected_count == 3:
            self.last_accident_time = now

            print(
                "[INFO] Accident confirmed "
                "on 3 consecutive frames"
            )

            #
            # annotated frames
            #

            annotated_frames = [

                x["annotated_frame"]

                for x in validation_detector_outputs
            ]

            #
            # gemma
            #

            gemma_description = (
                self.gemma.describe_accident(
                    validation_frames
                )
            )

            #
            # qwen
            #

            qwen_frame = validation_frames[-1]

            qwen_output = (
                self.qwen.analyze_damage(
                    qwen_frame
                )
            )

            #
            # report
            #

            pdf_path = (
                self.report_service.generate_report(
                    detector_output=
                    detector_output,

                    gemma_description=
                    gemma_description,

                    qwen_output=
                    qwen_output,

                    frames_for_gemma=
                    annotated_frames,

                    qwen_frame=
                    qwen_frame,
                )
            )

            print(
                f"[INFO] Report saved: "
                f"{pdf_path}"
            )

        return detector_output

    def _process_video(self):

        print(
            f"[INFO] Opening video: "
            f"{settings.video_path}"
        )

        cap = cv2.VideoCapture(
            settings.video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: "
                f"{settings.video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        print(f"[INFO] Video FPS: {fps}")

        frame_buffer = deque(maxlen=3)

        current_frame_idx = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                print("[INFO] End of video")
                break

            detector_output = self._handle_frame(
                frame=frame,
                frame_buffer=frame_buffer,
                cap=cap,
            )

            inference_time = detector_output[
                "inference_time"
            ]

            #
            # skip кадров на основе времени инференса
            #

            skip_frames = int(
                inference_time * fps
            )

            current_frame_idx += skip_frames

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                current_frame_idx
            )

        cap.release()

        print("[INFO] Video processing completed")

    def _process_webcam(self):

        print(
            f"[INFO] Opening webcam: "
            f"{settings.webcam_index}"
        )

        cap = cv2.VideoCapture(
            settings.webcam_index
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam: "
                f"{settings.webcam_index}"
            )

        frame_buffer = deque(maxlen=3)

        while True:

            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Webcam frame read failed")
                break

            detector_output = self._handle_frame(
                frame=frame,
                frame_buffer=frame_buffer,
                cap=cap,
            )

            inference_time = detector_output[
                "inference_time"
            ]

            #
            # realtime frame dropping
            #

            target_time = (
                time.time() + inference_time
            )

            while time.time() < target_time:
                cap.grab()

            #
            # ESC -> exit
            #

            cv2.imshow(
                "Traffic Accident Detection",
                frame
            )

            key = cv2.waitKey(1)

            if key == 27:
                print("[INFO] ESC pressed")
                break

        cap.release()

        cv2.destroyAllWindows()

        print("[INFO] Webcam processing completed")


    def _process_rtsp(self):

        print(
            f"[INFO] Opening RTSP stream: "
            f"{settings.rtsp_url}"
        )

        cap = cv2.VideoCapture(
            settings.rtsp_url
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open RTSP stream: "
                f"{settings.rtsp_url}"
            )

        frame_buffer = deque(maxlen=3)

        reconnect_delay_sec = 5

        while True:

            ret, frame = cap.read()

            #
            # reconnect logic
            #

            if not ret:
                print(
                    "[WARNING] RTSP frame read failed. "
                    "Reconnect attempt..."
                )

                cap.release()

                time.sleep(reconnect_delay_sec)

                cap = cv2.VideoCapture(
                    settings.rtsp_url
                )

                continue

            detector_output = self._handle_frame(
                frame=frame,
                frame_buffer=frame_buffer,
                cap=cap,
            )

            inference_time = detector_output[
                "inference_time"
            ]

            #
            # realtime frame dropping
            #

            target_time = (
                    time.time() + inference_time
            )

            while time.time() < target_time:
                cap.grab()

            #
            # preview
            #

            cv2.imshow(
                "Traffic Accident Detection",
                frame
            )

            key = cv2.waitKey(1)

            if key == 27:
                print("[INFO] ESC pressed")
                break

        cap.release()

        cv2.destroyAllWindows()

        print("[INFO] RTSP processing completed")


        # fps = cap.get(cv2.CAP_PROP_FPS)
        #
        # frame_buffer = deque(maxlen=3)
        #
        # current_frame_idx = 0
        #
        # while True:
        #     ret, frame = cap.read()
        #
        #     if not ret:
        #         break
        #
        #     detector_output = self.detector.infer(frame)
        #
        #     frame_buffer.append(frame)
        #
        #     if detector_output["detected"]:
        #         print("Accident detected")
        #
        #         additional_frames = []
        #
        #         for _ in range(2):
        #             ret2, frame2 = cap.read()
        #
        #             if not ret2:
        #                 break
        #
        #             frame_buffer.append(frame2)
        #             additional_frames.append(frame2)
        #
        #         frames_for_gemma = list(frame_buffer)
        #
        #         gemma_description = self.gemma.describe_accident(
        #             frames_for_gemma
        #         )
        #
        #         qwen_output = self.qwen.analyze_damage(
        #             frame_buffer[-1]
        #         )
        #
        #         pdf_path = self.report_service.generate_report(
        #             detector_output=detector_output,
        #             gemma_description=gemma_description,
        #             qwen_output=qwen_output,
        #         )
        #
        #         print(f"Report saved: {pdf_path}")
        #
        #     inference_time = detector_output["inference_time"]
        #
        #     skip_frames = int(inference_time * fps)
        #
        #     current_frame_idx += skip_frames
        #
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        #
        # cap.release()
