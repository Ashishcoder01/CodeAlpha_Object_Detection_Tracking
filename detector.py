from pathlib import Path

import cv2
from ultralytics import YOLO


class ObjectDetector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.25,
    ):

        project_dir = Path(__file__).resolve().parent

        model_path = Path(model_path)

        if not model_path.is_absolute():
            model_path = project_dir / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.model_path = str(model_path)
        self.confidence = confidence

        self.model = YOLO(
            self.model_path
        )

    def detect_image(
        self,
        image,
    ):

        if image is None:
            raise ValueError(
                "Image is empty."
            )

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False,
            save=False,
        )

        annotated_image = image.copy()

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                class_name = self.model.names[
                    class_id
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    }
                )

                # Bounding box

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                # Label

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated_image,
                    label,
                    (
                        x1,
                        max(
                            y1 - 10,
                            25,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return (
            annotated_image,
            detections,
        )

    def get_class_counts(
        self,
        detections,
    ):

        counts = {}

        for detection in detections:

            name = detection[
                "class_name"
            ]

            counts[name] = (
                counts.get(
                    name,
                    0,
                )
                + 1
            )

        return counts