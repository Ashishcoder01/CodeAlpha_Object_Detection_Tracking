from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, model_path="yolo11n.pt", confidence=0.40):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def track_frame(self, frame):
        """
        Track objects in a video frame.

        YOLO maintains tracking IDs between frames.
        """

        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            verbose=False
        )

        tracked_objects = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = self.model.names[class_id]

                tracked_objects.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return results, tracked_objects