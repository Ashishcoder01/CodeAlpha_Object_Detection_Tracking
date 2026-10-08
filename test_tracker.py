import cv2
from tracker import ObjectTracker


VIDEO_PATH = "input/test.mp4"
OUTPUT_PATH = "output/tracked.mp4"


tracker = ObjectTracker(
    model_path="yolo11n.pt",
    confidence=0.40
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


frame_count = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    results, tracked_objects = tracker.track_frame(frame)

    annotated_frame = results[0].plot()

    for obj in tracked_objects:

        x1, y1, x2, y2 = obj["bbox"]

        track_id = obj["track_id"]
        class_name = obj["class_name"]

        label = f"{class_name} ID:{track_id}"

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    writer.write(annotated_frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Processed {frame_count} frames | "
            f"Tracked objects: {len(tracked_objects)}"
        )


cap.release()
writer.release()


print("\nTracking completed.")
print(f"Processed frames: {frame_count}")
print(f"Output: {OUTPUT_PATH}")