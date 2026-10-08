from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO


# ============================================================
# FIND VIDEO AUTOMATICALLY
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

video_candidates = []

for folder in [
    PROJECT_DIR / "input",
    PROJECT_DIR,
]:

    if folder.exists():

        video_candidates.extend(
            folder.glob("*.mp4")
        )

        video_candidates.extend(
            folder.glob("*.avi")
        )

        video_candidates.extend(
            folder.glob("*.mov")
        )

        video_candidates.extend(
            folder.glob("*.mkv")
        )


if not video_candidates:

    print("\nERROR: No video file found.")
    print()
    print("Put your test video inside:")
    print(PROJECT_DIR / "input")
    print()

    raise SystemExit


print()
print("Video found:")
print(video_candidates[0])
print()


VIDEO = str(video_candidates[0])


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO11 Nano...")

model = YOLO(
    str(PROJECT_DIR / "yolo11n.pt")
)

print("Model loaded successfully.")
print()


# ============================================================
# TEST DIFFERENT CONFIDENCE LEVELS
# ============================================================

thresholds = [
    0.25,
    0.35,
    0.45,
    0.55,
]


for confidence in thresholds:

    print()
    print("=" * 60)
    print(
        f"CONFIDENCE: {confidence}"
    )
    print("=" * 60)

    cap = cv2.VideoCapture(
        VIDEO
    )

    if not cap.isOpened():

        print(
            "ERROR: OpenCV could not open:"
        )

        print(
            VIDEO
        )

        continue

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print()

    detection_events = Counter()

    frame_presence = Counter()

    frame_number = 0


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:

            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=640,
            verbose=False,
            save=False,
        )

        classes_this_frame = set()


        for result in results:

            if result.boxes is None:

                continue


            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                detection_events[
                    class_name
                ] += 1

                classes_this_frame.add(
                    class_name
                )


        for class_name in classes_this_frame:

            frame_presence[
                class_name
            ] += 1


        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number:,} / "
                f"{total_frames:,}"
            )


    cap.release()


    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print(
        "Detected classes:"
    )

    if not detection_events:

        print(
            "No objects detected."
        )

        continue


    print()

    print(
        f"{'Class':<20}"
        f"{'Events':>10}"
        f"{'Frames':>10}"
    )

    print(
        "-" * 40
    )


    for class_name, count in (
        detection_events.most_common()
    ):

        print(
            f"{class_name:<20}"
            f"{count:>10}"
            f"{frame_presence[class_name]:>10}"
        )


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)