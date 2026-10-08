from collections import Counter
import cv2


def format_count(count):
    return f"{count:,}"


def count_objects(detections):
    return Counter(
        detection["class_name"]
        for detection in detections
    )


def draw_statistics(frame, object_counts, fps=None):
    """
    Draw detection statistics on a video frame.
    """

    overlay = frame.copy()

    height, width = frame.shape[:2]

    cv2.rectangle(
        overlay,
        (15, 15),
        (330, 65),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        overlay,
        "OBJECT DETECTION",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y = 95

    for class_name, count in object_counts.items():

        text = f"{class_name}: {count}"

        cv2.putText(
            overlay,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 30

    if fps is not None:

        cv2.putText(
            overlay,
            f"FPS: {fps:.1f}",
            (20, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return overlay