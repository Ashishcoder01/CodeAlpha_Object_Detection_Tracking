import cv2
from detector import ObjectDetector


detector = ObjectDetector(
    model_path="yolo11n.pt",
    confidence=0.40
)

image = cv2.imread("input/test.jpg")

if image is None:
    raise FileNotFoundError("Could not load input/test.jpg")

annotated_image, detections = detector.detect_image(image)

print("\nDetected Objects:")

for detection in detections:
    print(
        f"{detection['class_name']} "
        f"- {detection['confidence']:.2f}"
    )

counts = detector.get_class_counts(detections)

print("\nObject Counts:")
print(counts)

cv2.imwrite(
    "output/detected.jpg",
    annotated_image
)

print("\nSaved result to:")
print("output/detected.jpg")