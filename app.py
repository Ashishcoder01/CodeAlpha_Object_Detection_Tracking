
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import av
import threading
import time
import base64
import requests
import textwrap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "yolo11n.pt"

# Balanced video inference size.
# 640 gives better detection quality than 480 while remaining
# reasonably fast with YOLO11 Nano.
VIDEO_INFERENCE_SIZE = 640

# Temporal confirmation for video detection.
# A detection must appear in consecutive processed frames
# before it is drawn and counted as a confirmed object.
MIN_CONFIRMATION_FRAMES = 2
TEMPORAL_IOU_THRESHOLD = 0.30

# Video-specific classes confirmed as relevant for the supplied warehouse
# video. Image detection remains unrestricted.
VIDEO_ALLOWED_CLASSES = {
    "person",
    "dog",
}



# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "video_mode" not in st.session_state:
    st.session_state.video_mode = "detection"

if "camera_last_result" not in st.session_state:
    st.session_state.camera_last_result = None


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #080b12;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    p {
        color: #a7b1c2;
    }

    section[data-testid="stSidebar"] {
        background: #0d111a;
        border-right: 1px solid #202938;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 10px;
        border: 1px solid #2c3a50;
        background: #151d2a;
        color: #ffffff;
        font-weight: 650;
    }

    .stButton > button:hover {
        border-color: #4f8cff;
        background: #1a2739;
        color: #ffffff;
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 10px;
        background: #2563eb;
        border: 1px solid #3b82f6;
        color: #ffffff;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        background: #0d131d;
        border: 1px dashed #35445b;
        border-radius: 16px;
        padding: 12px;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: #4f8cff;
    }

    div[data-testid="stMetric"] {
        background: #101620;
        border: 1px solid #222d3d;
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8995a7 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    button[data-baseweb="tab"] {
        color: #9aa6b8;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff;
    }

    div[role="radiogroup"] {
        gap: 8px;
    }

    .footer {
        text-align: center;
        color: #5f6b7d;
        font-size: 12px;
        padding-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEMPORAL DETECTION HELPERS
# ============================================================

def calculate_iou(box_a, box_b):
    """Calculate IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def update_temporal_detections(
    current_detections,
    previous_detections,
):
    """
    Match current detections to the previous processed frame.
    Only detections that persist across consecutive frames are
    returned as confirmed.
    """
    confirmed = []
    updated_detections = []
    used_previous = set()

    for detection in current_detections:
        class_name = detection["class_name"]
        current_box = detection["bbox"]

        best_iou = 0.0
        best_index = None

        for index, previous in enumerate(previous_detections):
            if index in used_previous:
                continue

            if previous["class_name"] != class_name:
                continue

            iou = calculate_iou(
                current_box,
                previous["bbox"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= TEMPORAL_IOU_THRESHOLD
        ):
            used_previous.add(best_index)
            streak = previous_detections[best_index]["streak"] + 1
        else:
            streak = 1

        updated = {
            **detection,
            "streak": streak,
        }

        updated_detections.append(updated)

        if streak >= MIN_CONFIRMATION_FRAMES:
            confirmed.append(updated)

    return confirmed, updated_detections


# ============================================================
# MODEL CHECK
# ============================================================

if not MODEL_PATH.exists():
    st.error("YOLO model file was not found.")
    st.code(str(MODEL_PATH))
    st.info("Run this command from the project folder:")
    st.code(
        'python -c "from ultralytics import YOLO; '
        'YOLO(\'yolo11n.pt\')"'
    )
    st.stop()


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_yolo_model():
    return YOLO(str(MODEL_PATH))


# ============================================================
# FFMPEG HELPERS
# ============================================================

def find_ffmpeg():
    """
    Find ffmpeg on Windows/Linux/macOS.
    """
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    common_windows_paths = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]

    for candidate in common_windows_paths:
        if os.path.exists(candidate):
            return candidate

    return None


def convert_to_browser_mp4(input_path, output_path):
    """
    Convert OpenCV's MP4V output to H.264/AAC MP4.

    Most browsers handle H.264 MP4 much more reliably than
    OpenCV's mp4v codec. The -movflags faststart option also
    moves the MP4 metadata to the beginning so browser playback
    can start without requiring a complete file download first.
    """
    ffmpeg = find_ffmpeg()

    if not ffmpeg:
        return False, "FFmpeg was not found."

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            return False, result.stderr[-3000:]

        if not os.path.exists(output_path):
            return False, "FFmpeg did not create the output file."

        if os.path.getsize(output_path) < 1000:
            return False, "The converted video file is empty."

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "FFmpeg conversion timed out."

    except Exception as error:
        return False, str(error)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("VisionTrack AI")

    st.caption("Object Detection & Tracking Platform")

    st.divider()

    st.subheader("Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help=(
            "Lower values detect more objects. "
            "Higher values provide stricter detection."
        ),
    )

    st.divider()

    st.subheader("Model")

    st.write("YOLO11 Nano")
    st.write("Ultralytics")
    st.write("OpenCV")

    st.divider()

    st.subheader("System Status")

    st.success("Detection engine ready")

    st.divider()

    if st.button("Home", key="sidebar_home"):
        st.session_state.page = "home"
        st.rerun()

    st.divider()

    st.caption("VisionTrack AI")
    st.caption("Computer Vision Project")


# ============================================================
# LIVE CAMERA - VISION LANGUAGE MODEL
# ============================================================

# ============================================================
# LIVE CAMERA DETECTION
# ============================================================
# The local Qwen3-VL test took ~69 seconds on this machine.
# That is not suitable for a live webcam loop. For Task 4, the
# camera therefore uses YOLO11n directly for real-time detection
# and tracking. Ollama remains optional and is not placed inside
# the live frame-processing path.

CAMERA_CONFIDENCE = 0.30
CAMERA_INFERENCE_SIZE = 640
CAMERA_PROCESS_EVERY_N_FRAMES = 2
CAMERA_TRACK_MAX_AGE = 12
CAMERA_BOX_SMOOTHING = 0.65


class LiveCameraProcessor(VideoProcessorBase):
    """Real-time YOLO11 detection and tracking for the webcam."""

    def __init__(self):
        self.frame_index = 0
        self.model = None
        self.model_error = None
        # Keep the latest tracked boxes between inference frames so the
        # overlay never disappears for a frame and visually blinks.
        self.tracked_cache = {}

        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            self.model = YOLO(str(MODEL_PATH))
        except Exception as exc:
            self.model_error = f"{type(exc).__name__}: {str(exc)[:220]}"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1

        # Run the detector periodically to keep the webcam responsive.
        # IMPORTANT: drawing the cached tracks happens on EVERY frame.
        # This prevents the boxes from blinking between inference frames.
        if self.model is not None and (
            self.frame_index % CAMERA_PROCESS_EVERY_N_FRAMES == 0
        ):
            try:
                results = self.model.track(
                    source=image,
                    persist=True,
                    conf=CAMERA_CONFIDENCE,
                    imgsz=CAMERA_INFERENCE_SIZE,
                    max_det=100,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )

                if results:
                    result = results[0]
                    boxes = result.boxes

                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]

                            x1, y1, x2, y2 = map(
                                int, box.xyxy[0].tolist()
                            )

                            track_id = (
                                int(box.id[0])
                                if box.id is not None
                                else None
                            )

                            # Prefer ByteTrack ID for stable identity.
                            # If an ID is temporarily unavailable, use a
                            # spatial/class key so the object does not flash.
                            if track_id is not None:
                                key = f"id:{track_id}"
                            else:
                                key = (
                                    f"obj:{class_id}:"
                                    f"{x1 // 100}:{y1 // 100}"
                                )

                            previous = self.tracked_cache.get(key)

                            if previous is not None:
                                px1, py1, px2, py2 = previous["bbox"]
                                a = CAMERA_BOX_SMOOTHING
                                x1 = int(a * px1 + (1 - a) * x1)
                                y1 = int(a * py1 + (1 - a) * y1)
                                x2 = int(a * px2 + (1 - a) * x2)
                                y2 = int(a * py2 + (1 - a) * y2)

                            self.tracked_cache[key] = {
                                "bbox": (x1, y1, x2, y2),
                                "class_name": class_name,
                                "confidence": confidence,
                                "track_id": track_id,
                                "last_seen": self.frame_index,
                            }

            except Exception as exc:
                self.model_error = (
                    f"{type(exc).__name__}: {str(exc)[:220]}"
                )

        # --------------------------------------------------------
        # DRAW THE LAST KNOWN TRACKS ON EVERY CAMERA FRAME.
        # This is the critical fix for blinking.
        # --------------------------------------------------------
        expired = []

        for key, item in list(self.tracked_cache.items()):
            age = self.frame_index - item["last_seen"]

            if age > CAMERA_TRACK_MAX_AGE:
                expired.append(key)
                continue

            x1, y1, x2, y2 = item["bbox"]
            confidence = item["confidence"]
            class_name = item["class_name"]
            track_id = item["track_id"]

            # Slightly fade old tracks instead of immediately removing them.
            # Fresh detections remain fully visible.
            thickness = 3 if age <= 2 else 2

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness,
            )

            if track_id is not None:
                label = f"{class_name} ID:{track_id} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"

            label_y = max(y1 - 10, 25)

            (label_w, label_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                2,
            )

            cv2.rectangle(
                image,
                (x1, label_y - label_h - 8),
                (x1 + label_w + 8, label_y + 2),
                (10, 15, 25),
                -1,
            )

            cv2.putText(
                image,
                label,
                (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for key in expired:
            self.tracked_cache.pop(key, None)

        # Camera status overlay.
        overlay = image.copy()

        panel_width = min(
            760,
            max(520, image.shape[1] - 40),
        )
        panel_height = 92

        cv2.rectangle(
            overlay,
            (20, 20),
            (20 + panel_width, 20 + panel_height),
            (10, 15, 25),
            -1,
        )

        image = cv2.addWeighted(
            overlay,
            0.88,
            image,
            0.12,
            0,
        )

        cv2.putText(
            image,
            "VISIONTRACK AI  |  YOLO11n",
            (40, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.model_error:
            status = "ERROR"
            status_color = (80, 120, 255)
        else:
            status = "LIVE"
            status_color = (80, 220, 120)

        cv2.putText(
            image,
            status,
            (max(40, panel_width - 120), 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if self.model_error:
            error_text = self.model_error[:100]
            cv2.putText(
                image,
                error_text,
                (40, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (160, 175, 195),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                "Real-time detection + ByteTrack tracking",
                (40, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (160, 175, 195),
                1,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.title("VisionTrack AI")

    st.subheader("Intelligent Object Detection & Tracking")

    st.write(
        "Analyze images and videos using YOLO-powered "
        "computer vision with accurate object detection "
        "and multi-object tracking."
    )

    st.divider()

    st.header("Choose Analysis Type")

    st.write("Select what you want VisionTrack AI to analyze.")

    st.write("")

    image_column, video_column, camera_column = st.columns(
        3,
        gap="large",
    )

    # ========================================================
    # IMAGE OPTION
    # ========================================================

    with image_column:

        st.subheader("Image Analysis")

        st.write(
            "Detect objects in photos with bounding "
            "boxes, class labels, and confidence scores."
        )

        st.write("")
        st.write("Object Detection")
        st.write("Bounding Boxes")
        st.write("Confidence Scores")
        st.write("")

        if st.button(
            "Start Image Analysis",
            key="start_image",
            use_container_width=True,
        ):
            st.session_state.page = "image"
            st.rerun()

    # ========================================================
    # VIDEO OPTION
    # ========================================================

    with video_column:

        st.subheader("Video Analysis")

        st.write(
            "Analyze uploaded videos frame-by-frame "
            "or track objects using persistent IDs."
        )

        st.write("")
        st.write("Video Object Detection")
        st.write("Multi-Object Tracking")
        st.write("Persistent Tracking IDs")
        st.write("")

        if st.button(
            "Start Video Analysis",
            key="start_video",
            use_container_width=True,
        ):
            st.session_state.page = "video"
            st.rerun()

    # ========================================================
    # CAMERA OPTION
    # ========================================================

    with camera_column:

        st.subheader("Live Camera")

        st.write(
            "Use your webcam for real-time object detection "
            "and persistent multi-object tracking."
        )

        st.write("")
        st.write("Live Camera Input")
        st.write("Real-Time Object Detection")
        st.write("Persistent Object Tracking")
        st.write("")

        if st.button(
            "Start Camera Detection",
            key="start_camera",
            use_container_width=True,
        ):
            st.session_state.page = "camera"
            st.rerun()

    st.divider()

    st.header("Platform Capabilities")

    capability1, capability2, capability3, capability4 = st.columns(4)

    with capability1:
        st.metric("Model", "YOLO11")

    with capability2:
        st.metric("Vision Engine", "OpenCV")

    with capability3:
        st.metric("Image Mode", "Detection")

    with capability4:
        st.metric("Video Mode", "Detect + Track")


# ============================================================
# LIVE CAMERA DETECTION
# ============================================================

elif st.session_state.page == "camera":

    if st.button(
        "← Back to Home",
        key="back_from_camera",
    ):
        st.session_state.page = "home"
        st.session_state.camera_last_result = None
        st.rerun()

    st.title("Live Camera Detection")

    st.write(
        "Use your webcam as a live computer-vision feed. "
        "VisionTrack AI continuously analyzes the camera stream "
        "and labels supported objects without taking photos."
    )

    st.divider()

    camera_left, camera_right = st.columns(
        [1.35, 0.65],
        gap="large",
    )

    with camera_left:

        st.subheader("Live Camera")

        camera_result = webrtc_streamer(
            key="visiontrack-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveCameraProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280, "min": 640},
                    "height": {"ideal": 720, "min": 480},
                    "frameRate": {"ideal": 30, "max": 30},
                    "facingMode": "user",
                },
                "audio": False,
            },
            async_processing=True,
        )

        if camera_result.state.playing:
            st.success("Camera is live — point it at an object")
        else:
            st.info(
                "Click START above and allow browser camera permission "
                "to begin real-time detection."
            )

    with camera_right:

        st.subheader("Live AI Detection")

        st.metric("Engine", "YOLO11n")
        st.metric("Mode", "Live Detection + Tracking")

        if camera_result.state.playing:
            st.success("Detection: ACTIVE")
        else:
            st.warning("Detection: STANDBY")

        st.write("**What VisionTrack AI does**")
        st.write("• Detects supported objects in the live camera")
        st.write("• Draws bounding boxes and confidence scores")
        st.write("• Assigns persistent tracking IDs")
        st.write("• Tracks multiple objects simultaneously")
        st.write("• Runs locally with YOLO11n")

    st.divider()

    st.subheader("Detection Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric("Input", "Webcam Stream")

    with info2:
        st.metric("Inference", "Continuous")

    with info3:
        st.metric("Model", "Qwen3-VL 4B")

    st.caption(
        "The camera remains live continuously. Qwen3-VL analyzes a lightweight frame "
        "periodically so the HD video stays responsive while AI vision runs locally."
    )

# ============================================================
# IMAGE ANALYSIS
# ============================================================
elif st.session_state.page == "image":

    if st.button(
        "← Back to Home",
        key="back_from_image",
    ):
        st.session_state.page = "home"
        st.rerun()

    st.title("Image Analysis")

    st.write(
        "Upload an image and VisionTrack AI will "
        "detect objects automatically."
    )

    st.divider()

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="image_upload",
    )

    if uploaded_image is None:

        st.info(
            "Upload an image to start detection."
        )

    else:

        try:

            # ------------------------------------------------
            # LOAD IMAGE
            # ------------------------------------------------

            image = Image.open(
                uploaded_image
            ).convert("RGB")

            image_array = np.array(
                image
            )

            image_bgr = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR,
            )

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            model = load_yolo_model()

            # ------------------------------------------------
            # DETECTION
            # ------------------------------------------------

            with st.spinner(
                "Running YOLO detection..."
            ):

                results = model.predict(
                    source=image_bgr,
                    conf=confidence,
                    imgsz=640,
                    verbose=False,
                    save=False,
                )

            # ------------------------------------------------
            # PROCESS RESULTS
            # ------------------------------------------------

            annotated_image = image_bgr.copy()

            detections = []

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    confidence_value = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist(),
                    )

                    class_name = model.names[
                        class_id
                    ]

                    detections.append(
                        {
                            "class_name": class_name,
                            "confidence": confidence_value,
                            "bbox": (
                                x1,
                                y1,
                                x2,
                                y2,
                            ),
                        }
                    )

                    cv2.rectangle(
                        annotated_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3,
                    )

                    label = (
                        f"{class_name} "
                        f"{confidence_value:.2f}"
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

            annotated_rgb = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB,
            )

            st.divider()

            if detections:

                st.success(
                    f"Detected {len(detections)} object(s)"
                )

            else:

                st.warning(
                    "No objects detected. "
                    "Try lowering the confidence threshold."
                )

            original_column, result_column = st.columns(
                2,
                gap="large",
            )

            with original_column:

                st.subheader(
                    "Original Image"
                )

                st.image(
                    image,
                    use_container_width=True,
                )

            with result_column:

                st.subheader(
                    "Detection Result"
                )

                st.image(
                    annotated_rgb,
                    use_container_width=True,
                )

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            counts = Counter(
                item["class_name"]
                for item in detections
            )

            average_confidence = (
                sum(
                    item["confidence"]
                    for item in detections
                )
                / len(detections)
                if detections
                else 0
            )

            st.divider()

            st.header(
                "Detection Statistics"
            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Objects Detected",
                    len(detections),
                )

            with metric2:

                st.metric(
                    "Object Classes",
                    len(counts),
                )

            with metric3:

                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.1%}",
                )

            if counts:

                st.subheader(
                    "Detected Classes"
                )

                for class_name, count in counts.items():

                    st.write(
                        f"**{class_name.title()}** "
                        f"— {count}"
                    )

        except Exception as error:

            st.error(
                "Image detection failed."
            )

            st.exception(
                error
            )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

elif st.session_state.page == "video":

    if st.button(
        "← Back to Home",
        key="back_from_video",
    ):
        st.session_state.page = "home"
        st.rerun()

    st.title("Video Analysis")

    st.write(
        "Choose between frame-by-frame object detection "
        "and persistent multi-object tracking."
    )

    st.divider()

    # ========================================================
    # VIDEO MODE
    # ========================================================

    st.subheader(
        "Processing Mode"
    )

    mode_detection, mode_tracking = st.columns(
        2,
        gap="large",
    )

    with mode_detection:

        if st.button(
            "Video Detection",
            key="video_detection",
            use_container_width=True,
        ):

            st.session_state.video_mode = (
                "detection"
            )

            st.rerun()

    with mode_tracking:

        if st.button(
            "Video Tracking",
            key="video_tracking",
            use_container_width=True,
        ):

            st.session_state.video_mode = (
                "tracking"
            )

            st.rerun()

    st.write("")

    if st.session_state.video_mode == "detection":

        st.info(
            "Video Detection mode analyzes every frame "
            "and draws bounding boxes around detected objects."
        )

    else:

        st.info(
            "Video Tracking mode detects objects and "
            "maintains persistent IDs across frames."
        )

    st.divider()

    # ========================================================
    # VIDEO UPLOAD
    # ========================================================

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
        key="video_upload",
    )

    if uploaded_video is None:

        st.info(
            "Upload a video to begin processing."
        )

    else:

        input_path = None
        raw_output_path = None
        browser_output_path = None

        cap = None
        writer = None

        try:

            # ------------------------------------------------
            # SAVE UPLOADED VIDEO
            # ------------------------------------------------

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            input_file.write(
                uploaded_video.getbuffer()
            )

            input_file.close()

            input_path = input_file.name

            # ------------------------------------------------
            # OPEN INPUT VIDEO
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                input_path
            )

            if not cap.isOpened():

                st.error(
                    "Unable to open uploaded video."
                )

                st.stop()

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0 or not np.isfinite(fps):

                fps = 30.0

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if (
                width <= 0
                or height <= 0
            ):

                cap.release()
                cap = None

                st.error(
                    "Invalid video dimensions."
                )

                st.stop()

            # ------------------------------------------------
            # RAW OUTPUT
            # ------------------------------------------------

            raw_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            raw_file.close()

            raw_output_path = raw_file.name

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (
                    width,
                    height,
                ),
            )

            if not writer.isOpened():

                cap.release()
                cap = None

                st.error(
                    "Unable to create output video."
                )

                st.stop()

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            model = load_yolo_model()

            # ------------------------------------------------
            # PROCESSING UI
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "Processing Video"
            )

            progress = st.progress(
                0
            )

            status = st.empty()

            frame_number = 0
            total_detections = 0

            unique_track_ids = set()

            detected_classes = Counter()

            # Number of frames in which each class appeared.
            class_frame_presence = Counter()

            # Number of detections in each class across frames.
            class_detection_events = Counter()

            # ------------------------------------------------
            # DETECTION MODE
            # ------------------------------------------------

            if st.session_state.video_mode == "detection":

                previous_detections = []

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_number += 1

                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=VIDEO_INFERENCE_SIZE,
                        verbose=False,
                        save=False,
                    )

                    current_detections = []

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            confidence_value = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            class_name = model.names[
                                class_id
                            ]

                            # Ignore COCO classes that are not relevant
                            # to the supplied warehouse video.
                            if class_name not in VIDEO_ALLOWED_CLASSES:
                                continue

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0].tolist(),
                            )

                            current_detections.append(
                                {
                                    "class_name": class_name,
                                    "class_id": class_id,
                                    "confidence": confidence_value,
                                    "bbox": (
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                    ),
                                    "streak": 1,
                                }
                            )

                    (
                        confirmed_detections,
                        previous_detections,
                    ) = update_temporal_detections(
                        current_detections,
                        previous_detections,
                    )

                    annotated_frame = frame.copy()

                    classes_in_current_frame = set()

                    for detection in confirmed_detections:

                        class_name = detection[
                            "class_name"
                        ]

                        confidence_value = detection[
                            "confidence"
                        ]

                        x1, y1, x2, y2 = detection[
                            "bbox"
                        ]

                        total_detections += 1

                        class_detection_events[
                            class_name
                        ] += 1

                        classes_in_current_frame.add(
                            class_name
                        )

                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2,
                        )

                        label = (
                            f"{class_name} "
                            f"{confidence_value:.2f}"
                        )

                        cv2.putText(
                            annotated_frame,
                            label,
                            (
                                x1,
                                max(
                                    y1 - 10,
                                    20,
                                ),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    for class_name in classes_in_current_frame:

                        class_frame_presence[
                            class_name
                        ] += 1

                    writer.write(
                        annotated_frame
                    )

                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_number
                                / total_frames,
                                1.0,
                            )
                        )

                    status.write(
                        f"Detecting frame "
                        f"{frame_number:,} / "
                        f"{total_frames:,}"
                    )

            # ------------------------------------------------
            # TRACKING MODE
            # ------------------------------------------------

            else:

                # Restrict tracking to the classes that are actually
                # relevant to the supplied warehouse video.
                # This prevents ByteTrack from tracking YOLO's
                # false classifications such as bed, suitcase,
                # horse, cake, cow, etc.
                allowed_class_ids = [
                    class_id
                    for class_id, class_name in model.names.items()
                    if class_name in VIDEO_ALLOWED_CLASSES
                ]

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_number += 1

                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        imgsz=VIDEO_INFERENCE_SIZE,
                        classes=allowed_class_ids,
                        verbose=False,
                        save=False,
                    )

                    annotated_frame = frame.copy()

                    classes_in_current_frame = set()

                    for result in results:

                        if result.boxes is None:
                            continue

                        boxes = result.boxes

                        for index in range(
                            len(boxes)
                        ):

                            box = boxes[index]

                            confidence_value = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            class_name = model.names[
                                class_id
                            ]

                            # Safety check in case the model returns
                            # anything outside the requested classes.
                            if class_name not in VIDEO_ALLOWED_CLASSES:
                                continue

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0].tolist(),
                            )

                            classes_in_current_frame.add(
                                class_name
                            )

                            class_detection_events[
                                class_name
                            ] += 1

                            track_id = None

                            if boxes.id is not None:

                                track_id = int(
                                    boxes.id[index]
                                )

                                unique_track_ids.add(
                                    track_id
                                )

                            cv2.rectangle(
                                annotated_frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2,
                            )

                            if track_id is not None:

                                label = (
                                    f"{class_name} "
                                    f"ID:{track_id}"
                                )

                            else:

                                label = (
                                    f"{class_name} "
                                    f"{confidence_value:.2f}"
                                )

                            cv2.putText(
                                annotated_frame,
                                label,
                                (
                                    x1,
                                    max(
                                        y1 - 10,
                                        20,
                                    ),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                    for class_name in classes_in_current_frame:

                        class_frame_presence[
                            class_name
                        ] += 1

                    writer.write(
                        annotated_frame
                    )

                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_number
                                / total_frames,
                                1.0,
                            )
                        )

                    status.write(
                        f"Tracking frame "
                        f"{frame_number:,} / "
                        f"{total_frames:,}"
                    )

            # ------------------------------------------------
            # RELEASE OPENCV
            # ------------------------------------------------

            cap.release()
            cap = None

            writer.release()
            writer = None

            progress.progress(
                1.0
            )

            status.success(
                "Video processing completed successfully."
            )

            # ------------------------------------------------
            # BROWSER-COMPATIBLE VIDEO
            # ------------------------------------------------

            browser_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            browser_file.close()

            browser_output_path = browser_file.name

            with st.spinner(
                "Preparing video for browser playback..."
            ):

                converted, conversion_error = (
                    convert_to_browser_mp4(
                        raw_output_path,
                        browser_output_path,
                    )
                )

            # If FFmpeg is unavailable or conversion fails,
            # keep the OpenCV output as a fallback.
            if not converted:

                if os.path.exists(
                    browser_output_path
                ):

                    try:
                        os.remove(
                            browser_output_path
                        )
                    except OSError:
                        pass

                browser_output_path = raw_output_path

                st.warning(
                    "Browser-compatible H.264 conversion was "
                    "not available. The processed MP4 is still "
                    "available for download."
                )

            else:

                # The H.264 version is the one used for browser
                # playback and download.
                if (
                    raw_output_path
                    and os.path.exists(raw_output_path)
                ):

                    try:
                        os.remove(
                            raw_output_path
                        )
                    except OSError:
                        pass

                raw_output_path = None

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            st.divider()

            st.header(
                "Processing Statistics"
            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Frames Processed",
                    f"{frame_number:,}",
                )

            with metric2:

                if (
                    st.session_state.video_mode
                    == "tracking"
                ):

                    st.metric(
                        "Unique Track IDs",
                        len(
                            unique_track_ids
                        ),
                    )

                else:

                    st.metric(
                        "Detection Events",
                        f"{total_detections:,}",
                    )

            with metric3:

                st.metric(
                    "Object Classes",
                    len(
                        class_frame_presence
                    ),
                )

            # ------------------------------------------------
            # PROCESSED VIDEO
            # ------------------------------------------------

            st.divider()

            st.header(
                "Processed Video"
            )

            if (
                browser_output_path
                and os.path.exists(
                    browser_output_path
                )
            ):

                with open(
                    browser_output_path,
                    "rb",
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )

                # Explicitly provide MP4 MIME type.
                # H.264 + yuv420p + faststart makes this
                # playable directly in modern browsers.
                st.video(
                    video_bytes,
                    format="video/mp4",
                )

                if (
                    st.session_state.video_mode
                    == "tracking"
                ):

                    output_name = (
                        "visiontrack_tracked.mp4"
                    )

                else:

                    output_name = (
                        "visiontrack_detected.mp4"
                    )

                st.download_button(
                    "Download Processed Video",
                    data=video_bytes,
                    file_name=output_name,
                    mime="video/mp4",
                    use_container_width=True,
                )

            # ------------------------------------------------
            # OBJECT SUMMARY
            # ------------------------------------------------

            if class_frame_presence:

                st.divider()

                if (
                    st.session_state.video_mode
                    == "tracking"
                ):

                    st.header(
                        "Tracked Objects"
                    )

                    st.caption(
                        "Unique persistent tracking IDs across the video."
                    )

                    st.write(
                        f"**Total Unique Tracked Objects:** "
                        f"{len(unique_track_ids):,}"
                    )

                else:

                    st.header(
                        "Detected Classes"
                    )

                    st.caption(
                        "Counts below show how many video frames "
                        "each class appeared in, not unique objects."
                    )

                sorted_objects = sorted(
                    class_frame_presence.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                for class_name, frame_count in sorted_objects:

                    st.write(
                        f"**{class_name.title()}** "
                        f"— present in {frame_count:,} frame(s)"
                    )

                if (
                    st.session_state.video_mode
                    == "detection"
                ):

                    st.caption(
                        f"Total frame-level detection events: "
                        f"{total_detections:,}"
                    )

        except Exception as error:

            if cap is not None:

                try:
                    cap.release()
                except Exception:
                    pass

            if writer is not None:

                try:
                    writer.release()
                except Exception:
                    pass

            st.error(
                "Video processing failed."
            )

            st.exception(
                error
            )

        finally:

            # Input can always be removed after processing.
            if (
                input_path
                and os.path.exists(
                    input_path
                )
            ):

                try:
                    os.remove(
                        input_path
                    )
                except OSError:
                    pass

            # Do not delete browser_output_path here.
            # Streamlit needs it during this script run.


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VisionTrack AI | YOLO11 Nano | OpenCV | "
    "Object Detection & Multi-Object Tracking"
)
