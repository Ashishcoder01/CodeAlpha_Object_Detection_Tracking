# VisionTrack AI — Object Detection & Tracking

VisionTrack AI is a computer-vision web application for detecting and tracking objects in images, uploaded videos, and live webcam streams. Built with Python, Streamlit, and Ultralytics YOLO11, it provides annotated visual results, confidence scores, tracking IDs, and video exports through a single interface.

<p align="center">
  <a href="https://visiontrack-ai1.streamlit.app/"><strong>Launch Live Demo</strong></a>
  &nbsp; • &nbsp;
  <a href="https://github.com/Ashishcoder01/CodeAlpha_Object_Detection_Tracking">View Source Code</a>
</p>

## Table of Contents

- [Live Demo](#live-demo)
- [Features](#features)
- [Detection Modes](#detection-modes)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Model Weights](#model-weights)
- [Tests](#tests)
- [Deployment Notes](#deployment-notes)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Context](#project-context)
- [Author](#author)

## Live Demo

**Try VisionTrack AI:** https://visiontrack-ai1.streamlit.app/

The deployed application supports image detection, video processing and tracking, and live webcam detection. Browser and camera permissions may be required for the corresponding features.

> **Note:** Live inference speed depends on available compute resources, input resolution, and video length.

## Features

- **Image object detection** with annotated bounding boxes and confidence scores.
- **Video processing** to produce an annotated output video.
- **Multi-object tracking** with persistent track IDs across frames.
- **Live webcam detection and tracking** using Streamlit-WebRTC.
- **YOLO11 inference** using Ultralytics models.
- **ByteTrack tracking pipeline** for associating detections across frames.
- **Detection statistics** including detected classes and processing information.
- **Processed video download** and browser-compatible video conversion when FFmpeg and the required encoder are available.
- **Streamlit interface** for running the available workflows from one application.

## Detection Modes

### 1. Image Detection

Upload an image to detect supported objects. The application draws bounding boxes, displays confidence scores, and provides detection results.

### 2. Video Detection

Upload a video to process its frames with YOLO-based detection. The application creates an annotated video that can be downloaded.

### 3. Object Tracking

Track detected objects across video frames. The tracking pipeline associates objects between frames and can display persistent tracking IDs.

### 4. Live Webcam

Use a supported camera through the browser. YOLO11 performs detection and tracking on incoming frames, with cached track overlays used to reduce visible box flicker between inference frames.

## How It Works

1. The user supplies an image, video, or live camera stream.
2. OpenCV or WebRTC provides frames to the inference pipeline.
3. Ultralytics YOLO11 detects supported object classes.
4. The tracking pipeline uses ByteTrack to associate detections over time where tracking is enabled.
5. VisionTrack AI renders annotations and statistics, and provides processed results for preview or download.

```text
Image / Uploaded Video / Webcam
               |
               v
        OpenCV / WebRTC
               |
               v
            YOLO11
               |
        +------+------+
        |             |
    Detection      Tracking
        |          ByteTrack
        +------+------+
               |
               v
      Annotated Results
       and Statistics
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and processing logic |
| Streamlit | Web application interface |
| Ultralytics YOLO11 | Object detection and inference |
| OpenCV | Image and video processing, annotations |
| NumPy | Array and frame operations |
| Pillow | Image support |
| Streamlit-WebRTC | Browser camera streaming |
| PyAV | Video frame handling for WebRTC |
| ByteTrack | Multi-object tracking |
| FFmpeg | Optional H.264 conversion for browser-compatible video playback |

## Project Structure

```text
CodeAlpha_Object_Detection_Tracking/
├── app.py
├── detector.py
├── tracker.py
├── utils.py
├── requirements.txt
├── packages.txt
├── test_detector.py
├── test_tracker.py
├── test_video_detection.py
├── input/
├── output/
├── assets/
├── models/
└── .gitignore
```

Generated output files and model weights may be excluded from Git by the repository's ignore rules.

## Run Locally

### Prerequisites

- Python 3.12 recommended
- Git
- A working camera and browser permission for live webcam mode
- FFmpeg with the `libx264` encoder for browser-compatible H.264 video conversion (optional; needed for that conversion workflow)

### 1. Clone the repository

```bash
git clone https://github.com/Ashishcoder01/CodeAlpha_Object_Detection_Tracking.git
cd CodeAlpha_Object_Detection_Tracking
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit.

## Model Weights

The application uses Ultralytics YOLO11 weights, such as `yolo11n.pt`. Model files are not committed to the repository to avoid storing large binary files in Git.

When `yolo11n.pt` is missing, the application asks Ultralytics to load the named model, which can download the weights if network access is available. The first model load may therefore take longer and requires the deployment environment to be able to access the model source.

## Tests

Run the available project tests from the repository root:

```powershell
python test_detector.py
python test_tracker.py
python test_video_detection.py
```

These scripts exercise the detector, tracker, and video-detection workflows. Test requirements and available input data may vary by local environment.

## Deployment Notes

The live application is hosted on Streamlit Community Cloud.

- `requirements.txt` lists Python dependencies.
- `packages.txt` lists system packages needed by the cloud environment, including FFmpeg for video conversion.
- The deployed app must be able to download YOLO model weights if they are not already present.
- Browser-compatible MP4 conversion requires FFmpeg with H.264 encoder support. If conversion is unavailable, the app may still provide the processed video for download.

## Limitations

- The pretrained model recognizes classes represented in its training data; unsupported or unfamiliar objects may not be detected or may be mislabeled.
- Detection quality and speed depend on model choice, frame resolution, available CPU/GPU resources, and input-video characteristics.
- Live webcam access depends on browser permissions and a supported device.
- Browser-compatible video playback depends on successful H.264 conversion and browser codec support.

## Future Improvements

- Train or fine-tune models on domain-specific datasets.
- Add object counting, line-crossing events, and configurable alerts.
- Expand tracking analytics and downloadable reports.
- Add additional model choices and performance controls.
- Extend export options and evaluation metrics.

## Project Context

VisionTrack AI was developed as part of the CodeAlpha Artificial Intelligence / Machine Learning internship project work and is maintained as a portfolio project.

## Author

**Ashish**  
Computer Science & Engineering — AI/ML

- **GitHub:** https://github.com/Ashishcoder01
- **Project repository:** https://github.com/Ashishcoder01/CodeAlpha_Object_Detection_Tracking
- **Live application:** https://visiontrack-ai1.streamlit.app/

## License

This repository is intended for educational and portfolio use. Check the licenses of third-party libraries and pretrained model weights before redistributing them or using them commercially.
