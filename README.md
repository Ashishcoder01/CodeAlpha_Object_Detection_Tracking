# VisionTrack AI — Object Detection & Tracking

VisionTrack AI is a computer-vision application built for real-time and offline object detection and tracking. It supports image analysis, video detection, multi-object tracking, and live webcam detection through a professional Streamlit interface.

## Features

- Image object detection
- Video object detection
- Multi-object tracking with persistent tracking IDs
- Real-time webcam detection
- Stable bounding boxes for live tracking
- Confidence-based detection
- YOLO11 models through Ultralytics
- ByteTrack-based object tracking
- Processed video preview and download
- Clean Streamlit interface

## Detection Modes

### Image Detection
Upload an image and VisionTrack AI identifies supported objects, draws bounding boxes, displays confidence scores, and provides detection statistics.

### Video Detection
Upload a video and process its frames with YOLO-based object detection. The application generates an annotated output video.

### Video Tracking
Track objects across video frames while maintaining persistent IDs using the tracking pipeline.

### Live Camera
Use a webcam as a continuous computer-vision stream without taking a photo. YOLO11 performs real-time detection and tracking directly on the incoming camera frames.

## Technology Stack

- Python
- Streamlit
- Ultralytics YOLO11
- OpenCV
- NumPy
- Pillow
- Streamlit-WebRTC
- PyAV
- Requests
- ByteTrack

## Project Workflow

```text
Image / Video / Webcam
          |
          v
      OpenCV / WebRTC
          |
          v
       YOLO11
          |
    +-----+-----+
    |           |
Detection    Tracking
    |           |
    |        ByteTrack
    |           |
    +-----+-----+
          |
          v
 Annotated Results
```

## Project Structure

```text
CodeAlpha_Object_Detection_Tracking/
├── app.py
├── detector.py
├── tracker.py
├── utils.py
├── requirements.txt
├── test_detector.py
├── test_tracker.py
├── test_video_detection.py
├── input/
├── output/
├── assets/
├── models/
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CodeAlpha_Object_Detection_Tracking
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the application

```powershell
python -m streamlit run app.py
```

The application will open in your browser.

## YOLO Models

The project uses Ultralytics YOLO11 models. Model weights are intentionally excluded from Git because binary model files increase repository size.

If the required model is not already available, Ultralytics can download it when the application loads it.

The project has been tested with:

- `yolo11n.pt` — lightweight model for fast inference
- `yolo11s.pt` — larger model for improved detection quality

## Testing

Run the available tests from the project directory:

```powershell
python test_detector.py
python test_tracker.py
python test_video_detection.py
```

## Limitations

The pretrained YOLO model recognizes the object categories included in its training dataset. Objects outside those categories may be classified incorrectly or may not be detected.

Live-camera performance also depends on the computer's CPU/GPU, camera resolution, and inference settings.

## Future Improvements

- Custom dataset training for domain-specific objects
- Improved model selection and GPU acceleration
- Advanced analytics and tracking statistics
- Object counting and line-crossing detection
- Custom alert/event rules
- Additional export formats
- Cloud deployment

## Academic / Internship Project

This project was developed as part of the **CodeAlpha Artificial Intelligence / Machine Learning Internship** project work.

## Author

**Ashish**

Computer Science & Engineering — AI/ML

## License

This project is intended for educational and portfolio purposes.
