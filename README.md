# Driver Drowsiness & Distraction Detection System

This project is a real-time safety application designed to monitor a driver's state using computer vision. It utilizes **YOLOv8** for external distraction detection and **MediaPipe’s Tasks API** for internal state monitoring, such as drowsiness and yawning.

---

## Features

*   **Drowsiness Detection**: Monitors the Eye Aspect Ratio (EAR) and triggers a loud alarm if eyes remain closed for a sustained period (approx. 1 second).
*   **Yawn Detection**: Monitors the Mouth Aspect Ratio (MAR) to detect signs of fatigue.
*   **Distraction Detection**: Uses YOLOv8 to identify if the driver is using a **Cell Phone** or smoking.
*   **Real-time Alerts**: Provides both visual on-screen warnings and audible beep alerts using the `winsound` library.
*   **Modern Architecture**: Built using the MediaPipe Tasks API for robust facial landmarking in modern Python environments.

---

## Tech Stack

*   **Language**: Python 3.11.x (Stable)
*   **Deep Learning**: YOLOv8 (Ultralytics)
*   **Facial Mesh**: MediaPipe Tasks API
*   **Computer Vision**: OpenCV
*   **Numerical Processing**: NumPy

---

## Prerequisites

1.  **Python 3.11**: Optimized for stability with AI libraries.
2.  **Webcam**: Required for real-time video feed.
3.  **Task Model File**: Download `face_landmarker.task` from [Google MediaPipe](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) and place it in the root directory.

---

## Installation
---

## Cloning and Setup

To get a local copy of this project up and running, follow these steps:

### 1. Clone the Repository
    Open your terminal or command prompt and run:
    
    git clone https://github.com/NKumar-B/DriverDrowsinessDetection.git
    
    cd DriverDrowsinessDetection
    

### 2.  **Create and activate a virtual environment**:
    
    py -3.11 -m venv .venv
    
    .\.venv\Scripts\activate
    
### 3.  **Install dependencies**:
    
    pip install opencv-python mediapipe ultralytics numpy
                     (or)
    pip install -r requirements.txt
       
    

---

## How to Run

1.  Place `face_landmarker.task` in the project folder.
2.  Execute the script:
    
    python DriverDrowsiness.py
    
3.  **Controls**: Press **'q'** to exit the application.

---

## Logic Overview

*   **EAR (Eye Aspect Ratio)**: Calculates the distance between eyelid landmarks. An alert triggers if EAR falls below **0.2** for **20 consecutive frames**.
*   **MAR (Mouth Aspect Ratio)**: Monitors mouth opening. A "Yawn Detected" message appears if the ratio exceeds **0.5**.
*   **Object Detection**: YOLOv8 scans for the `cell phone` class with a confidence threshold of **0.5**.

---

## Future Enhancements

*   **Infrared (IR) Camera Integration**: To enable reliable monitoring during nighttime driving.
*   **Head Pose Estimation**: Tracking X, Y, and Z axes of the head to detect if the driver is looking away from the road for too long.
*   **Multi-threading**: Moving audio alerts to a separate thread to prevent video frame stuttering during alarms.
*   **Baseline Calibration**: A startup routine to calibrate EAR/MAR thresholds based on an individual driver's unique facial features.
*   **Mobile App Integration**: Sending alerts to a connected mobile device or a fleet management dashboard.

---

## Author


**Nithin**
**Linkedin**:[Nithin Kumar](https://www.linkedin.com/in/nithin-kumar-badduluri-3942512a6)
*   *Project created for Python AI/ML Academic Studies.*
