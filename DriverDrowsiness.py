import cv2
import mediapipe as mp
import numpy as np
import os
import winsound
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Suppress internal MediaPipe/TF logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration & Thresholds ---
EYE_CLOSED_THRESHOLD = 0.2
YAWN_THRESHOLD = 0.5  # Adjust based on testing
CONSECUTIVE_FRAMES = 20 # Number of frames eyes must be closed to trigger alarm
PHONE_CONF_THRESHOLD = 0.5

# --- Initialization ---
yolo_model = YOLO('yolov8n.pt')

# Ensure 'face_landmarker.task' is in your directory
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Track sustained eye closure
closed_frame_count = 0

def get_ratio(landmarks, idx1, idx2, idx3, idx4, idx5, idx6):
    """Calculates the aspect ratio for eyes or mouth."""
    # Vertical distances
    v1 = np.linalg.norm(np.array([landmarks[idx2].x, landmarks[idx2].y]) - np.array([landmarks[idx6].x, landmarks[idx6].y]))
    v2 = np.linalg.norm(np.array([landmarks[idx3].x, landmarks[idx3].y]) - np.array([landmarks[idx5].x, landmarks[idx5].y]))
    # Horizontal distance
    h = np.linalg.norm(np.array([landmarks[idx1].x, landmarks[idx1].y]) - np.array([landmarks[idx4].x, landmarks[idx4].y]))
    return (v1 + v2) / (2.0 * h)

cap = cv2.VideoCapture(0)

print("System Starting... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. YOLOv8 Detection (Distractions)
    yolo_results = yolo_model(frame, verbose=False, conf=PHONE_CONF_THRESHOLD)[0]
    for box in yolo_results.boxes:
        label = yolo_model.names[int(box.cls[0])]
        if label in ["cell phone"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "DISTRACTION: PHONE", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Short beep for immediate distraction
            winsound.Beep(1000, 100) 

    # 2. MediaPipe Face Landmarker (Drowsiness)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        
        # EAR (Eye Aspect Ratio) - Right Eye as reference
        ear = get_ratio(face_landmarks, 33, 160, 158, 133, 153, 144)
        
        # MAR (Mouth Aspect Ratio) - Using mouth indices
        mar = get_ratio(face_landmarks, 78, 81, 13, 308, 14, 178)

        # Logic for Eye Closure
        if ear < EYE_CLOSED_THRESHOLD:
            closed_frame_count += 1
        else:
            closed_frame_count = 0

        # Trigger Drowsiness Alarm
        if closed_frame_count >= CONSECUTIVE_FRAMES:
            cv2.putText(frame, "!!! WAKE UP !!!", (120, 250), 
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
            winsound.Beep(2500, 500) # High pitch long beep

        # Logic for Yawning
        if mar > YAWN_THRESHOLD:
            cv2.putText(frame, "YAWN DETECTED", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Driver Monitoring System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()