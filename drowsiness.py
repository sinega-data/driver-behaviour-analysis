import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_EAR, is_drowsy, EAR_CONSEC_FRAMES

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

counter = 0
status  = "ALERT"

def process_drowsiness(frame):
    global counter, status

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame, "NO FACE"

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    def get_coords(indices):
        return np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in indices
        ])

    left_ear  = calculate_EAR(get_coords(LEFT_EYE))
    right_ear = calculate_EAR(get_coords(RIGHT_EYE))
    avg_ear   = (left_ear + right_ear) / 2.0

    if is_drowsy(avg_ear):
        counter += 1
        if counter >= EAR_CONSEC_FRAMES:
            status = "DROWSY"
    else:
        counter = 0
        status  = "ALERT"

    # Display EAR value
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, status