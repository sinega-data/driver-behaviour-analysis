import cv2
import mediapipe as mp
import numpy as np
from utils import HEAD_PITCH_THRESHOLD

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Key landmark indices for head pose
NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE     = 33
RIGHT_EYE    = 263
LEFT_MOUTH   = 61
RIGHT_MOUTH  = 291

def process_attention(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame, "NO FACE"

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    # 2D image points
    image_points = np.array([
        [landmarks[NOSE_TIP].x * w,    landmarks[NOSE_TIP].y * h],
        [landmarks[CHIN].x * w,        landmarks[CHIN].y * h],
        [landmarks[LEFT_EYE].x * w,    landmarks[LEFT_EYE].y * h],
        [landmarks[RIGHT_EYE].x * w,   landmarks[RIGHT_EYE].y * h],
        [landmarks[LEFT_MOUTH].x * w,  landmarks[LEFT_MOUTH].y * h],
        [landmarks[RIGHT_MOUTH].x * w, landmarks[RIGHT_MOUTH].y * h],
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        [0.0,    0.0,    0.0],
        [0.0,   -63.6, -12.5],
        [-43.3,  32.7, -26.0],
        [43.3,   32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9,  -28.9, -24.1]
    ])

    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    _, rotation_vec, _ = cv2.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs
    )

    # Convert rotation to angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    pitch = angles[0]  # head up/down

    if pitch < -HEAD_PITCH_THRESHOLD:
        attention_status = "DISTRACTED"
    else:
        attention_status = "FOCUSED"

    # Display pitch
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, attention_status