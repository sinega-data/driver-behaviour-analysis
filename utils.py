import numpy as np

# Thresholds
EAR_THRESHOLD = 0.25      # below this = eye closed
EAR_CONSEC_FRAMES = 20    # frames eye closed = drowsy
HEAD_PITCH_THRESHOLD = 15  # degrees down = inattentive

def calculate_EAR(eye_landmarks):
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def is_drowsy(ear):
    return ear < EAR_THRESHOLD

def get_status_color(status):
    if status == "DROWSY":
        return (0, 0, 255)    # Red
    elif status == "DISTRACTED":
        return (0, 255, 255)  # Yellow
    else:
        return (0, 255, 0)    # Green