import cv2
import numpy as np
from drowsiness import process_drowsiness
from attention import process_attention
from utils import get_status_color
import winsound

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, drowsy_status = process_drowsiness(frame)
    frame, attention_status = process_attention(frame)

    if drowsy_status == "DROWSY":
        final_status = "DROWSY"
    elif attention_status == "DISTRACTED":
        final_status = "DISTRACTED"
    else:
        final_status = "ACTIVE"

    color = get_status_color(final_status)
    cv2.putText(frame, f"STATUS: {final_status}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    if final_status in ["DROWSY", "DISTRACTED"]:
        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1], frame.shape[0]),
                      color, 4)

    if final_status == "DROWSY":
        try:
            winsound.Beep(1000, 500)
        except:
            pass

    cv2.imshow("Driver Behaviour Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()