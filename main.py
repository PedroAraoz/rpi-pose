import cv2
import mediapipe as mp
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration({'format': 'RGB888'})
cam.configure(config)
cam.start()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=.5,
    min_tracking_confidence=.5
)

while True:
    frame = cam.capture_array()
    frame = cv2.flip(frame, -1)

    results = pose.process(frame)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow('pose detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
