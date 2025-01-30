import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=.5,
    min_tracking_confidence=.5
)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    results = pose.process(frame)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow('pose detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
