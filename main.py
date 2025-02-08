import cv2
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark as PL
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


def remove_useless(landmarks):
    del_elems = [PL.NOSE,
                 PL.LEFT_EYE_INNER,
                 PL.LEFT_EYE,
                 PL.LEFT_EYE_OUTER,
                 PL.RIGHT_EYE_INNER,
                 PL.RIGHT_EYE,
                 PL.RIGHT_EYE_OUTER,
                 PL.LEFT_EAR,
                 PL.RIGHT_EAR,
                 PL.MOUTH_LEFT,
                 PL.MOUTH_RIGHT,
                 PL.LEFT_KNEE,
                 PL.RIGHT_KNEE,
                 PL.LEFT_ANKLE,
                 PL.RIGHT_ANKLE,
                 PL.LEFT_HEEL,
                 PL.RIGHT_HEEL,
                 PL.LEFT_FOOT_INDEX,
                 PL.RIGHT_FOOT_INDEX,
                 #
                 PL.LEFT_HIP,
                 PL.RIGHT_HIP
                 ]
    return [x for i, x in enumerate(landmarks) if i not in del_elems]


def calculate_centroid(landmarks):
    x = [p.x for p in landmarks]
    y = [p.y for p in landmarks]
    return sum(x) / len(landmarks), sum(y) / len(landmarks)


def draw_point(image, landmark):
    # De-normalize
    h, w, _ = image.shape
    x = int(landmark[0] * w)
    y = int(landmark[1] * h)
    cv2.circle(image, (x, y), 5, (255, 255, 0), -1)


while True:
    _, img = cam.capture_array()
    img = cv2.flip(img, 1)

    results = pose.process(img)

    if results.pose_landmarks:
        centroid = calculate_centroid(
            remove_useless(results.pose_landmarks.landmark))
        draw_point(img, centroid)
        mp.solutions.drawing_utils.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow('pose detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
