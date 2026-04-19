# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GLOG_minloglevel"] = "3"

from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

# Initialise mediapipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# load all 6 trained model files from the models folder
models_path = "../../models"

front_model         = joblib.load(os.path.join(models_path, "front_model.pkl"))
front_scaler        = joblib.load(os.path.join(models_path, "front_scaler.pkl"))
front_label_encoder = joblib.load(os.path.join(models_path, "front_label_encoder.pkl"))

side_model         = joblib.load(os.path.join(models_path, "side_model.pkl"))
side_scaler        = joblib.load(os.path.join(models_path, "side_scaler.pkl"))
side_label_encoder = joblib.load(os.path.join(models_path, "side_label_encoder.pkl"))

print("All models loaded successfully!")

# stores last 5 predictions to smooth out flickering
pred_history = deque(maxlen=5)

# initialise prediction variable
prediction = "Ready"

# calculates the angle between 3 joint points using arctan2
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

# detects if the user is facing front or side on
# compares shoulder width vs shoulder depth for more reliable detection
def detect_view(landmarks):
    left_shoulder_x  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    left_shoulder_z  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
    right_shoulder_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z

    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    shoulder_depth = abs(left_shoulder_z - right_shoulder_z)

    if shoulder_width > shoulder_depth:
        return "Front"
    else:
        return "Side"

# Start video capture from default webcam
live_cam = cv2.VideoCapture(0)

# Detects if webcam is available
if not live_cam.isOpened():
    print("Live cam not enabled")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while live_cam.isOpened():

        ret, frame = live_cam.read()
        if not ret:
            print("Live feed over")
            break

        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=6),
            )

            lm = results.pose_landmarks.landmark

            # detect which view the user is in
            view = detect_view(lm)

            # extract landmark coordinates
            ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,       lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,      lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            la = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,      lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            rk = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,     lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ra = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # calculate joint angles
            left_knee_angle   = calculate_angle(lh, lk, la)
            left_hip_angle    = calculate_angle(ls, lh, lk)
            left_trunk_angle  = calculate_angle(ls, lh, la)
            right_knee_angle  = calculate_angle(rh, rk, ra)
            right_hip_angle   = calculate_angle(rs, rh, rk)
            right_trunk_angle = calculate_angle(rs, rh, ra)

            # calculate distance features
            knee_distance          = abs(lk[0] - rk[0])
            ankle_distance         = abs(la[0] - ra[0])
            knee_ankle_ratio       = knee_distance / ankle_distance if ankle_distance != 0 else 0
            left_knee_foot_offset  = abs(lk[0] - la[0])
            right_knee_foot_offset = abs(rk[0] - ra[0])
            avg_offset             = (left_knee_foot_offset + right_knee_foot_offset) / 2

            # calculate average knee angle
            avg_knee = (left_knee_angle + right_knee_angle) / 2

            # build feature dataframe matching training data columns
            features = pd.DataFrame([[
                left_knee_angle, left_hip_angle, left_trunk_angle,
                right_knee_angle, right_hip_angle, right_trunk_angle,
                knee_distance, ankle_distance, knee_ankle_ratio,
                left_knee_foot_offset, right_knee_foot_offset
            ]], columns=[
                "left_knee_angle", "left_hip_angle", "left_trunk_angle",
                "right_knee_angle", "right_hip_angle", "right_trunk_angle",
                "knee_distance", "ankle_distance", "knee_ankle_ratio",
                "left_knee_foot_offset", "right_knee_foot_offset"
            ])

            if avg_knee > 170:
                prediction = "Ready"
                pred_history.clear()
            else:
                if view == "Front":
                    scaled_features = front_scaler.transform(features)
                    raw_prediction  = front_label_encoder.inverse_transform(
                        front_model.predict(scaled_features)
                    )[0]
                else:
                    scaled_features = side_scaler.transform(features)
                    raw_prediction  = side_label_encoder.inverse_transform(
                        side_model.predict(scaled_features)
                    )[0]

                # smooth prediction using last 5 frames to reduce flickering
                pred_history.append(raw_prediction)
                prediction = Counter(pred_history).most_common(1)[0][0]

            # display view and prediction on screen
            cv2.putText(image, f"View: {view}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Class: {prediction}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Live Webcam", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

live_cam.release()
cv2.destroyAllWindows()