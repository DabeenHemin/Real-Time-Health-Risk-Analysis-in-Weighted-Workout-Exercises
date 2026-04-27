# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GLOG_minloglevel"] = "3"

from collections import Counter, deque
import threading
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

# Initialise mediapipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# tracks state to prevent overlapping speech
last_spoken = ""
is_speaking = False
last_speak_time = 0

# tracks rep count and stage for squat rep tracking
counter = 0
current_stage = ""

# uses Mac native say command for reliable text to speech
# prevents repeat of same message within 5 seconds
def speak(message):
    global is_speaking, last_spoken, last_speak_time
    if is_speaking:
        return
    # dont repeat the same message within 5 seconds
    if message == last_spoken and time.time() - last_speak_time < 5:
        return
    is_speaking = True
    last_spoken = message
    last_speak_time = time.time()
    def run():
        global is_speaking
        try:
            subprocess.run(['say', message], check=False, timeout=5)
        except:
            pass
        is_speaking = False
    threading.Thread(target=run, daemon=True).start()

# load all 6 trained model files from the models folder
models_path = "../../models"

front_model         = joblib.load(os.path.join(models_path, "front_model.pkl"))
front_scaler        = joblib.load(os.path.join(models_path, "front_scaler.pkl"))
front_label_encoder = joblib.load(os.path.join(models_path, "front_label_encoder.pkl"))

side_model         = joblib.load(os.path.join(models_path, "side_model.pkl"))
side_scaler        = joblib.load(os.path.join(models_path, "side_scaler.pkl"))
side_label_encoder = joblib.load(os.path.join(models_path, "side_label_encoder.pkl"))

print("All models loaded successfully!")

pred_history = deque(maxlen=5)
prediction = "Ready"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

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

def get_skeleton_colour(prediction):
    if prediction == "good":
        return (0, 255, 0)
    elif prediction == "Ready":
        return (255, 255, 255)
    else:
        return (0, 0, 255)

live_cam = cv2.VideoCapture(0)

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

            lm = results.pose_landmarks.landmark
            view = detect_view(lm)

            ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,       lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,      lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            la = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,      lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            rk = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,     lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ra = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_knee_angle   = calculate_angle(lh, lk, la)
            left_hip_angle    = calculate_angle(ls, lh, lk)
            left_trunk_angle  = calculate_angle(ls, lh, la)
            right_knee_angle  = calculate_angle(rh, rk, ra)
            right_hip_angle   = calculate_angle(rs, rh, rk)
            right_trunk_angle = calculate_angle(rs, rh, ra)

            knee_distance          = abs(lk[0] - rk[0])
            ankle_distance         = abs(la[0] - ra[0])
            knee_ankle_ratio       = knee_distance / ankle_distance if ankle_distance != 0 else 0
            left_knee_foot_offset  = abs(lk[0] - la[0])
            right_knee_foot_offset = abs(rk[0] - ra[0])
            avg_offset             = (left_knee_foot_offset + right_knee_foot_offset) / 2

            avg_knee = (left_knee_angle + right_knee_angle) / 2

            # check visibility of key landmarks before counting reps
            # this prevents false reps when a person is walking into frame
            hip_vis = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].visibility +
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility) / 2
            knee_vis = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility +
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility) / 2
            ankle_vis = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility +
                         lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility) / 2

            # only track squat stage and count reps if all key landmarks are clearly visible
            if hip_vis > 0.7 and knee_vis > 0.7 and ankle_vis > 0.7:
                if avg_knee < 95:
                    current_stage = "down"

                if avg_knee > 165 and current_stage == "down":
                    current_stage = "up"
                    counter += 1

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
                    proba           = front_model.predict_proba(scaled_features)[0]
                    confidence      = max(proba)
                    raw_prediction  = front_label_encoder.inverse_transform(
                        front_model.predict(scaled_features)
                    )[0]

                    if raw_prediction == "knees_in":
                        if not (knee_ankle_ratio < 0.92 and avg_offset > 0.01):
                            raw_prediction = "good"

                    if knee_ankle_ratio < 0.7 and avg_offset > 0.05:
                        raw_prediction = "knees_in"

                else:
                    scaled_features = side_scaler.transform(features)
                    proba           = side_model.predict_proba(scaled_features)[0]
                    confidence      = max(proba)
                    raw_prediction  = side_label_encoder.inverse_transform(
                        side_model.predict(scaled_features)
                    )[0]

                    if raw_prediction == "leaning_forward" and confidence < 0.80:
                        raw_prediction = "good"

                pred_history.append(raw_prediction)
                prediction = Counter(pred_history).most_common(1)[0][0]

            skeleton_colour = get_skeleton_colour(prediction)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=skeleton_colour, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=skeleton_colour, thickness=4, circle_radius=6),
            )

            if prediction == "Ready":
                feedback = "Get ready to squat"
            elif prediction == "good":
                feedback = "Great form keep it up"
            elif prediction == "knees_in":
                feedback = "Push your knees out"
            elif prediction == "leaning_forward":
                feedback = "Keep your back straight"
            else:
                feedback = ""

            if feedback != "":
                speak(feedback)

            # draw blue info banner at the top of the screen
            cv2.rectangle(image, (0, 0), (1100, 130), (200, 80, 30), -1)

            # labels on top row
            cv2.putText(image, "VIEW", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "CLASS", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "COUNT", (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "STAGE", (480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "FEEDBACK", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # values on bottom row
            cv2.putText(image, view, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, prediction, (150, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, skeleton_colour, 2)
            cv2.putText(image, str(counter), (350, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, current_stage, (480, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, feedback, (650, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, skeleton_colour, 2)

        cv2.imshow("Live Webcam", image)

        # handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            # reset the rep counter and stage
            counter = 0
            current_stage = ""
            pred_history.clear()

live_cam.release()
cv2.destroyAllWindows()