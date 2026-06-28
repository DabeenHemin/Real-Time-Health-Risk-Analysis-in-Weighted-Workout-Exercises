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
ready_spoken = False

# tracks rep count and stage for both views separately
front_counter = 0
side_counter = 0
current_stage = ""

# hip-drop counting state (front view). these self-calibrate per session:
# standing_hip_y    = where the hips sit when the person is upright
# standing_leg_span = hip-to-ankle distance when upright, used to make the
#                     drop measurement independent of distance from the camera
standing_hip_y = None
standing_leg_span = None

# rolling window of confident leaning calls (live version of the video-level
# leaning rule). each side frame appends 1 (confident lean) or 0; if the share
# over the window crosses the threshold, the whole readout shows leaning.
side_lean_window = deque(maxlen=45)   # ~1.5s at 30fps; tune if needed
LEAN_CONF_SHARE = 0.393               # carried over from the test-set analysis

# uses Mac native say command for reliable text to speech
def speak(message):
    global is_speaking, last_spoken, last_speak_time, ready_spoken
    if is_speaking:
        return

    if message == "Get ready to squat":
        if ready_spoken:
            return
        ready_spoken = True

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
prediction = "Standing"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

def vertical_trunk_angle(shoulder, hip):
    # how far the shoulder->hip line tilts from vertical (0 = upright,
    # larger = more forward lean). must match the extraction/training exactly.
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    radians = np.arctan2(abs(dx), abs(dy))
    angle = radians * 180.0 / np.pi
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
    elif prediction == "Standing":
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

            # vertical trunk angle: torso tilt from upright. used by the retrained
            # SIDE model (13 features) to detect leaning_forward.
            left_vertical_trunk_angle  = vertical_trunk_angle(ls, lh)
            right_vertical_trunk_angle = vertical_trunk_angle(rs, rh)

            knee_distance          = abs(lk[0] - rk[0])
            ankle_distance         = abs(la[0] - ra[0])
            knee_ankle_ratio       = knee_distance / ankle_distance if ankle_distance != 0 else 0
            left_knee_foot_offset  = abs(lk[0] - la[0])
            right_knee_foot_offset = abs(rk[0] - ra[0])
            avg_offset             = (left_knee_foot_offset + right_knee_foot_offset) / 2

            avg_knee = (left_knee_angle + right_knee_angle) / 2

            # check visibility of key landmarks before counting reps
            hip_vis = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].visibility +
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility) / 2
            knee_vis = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility +
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility) / 2
            ankle_vis = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility +
                         lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility) / 2

            # use different visibility thresholds based on view
            # side view has lower visibility because one side is hidden
            if view == "Front":
                visibility_threshold = 0.7
            else:
                visibility_threshold = 0.4

            # --- hip-drop measurement (used for FRONT-view rep counting) ---
            # the front 2D knee angle compresses depth, so instead we measure how
            # far the hips drop vertically (y grows downward: bigger = lower).
            hip_y   = (lh[1] + rh[1]) / 2
            ankle_y = (la[1] + ra[1]) / 2
            leg_span = abs(ankle_y - hip_y)

            # refresh the standing baseline whenever the person is upright
            if avg_knee > 165:
                standing_hip_y = hip_y
                if leg_span > 0:
                    standing_leg_span = leg_span

            # how far the hips have dropped, as a fraction of standing leg length
            if standing_hip_y is not None and standing_leg_span:
                hip_drop_ratio = (hip_y - standing_hip_y) / standing_leg_span
            else:
                hip_drop_ratio = 0.0

            # >>> THE ONE TUNABLE NUMBER (front rep counting) <<<
            # hips must drop this fraction of leg length to count as a squat.
            FRONT_DROP_DOWN = 0.10
            FRONT_DROP_UP   = 0.04

            # only track squat stage and count reps if landmarks are visible enough
            if hip_vis > visibility_threshold and knee_vis > visibility_threshold and ankle_vis > visibility_threshold:
                if view == "Front":
                    # FRONT view: count by hip drop (reliable from this angle)
                    if hip_drop_ratio > FRONT_DROP_DOWN:
                        current_stage = "down"
                    if hip_drop_ratio < FRONT_DROP_UP and current_stage == "down":
                        current_stage = "up"
                        front_counter += 1
                else:
                    # SIDE view: knee angle is accurate here, keep using it
                    if avg_knee < 95:
                        current_stage = "down"
                    if avg_knee > 165 and current_stage == "down":
                        current_stage = "up"
                        side_counter += 1

            # FRONT model uses the original 11 features
            front_features = pd.DataFrame([[
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

            # SIDE model uses the same 11 PLUS the two vertical trunk angles (13 total)
            side_features = pd.DataFrame([[
                left_knee_angle, left_hip_angle, left_trunk_angle,
                right_knee_angle, right_hip_angle, right_trunk_angle,
                knee_distance, ankle_distance, knee_ankle_ratio,
                left_knee_foot_offset, right_knee_foot_offset,
                left_vertical_trunk_angle, right_vertical_trunk_angle
            ]], columns=[
                "left_knee_angle", "left_hip_angle", "left_trunk_angle",
                "right_knee_angle", "right_hip_angle", "right_trunk_angle",
                "knee_distance", "ankle_distance", "knee_ankle_ratio",
                "left_knee_foot_offset", "right_knee_foot_offset",
                "left_vertical_trunk_angle", "right_vertical_trunk_angle"
            ])

            if avg_knee > 170:
                prediction = "Standing"
                pred_history.clear()
            else:
                if view == "Front":
                    scaled_features = front_scaler.transform(front_features)
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
                    scaled_features = side_scaler.transform(side_features)
                    proba           = side_model.predict_proba(scaled_features)[0]
                    confidence      = max(proba)
                    raw_prediction  = side_label_encoder.inverse_transform(
                        side_model.predict(scaled_features)
                    )[0]

                    if raw_prediction == "leaning_forward" and confidence < 0.80:
                        raw_prediction = "good"

                    # track confident leaning over a rolling window (live version
                    # of the video-level leaning rule). 1 = confident lean frame.
                    side_lean_window.append(
                        1 if (raw_prediction == "leaning_forward" and confidence >= 0.90) else 0
                    )

                pred_history.append(raw_prediction)
                prediction = Counter(pred_history).most_common(1)[0][0]

                # leaning verdict: if a confident share of recent side frames show
                # leaning, show leaning even if the instant smoothing missed it.
                if view == "Side" and len(side_lean_window) > 0:
                    if sum(side_lean_window) / len(side_lean_window) >= LEAN_CONF_SHARE:
                        prediction = "leaning_forward"

            skeleton_colour = get_skeleton_colour(prediction)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=skeleton_colour, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=skeleton_colour, thickness=4, circle_radius=6),
            )

            if prediction == "Standing":
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
            cv2.putText(image, "FRONT", (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "SIDE", (460, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "STAGE", (560, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, "FEEDBACK", (720, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # values on bottom row
            cv2.putText(image, view, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, prediction, (150, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, skeleton_colour, 2)
            cv2.putText(image, str(front_counter), (350, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, str(side_counter), (460, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, current_stage, (560, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, feedback, (720, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, skeleton_colour, 2)

        cv2.imshow("Live Webcam", image)

        # handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            # reset both rep counters and stage
            front_counter = 0
            side_counter = 0
            current_stage = ""
            pred_history.clear()
            side_lean_window.clear()
            standing_hip_y = None
            standing_leg_span = None

live_cam.release()
cv2.destroyAllWindows()