# Test script that goes through multiple folders of squat videos
# Tests the system on real and AI-generated test footage

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["GLOG_minloglevel"] = "3"

from collections import Counter, deque
import csv
import re
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

# Initialise mediapipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# CHANGE THESE to the folders you want to test
# each folder maps to an expected outcome
test_folders = {
    "../../Test_Squat_Folder/front": "good",       # good form front squats
    "../../Test_Squat_Folder/side":  "good",       # good form side squats
    "../../Test_Squat_Folder/knees": "knees_in",   # knees caving in
    "../../Test_Squat_Folder/lean":  "leaning_forward"  # leaning forward
}

# load all 6 trained model files
models_path = "../../models"

front_model         = joblib.load(os.path.join(models_path, "front_model.pkl"))
front_scaler        = joblib.load(os.path.join(models_path, "front_scaler.pkl"))
front_label_encoder = joblib.load(os.path.join(models_path, "front_label_encoder.pkl"))

side_model         = joblib.load(os.path.join(models_path, "side_model.pkl"))
side_scaler        = joblib.load(os.path.join(models_path, "side_scaler.pkl"))
side_label_encoder = joblib.load(os.path.join(models_path, "side_label_encoder.pkl"))

print("All models loaded successfully!")


def natural_key(filename):
    # pull the digits out of the filename so videos sort by value (1, 2, ... 10, 11)
    # instead of as text (1, 10, 11, ... 2, 20), which is how sorted() treats strings
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


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
    elif prediction == "Standing":
        return (255, 255, 255)
    else:
        return (0, 0, 255)


def test_video(video_path, expected, folder_name):
    """Process a single video and return the test results"""

    # reset state for each new video
    front_counter = 0
    side_counter = 0
    current_stage = ""
    pred_history = deque(maxlen=5)
    prediction = "Standing"

    # hip-drop counting state (front view). these self-calibrate per video:
    # standing_hip_y  = where the hips sit when the person is upright
    # standing_leg_span = hip-to-ankle distance when upright, used to make the
    #                     drop measurement independent of how far they are from camera
    standing_hip_y = None
    standing_leg_span = None

    # track all predictions across the entire video
    all_predictions = []
    detected_views = []

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        return None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while video.isOpened():

            ret, frame = video.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:

                lm = results.pose_landmarks.landmark
                view = detect_view(lm)
                detected_views.append(view)

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

                hip_vis = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].visibility +
                           lm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility) / 2
                knee_vis = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility +
                            lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility) / 2
                ankle_vis = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility +
                             lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility) / 2

                if view == "Front":
                    visibility_threshold = 0.7
                else:
                    visibility_threshold = 0.4

                # --- hip-drop measurement (used for FRONT-view rep counting) ---
                # knee angle compresses depth from the front, so instead we measure
                # how far the hips drop vertically (y grows downward: bigger = lower).
                hip_y   = (lh[1] + rh[1]) / 2
                ankle_y = (la[1] + ra[1]) / 2
                leg_span = abs(ankle_y - hip_y)

                # capture / refresh the standing baseline whenever the person is upright
                if avg_knee > 165:
                    standing_hip_y = hip_y
                    if leg_span > 0:
                        standing_leg_span = leg_span

                # how far the hips have dropped, as a fraction of standing leg length
                # (normalising by leg length makes this independent of camera distance)
                if standing_hip_y is not None and standing_leg_span:
                    hip_drop_ratio = (hip_y - standing_hip_y) / standing_leg_span
                else:
                    hip_drop_ratio = 0.0

                # >>> THE ONE TUNABLE NUMBER <<<
                # hips must drop this fraction of leg length to count as a squat.
                # lower it (e.g. 0.08) if a real squat is missed; raise it if shallow
                # movements get counted by mistake.
                FRONT_DROP_DOWN = 0.10
                FRONT_DROP_UP   = 0.04

                print(f"avg_knee: {avg_knee:.1f} | hip_drop: {hip_drop_ratio:.3f} | stage: {current_stage} | view: {view} | front: {front_counter} | side: {side_counter}")

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
                    prediction = "Standing"
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

                        # DIAGNOSTIC (remove later): raw model prediction before override
                        print(f"  SIDE raw: {raw_prediction} | conf: {confidence:.2f} | avg_knee: {avg_knee:.0f}")

                        # suppress low-confidence leaning_forward calls. note: this only
                        # filters borderline frames -- the side model's main limitation is
                        # that it associates leaning with the upright/transition phase and
                        # labels the deep squat as "good", which no threshold can fix
                        # (would require retraining). documented as a limitation.
                        if raw_prediction == "leaning_forward" and confidence < 0.80:
                            raw_prediction = "good"

                    pred_history.append(raw_prediction)
                    prediction = Counter(pred_history).most_common(1)[0][0]
                    all_predictions.append(prediction)

                skeleton_colour = get_skeleton_colour(prediction)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=skeleton_colour, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=skeleton_colour, thickness=4, circle_radius=6),
                )

                # draw blue info banner
                cv2.rectangle(image, (0, 0), (1200, 130), (200, 80, 30), -1)

                cv2.putText(image, f"Folder: {folder_name}", (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(image, f"Expected: {expected}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(image, f"View: {view}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Class: {prediction}", (250, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, skeleton_colour, 2)
                cv2.putText(image, f"Front reps: {front_counter}", (500, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Side reps: {side_counter}", (750, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Video Test", image)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                video.release()
                cv2.destroyAllWindows()
                exit()
            if key == ord('n'):
                break
            if key == ord('p'):
                cv2.waitKey(0)

    video.release()

    # determine most common prediction (excluding Standing)
    non_standing = [p for p in all_predictions if p != "Standing"]
    if non_standing:
        most_common_class = Counter(non_standing).most_common(1)[0][0]
    else:
        most_common_class = "Standing"

    # determine most common view
    if detected_views:
        most_common_view = Counter(detected_views).most_common(1)[0][0]
    else:
        most_common_view = "Unknown"

    # check if classification matches expectation
    classification_correct = (most_common_class == expected)

    total_reps = front_counter + side_counter

    return {
        'filename': os.path.basename(video_path),
        'folder': folder_name,
        'expected': expected,
        'most_common_class': most_common_class,
        'classification_correct': classification_correct,
        'detected_view': most_common_view,
        'front_reps': front_counter,
        'side_reps': side_counter,
        'total_reps': total_reps
    }


# main testing loop
all_results = []
video_extensions = ('.mp4', '.mov', '.avi', '.MOV', '.MP4')

for folder_path, expected_outcome in test_folders.items():
    if not os.path.exists(folder_path):
        print(f"Folder not found, skipping: {folder_path}")
        continue

    folder_name = os.path.basename(folder_path)
    # sort by the number in the filename so videos run 1, 2, 3 ... 10, 11
    video_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(video_extensions)],
        key=natural_key
    )

    print(f"\n{'=' * 60}")
    print(f"Testing folder: {folder_name} ({len(video_files)} videos)")
    print(f"Expected outcome: {expected_outcome}")
    print(f"{'=' * 60}")

    for video_filename in video_files:
        video_path = os.path.join(folder_path, video_filename)
        print(f"\nTesting: {video_filename}")

        result = test_video(video_path, expected_outcome, folder_name)
        if result:
            all_results.append(result)
            print(f"  View: {result['detected_view']} | "
                  f"Class: {result['most_common_class']} | "
                  f"Reps: {result['total_reps']} | "
                  f"Correct: {result['classification_correct']}")

cv2.destroyAllWindows()

# save results to csv file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "../../docs/test_logs"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

csv_path = f"{results_dir}/test_results_{timestamp}.csv"

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'filename', 'folder', 'expected', 'most_common_class',
        'classification_correct', 'detected_view',
        'front_reps', 'side_reps', 'total_reps'
    ])
    writer.writeheader()
    writer.writerows(all_results)

# print summary
print("\n\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)
print(f"\nTotal videos tested: {len(all_results)}")

# accuracy per folder
for folder_path, expected in test_folders.items():
    folder_name = os.path.basename(folder_path)
    folder_results = [r for r in all_results if r['folder'] == folder_name]
    if folder_results:
        correct = sum(1 for r in folder_results if r['classification_correct'])
        total = len(folder_results)
        accuracy = (correct / total) * 100
        print(f"\n{folder_name}:")
        print(f"  Videos: {total}")
        print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")

# overall accuracy
total_correct = sum(1 for r in all_results if r['classification_correct'])
total_videos = len(all_results)
if total_videos > 0:
    overall_accuracy = (total_correct / total_videos) * 100
    print(f"\nOverall accuracy: {total_correct}/{total_videos} ({overall_accuracy:.1f}%)")

print(f"\nResults saved to: {csv_path}")