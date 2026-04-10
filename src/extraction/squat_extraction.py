#Import Packages

import os # os lets the system navigate thorough files and reads them
import re # re is used the organise files by splitting filenames into text and numbers

import cv2 # Allows for image processing and video capture
import mediapipe as mp # This will be used to identify key body locations, analyze posture, and categorize movements

import numpy as np  # Numpy is used for numerical calculations and array operations needed for angle calculations
import csv  # Used to create and write data to CSV files

squat_dataset = "../../data/squat/Unfinished_Optimised_Squat_Dataset"
mp_pose = mp.solutions.pose

# Sorts out files via splitting the numbers from the rest of the filename
def file_sorting(name):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# Calculates the angles between the 3 joint points using arctan2
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

#  Checks the dataset folder to see if it exists before doing anything
if not os.path.exists(squat_dataset):
    print(f"ERROR: Dataset folder not found at: {squat_dataset}")
    exit()

# Sets column headers for both CSVs
csv_header = [
    "file",
    "class",
    "left_knee_angle",
    "left_hip_angle",
    "left_trunk_angle",
    "right_knee_angle",
    "right_hip_angle",
    "right_trunk_angle",
    "knee_distance",
    "ankle_distance",
    "knee_ankle_ratio",
    "left_knee_foot_offset",
    "right_knee_foot_offset"
]

# Output paths for both CSVs
csv_files = {
    "Side":  "../../data/squat/side.csv",
    "Front": "../../data/squat/front.csv"
}

# Creates both the side squat and front squat CSV files and write the header row
for view in ["Side", "Front"]:
    with open(csv_files[view], mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

# Loops through every video in both views
for view in ["Side", "Front"]:
    dataset_path = os.path.join(squat_dataset, view)
    print(f"\n=== Processing {view} view ===")

    # SkipS if the view folder is missing
    if not os.path.exists(dataset_path):
        print(f"WARNING: {view} folder not found, skipping...")
        continue

    try:
        for folder in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, folder)

            # Only process label directories
            if not os.path.isdir(folder_path):
                continue

            # Folder name is the label (good / knees_in / leaning_forward)
            label = folder
            print(f"  Label: {label}")

            try:
                for video_file in sorted(os.listdir(folder_path), key=file_sorting):
                    if not video_file.lower().endswith(".mp4",):
                        continue

                    video_path = os.path.join(folder_path, video_file)
                    print(f"    Processing: {video_file}")

                    cap = cv2.VideoCapture(video_path)

                    # Skips if video cant be opened
                    if not cap.isOpened():
                        print(f"    WARNING: Could not open {video_file}, skipping...")
                        continue

                    frames_processed = 0
                    rows_written = 0

                    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Processes every third frame to reduce duplicate data and improve speed
                            frames_processed += 1
                            if frames_processed % 3 != 0:
                                continue

                            try:
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = pose.process(image)
                            except Exception as e:
                                print(f"    ERROR: MediaPipe failed on frame {frames_processed}: {e}")
                                continue

                            if results.pose_landmarks:
                                lm = results.pose_landmarks.landmark

                                # Gets the x and y position of each joint from mediapipe
                                ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,       lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,      lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                                la = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                                rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                                rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,      lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                rk = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,     lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                                ra = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                                # calculate angles for both left and right sides
                                left_knee_angle  = calculate_angle(lh, lk, la)
                                left_hip_angle   = calculate_angle(ls, lh, lk)
                                left_trunk_angle = calculate_angle(ls, lh, la)
                                right_knee_angle  = calculate_angle(rh, rk, ra)
                                right_hip_angle   = calculate_angle(rs, rh, rk)
                                right_trunk_angle = calculate_angle(rs, rh, ra)

                                # calculate distance features for knees_in detection
                                knee_distance        = abs(lk[0] - rk[0])
                                ankle_distance       = abs(la[0] - ra[0])
                                knee_ankle_ratio     = knee_distance / ankle_distance if ankle_distance != 0 else 0
                                left_knee_foot_offset  = abs(lk[0] - la[0])
                                right_knee_foot_offset = abs(rk[0] - ra[0])

                                # Write this frames data to the correct CSV
                                with open(csv_files[view], mode="a", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        video_file, label,
                                        left_knee_angle, left_hip_angle, left_trunk_angle,
                                        right_knee_angle, right_hip_angle, right_trunk_angle,
                                        knee_distance, ankle_distance, knee_ankle_ratio,
                                        left_knee_foot_offset, right_knee_foot_offset
                                    ])
                                rows_written += 1

                    cap.release()
                    print(f"      {rows_written} rows written")

            # Handles errors inside an individual label folder
            except Exception as e:
                print(f"  ERROR in label folder {label}: {e}")

    # Handles errors in the main view folder
    except Exception as e:
        print(f"ERROR in view folder {view}: {e}")

print("\nAll side and front squat angle data has been collected")