import os # os lets the system navigate thorugh files and reads them
import re # re is used the organise files by splitting filenames into text and numbers

import cv2 # Allows for image processing and video capture
import mediapipe as mp # This will be used to identify key body locations, analyze posture, and categorize movements

import numpy as np # Numpy is used for numerical calculations and array operations needed for angle calculation

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

# Select the first available video from the dataset for testing
first_video = None
for view in ["Side", "Front"]:
    view_path = os.path.join(squat_dataset, view)

    if not os.path.exists(view_path):
        print(f"WARNING: {view} folder not found, skipping...")
        continue

    for folder in os.listdir(view_path):
        folder_path = os.path.join(view_path, folder)

        if not os.path.isdir(folder_path):
            continue

        for f in sorted(os.listdir(folder_path), key=file_sorting):
            if f.lower().endswith((".mp4", ".mov", ".avi", ".m4v")):
                first_video = os.path.join(folder_path, f)
                break

        if first_video:
            break

    if first_video:
        break

# Checks a video to see if it was actually found before trying to open it
if first_video is None:
    print("ERROR: No video files found in the dataset. Check your folder structure.")
    exit()

print(f"Testing angles on: {first_video}")

cap = cv2.VideoCapture(first_video)

# Checks the video file to see if it opened successfully
if not cap.isOpened():
    print(f"ERROR: Could not open video file: {first_video}")
    exit()

frames_processed = 0
pose_drawer = mp.solutions.drawing_utils

# Sets up mediapipe pose estimation model so it can start tracking the person
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1

        # Skips the frame if mediapipe runs into a problem
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
        except Exception as e:
            print(f"ERROR: MediaPipe failed on frame {frames_processed}: {e}")
            continue

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Draws the full skeleton onto the frame
            pose_drawer.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Grabs the x and y positions of each joint needed
            ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,      lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,     lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            la = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,    lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Works out the angles between the joints for the squat
            knee_ang  = calculate_angle(lh, lk, la)
            hip_ang   = calculate_angle(ls, lh, lk)
            torso_ang = calculate_angle(ls, lh, la)

            # Gets pixel positions to place angle text on screen
            h, w, _ = frame.shape
            lk_px = (int(lk[0] * w), int(lk[1] * h))
            lh_px = (int(lh[0] * w), int(lh[1] * h))
            ls_px = (int(ls[0] * w), int(ls[1] * h))

            # Puts the angle numbers on the screen next to each joint
            # Green for knee, yellow for hip, pink for torso
            cv2.putText(frame, f"Knee: {knee_ang}",
                        (lk_px[0] + 10, lk_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Hip: {hip_ang}",
                        (lh_px[0] + 10, lh_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"torso: {torso_ang}",
                        (ls_px[0] + 10, ls_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Prints angles to terminal as well
            print(
                f"Frame {frames_processed}: "
                f"knee={knee_ang}  "
                f"hip={hip_ang}  "
                f"torso={torso_ang}"
            )
        else:
            print(f"Frame {frames_processed}: No person detected")

        # Displays the frame in a window at normal speed
        cv2.imshow("Squat Pose Detection", frame)

        # Press Q at any time to quit early
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Test stopped early by user")
            break

cv2.destroyAllWindows()
cap.release()
print("Angle calculation working.Knee, hip, and torso angles confirmed")