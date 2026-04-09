#Imports

import os # os lets the system navigate thorugh files and reads them
import re # re is used the organise files by splitting filenames into text and numbers

import cv2 # Allows for image processing and video capture
import mediapipe as mp # This will be used to identify key body locations, analyze posture, and categorize movements

squat_dataset = "../../data/squat/Unfinished_Optimised_Squat_Dataset"
mp_pose = mp.solutions.pose

# Sorts out files via splitting the numbers from the rest of the filename
def file_sorting(name):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# Checks the dataset folder exists before doing anything
if not os.path.exists(squat_dataset):
    print(f"ERROR: Dataset folder not found at: {squat_dataset}")
    exit()

# Select the first available video from the dataset for testing
test_video_path = None
for view in ["Side", "Front"]:
    view_path = os.path.join(squat_dataset, view)

    # Skips over missing view folders
    if not os.path.exists(view_path):
        print(f"WARNING: {view} folder not found, skipping...")
        continue

    for folder in os.listdir(view_path):
        folder_path = os.path.join(view_path, folder)

        # Ignore files and only process label directories
        if not os.path.isdir(folder_path):
            continue

        for f in sorted(os.listdir(folder_path), key=file_sorting):
            if f.lower().endswith((".mp4", ".mov", ".avi", ".m4v")):
                test_video_path = os.path.join(folder_path, f)
                break

        if test_video_path:
            break

    if test_video_path:
        break

# Check a video was actually found before trying to open it
if test_video_path is None:
    print("ERROR: No video files found in the dataset. Check your folder structure.")
    exit()

print(f"Testing on: {test_video_path}")

cap = cv2.VideoCapture(test_video_path)

# Check the video file opened successfully
if not cap.isOpened():
    print(f"ERROR: Could not open video file: {test_video_path}")
    exit()

frame_count = 0
mp_drawing = mp.solutions.drawing_utils

# Initialise MediaPipe Pose for landmark detection on a user
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Frame skipping disabled for Stage 1 so every frame is visible for testing
        # if frame_count % 3 != 0:
        #     continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Catches any errors that may occur during MediaPipe processing
        try:
            results = pose.process(rgb_frame)
        except Exception as e:
            print(f"ERROR: MediaPipe failed on frame {frame_count}: {e}")
            continue

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Draw the full skeleton and keypoints onto the frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Extract left and right knee landmark positions
            lk = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            # Highlight the knee landmarks specifically in red
            h, w, _ = frame.shape
            lk_px = (int(lk.x * w), int(lk.y * h))
            rk_px = (int(rk.x * w), int(rk.y * h))
            cv2.circle(frame, lk_px, 8, (0, 0, 255), -1)
            cv2.circle(frame, rk_px, 8, (0, 0, 255), -1)

            # Label the knee landmarks on the frame
            cv2.putText(frame, "L Knee", (lk_px[0] + 10, lk_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "R Knee", (rk_px[0] + 10, rk_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Show the current frame number on screen
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            print(
                f"Frame {frame_count}: "
                f"Left knee x={lk.x:.3f} y={lk.y:.3f} | "
                f"Right knee x={rk.x:.3f} y={rk.y:.3f}"
            )
        else:
            print(f"Frame {frame_count}: No person detected")

        # Display the frame in a window at normal speed (30ms per frame)
        cv2.imshow("Squat Pose Detection Test", frame)

        # Press Q at any time to quit the window early
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Test aborted")
            break

cv2.destroyAllWindows()
cap.release()
print("MediaPipe is detecting landmarks correctly")