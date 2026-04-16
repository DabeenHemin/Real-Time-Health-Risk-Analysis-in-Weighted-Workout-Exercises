# Dabeen Hemin - FYP: Real-Time Health Risk Analysis
# Live squat form analysis app using trained front and side view models

# Import Packages
import warnings
warnings.filterwarnings("ignore")
import os   # used to build file paths to the model files
os.environ["GLOG_minloglevel"] = "3"

import cv2  # Allows for image processing and video capture
import mediapipe as mp  # This will be used to identify key body locations, analyze posture, and categorize movements
import numpy as np      # used for numerical calculations needed for angle calculation
import joblib           # used to load the trained model files

# Initialise the Mediapipe pose to inspect the live cam footage model and find landmarks (keypoints)
# Initialise drawing utilities which draw connections between landmarks to create a skeleton overlay over an individual
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
# measures horizontal distance between shoulders to determine view
def detect_view(landmarks):
    left_shoulder_x  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x

    # if shoulders are far apart the user is facing front
    # if shoulders are close together the user is side on
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)

    if shoulder_width > 0.08:
        return "Front"
    else:
        return "Side"

# Start video capture from default webcam
live_cam = cv2.VideoCapture(0)

# Detects if webcam is available
if not live_cam.isOpened():
    print("Live cam not enabled")
    exit()

# Creates and implements pose detection model with 2 confidence thresholds
# to tell how confident it has to be to detect a person and how confident it is to detect a pose and keep following it smoothly across frames
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # Process frames continuously while webcam is active
    while live_cam.isOpened():

        ret, frame = live_cam.read()
        # Stop if frame cannot be read
        if not ret:
            print("Live feed over")
            break

        frame = cv2.flip(frame, 1)  # Flips the live cam to make it mirror like and not confuse a user whilst using it

        # Convert image to RGB (required for MediaPipe processing)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=6),
            )

            # detect which view the user is in
            view = detect_view(results.pose_landmarks.landmark)

            # display the detected view on screen
            cv2.putText(image, f"View: {view}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the processed frame
        cv2.imshow("Live Webcam", image)

        # Stops program when the 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

live_cam.release()
cv2.destroyAllWindows()