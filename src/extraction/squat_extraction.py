import os
import re
import cv2

squat_dataset = "../../data/squat/Unfinished_Optimised_Squat_Dataset"

# Sorts out files via splitting the numbers from the rest of the filename
def file_sorting(name):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

total_frames = 0
videos_checked = 0

# Checks to see if dataset path does exist before processing
if not os.path.exists(squat_dataset):
    print("Dataset path doesn't exist.")
else:
    # Process both camera views in the dataset
    for view in ["Side", "Front"]:
        view_path = os.path.join(squat_dataset, view)
        print(f"\n=== {view} view ===")

        # Skips over missing view folders
        if not os.path.exists(view_path):
            print(f"Missing folder: {view_path}")
            continue

        try:
            for folder in sorted(os.listdir(view_path)):
                folder_path = os.path.join(view_path, folder)

                # Ignore files and only process label directories
                if not os.path.isdir(folder_path):
                    continue

                try:
                    videos = [
                        f for f in sorted(os.listdir(folder_path), key=file_sorting)
                        if f.lower().endswith(".mp4")
                    ]

                    print(f"\n  Squat Form: {folder} ({len(videos)} videos)")

                    for video_file in videos:
                        video_path = os.path.join(folder_path, video_file)

                        # Try to open each video with OpenCV
                        cap = cv2.VideoCapture(video_path)

                        if not cap.isOpened():
                            print(f"    WARNING: Could not open {video_file}")
                            continue

                        # Read basic video information
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        print(f"    {video_file} — {frames} frames | {fps:.1f} fps | {width}x{height}")

                        total_frames += frames
                        videos_checked += 1
                        cap.release()

                # Handles errors inside an individual label folder
                except Exception as e:
                    print(f"Couldn't read folder {folder_path}: {e}")

        # Handles errors in a main view folder
        except Exception as e:
            print(f"Error reading view folder {view_path}: {e}")

# Dataset Summary
print(f"\nTotal videos checked : {videos_checked}")
print(f"Total frames across all videos: {total_frames}")
print("OpenCV successfully read files")