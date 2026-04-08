import os
import re

squat_dataset = "../../data/squat/Unfinished_Optimised_Squat_Dataset"

# Sorts out files via spliting the numbers from the rest of the filename
def file_sorting(name):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

total_videos = 0

# Checks to see if dataset path does exist before processing
if not os.path.exists(squat_dataset):
    print("Dataset path doesn't exist.")
else:
    # Process both camera views in the dataset
    for view in ["Side", "Front"]:
        view_path = os.path.join(squat_dataset, view)
        print(f"\n=== {view} view ===")

        #skips over missing view folders
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

                    for f in videos:
                        print(os.path.join(folder_path, f))
                        total_videos += 1
                # Handles errors inside an individual label folder
                except Exception as e:
                    print(f"Couldn't read folder {folder_path}: {e}")
        # Handle errors in a main view folder
        except Exception as e:
            print(f"Error reading view folder {view_path}: {e}")

# Dataset Summary
print(f"\nTotal videos found: {total_videos}")
print(f"Dataset path: {squat_dataset}")