# HSHL Line Following Lab

This lab has one goal: implement line following (detect the green line and output steering).

You only edit one file:
- line_following_solution/my_line_follower.py

Everything else (camera subscription, command publishing, ROS setup, Docker) is already provided.

For the lab baseline, the framework continuously sends a constant forward throttle (about 30 km/h target behavior), and your code controls steering.

If the student node is not connected, the lab demo stays in manual control.

## Quick Start (Student Path)

1. Clone and enter repo.

git clone https://github.com/AAT24/student_lineFollowing_HSHL.git
cd student_lineFollowing_HSHL

2. Place the instructor bag in bags/. The folder must contain metadata.yaml and .db3.

Expected layout:
student_lineFollowing_HSHL/
bags/
session_2026-XX-XX_XX-XX-XX/
metadata.yaml
session_..._0.db3

3. Run bag replay stack.

docker compose --profile bag up --build

4. In a second terminal, watch your node logs.

docker compose logs -f line-follower

5. Stop when done.

docker compose --profile bag down --remove-orphans

## What To Implement

Edit line_following_solution/my_line_follower.py and implement:

detect_line(self, image) -> float | None

Return value meaning:
- -1.0 means steer left
- 0.0 means straight
- +1.0 means steer right
- None means line not detected (framework falls back to straight steering)

Reference example:
- docs/line_detection_example.py

## Visualize Steering Offline (Recommended)

Use this one command on Windows:

powershell -ExecutionPolicy Bypass -File .\run_visualization.ps1

What it does:
- creates .venv if needed
- installs missing packages (opencv-python, rosbags, numpy)
- auto-detects bag folder
- extracts camera frames
- runs steering visualization and writes CSV

Useful options:

Start compose first:
powershell -ExecutionPolicy Bypass -File .\run_visualization.ps1 -StartCompose

No preview window:
powershell -ExecutionPolicy Bypass -File .\run_visualization.ps1 -NoPreview

Outputs:
- visualization_output/frames
- visualization_output/steering.csv

## Manual Visualization Commands

If you do not want the helper script:

python docs/extract_images_from_bag.py ./bags/session_2026-XX-XX_XX-XX-XX --out extracted_images
python visualize_line_follower.py --images extracted_images --output visualization_output --show

## Fast Iteration (No Rebuild)

In docker-compose.yaml, uncomment volume mount for line_following_solution, then restart:

docker compose restart line-follower

This lets code edits apply instantly without rebuilding images.

## Minimal Validation Checklist

Before submission, confirm:
- compose starts without errors
- line-follower receives camera frames
- steering.csv has status ok rows and steer values
- your steering direction is correct (left line -> negative steer, right line -> positive steer)
