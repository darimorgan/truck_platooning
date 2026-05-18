#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import types

import cv2
import importlib
import numpy as np


def install_stubs() -> None:
    # Stub ROS imports so student code can be imported outside Docker/ROS.
    if "rclpy" not in sys.modules:
        rclpy_stub = types.ModuleType("rclpy")
        rclpy_stub.init = lambda *args, **kwargs: None
        rclpy_stub.spin = lambda *args, **kwargs: None
        rclpy_stub.shutdown = lambda *args, **kwargs: None
        sys.modules["rclpy"] = rclpy_stub

    if "line_following_solution.interface" not in sys.modules:
        interface_stub = types.ModuleType("line_following_solution.interface")

        class LineFollowingInterface:
            pass

        interface_stub.LineFollowingInterface = LineFollowingInterface
        sys.modules["line_following_solution.interface"] = interface_stub


class DummySelf:
    def __init__(self) -> None:
        self._frame_count = 0

    class _Logger:
        def info(self, msg: str) -> None:
            _ = msg

        def warning(self, msg: str) -> None:
            _ = msg

        def error(self, msg: str) -> None:
            _ = msg

    def get_logger(self) -> "DummySelf._Logger":
        return DummySelf._Logger()

    def show_notification(self, text: str, level: str = "info", duration: float = 3.0) -> None:
        _ = (text, level, duration)

    def show_warning(self, text: str, duration: float = 5.0) -> None:
        _ = (text, duration)

    def show_alert(self, text: str, duration: float = 5.0) -> None:
        _ = (text, duration)


def load_detector():
    install_stubs()
    module = importlib.import_module("line_following_solution.my_line_follower")
    follower_cls = getattr(module, "MyLineFollower", None)
    if follower_cls is None or not hasattr(follower_cls, "detect_line"):
        raise RuntimeError("Could not find MyLineFollower.detect_line in line_following_solution/my_line_follower.py")
    return follower_cls.detect_line


def draw_steering_overlay(image: np.ndarray, steer: float | None) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    cx = w // 2
    y = h - 40

    cv2.line(out, (cx - 220, y), (cx + 220, y), (160, 160, 160), 2)
    cv2.line(out, (cx, y - 12), (cx, y + 12), (200, 200, 200), 2)

    if steer is None:
        text = "steer=None (no command sent)"
        color = (0, 180, 255)
        cv2.putText(out, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return out

    steer = float(np.clip(steer, -1.0, 1.0))
    px = int(cx + steer * 220)
    cv2.circle(out, (px, y), 10, (0, 0, 255), -1)

    if steer < -0.05:
        direction = "LEFT"
    elif steer > 0.05:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"

    text = f"steer={steer:+.3f}  ({direction})"
    cv2.putText(out, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize steering behavior from student detect_line(image).")
    parser.add_argument("--images", default="extracted_images", help="Folder with extracted images.")
    parser.add_argument("--output", default="visualization_output", help="Folder for annotated frames and CSV.")
    parser.add_argument("--max", type=int, default=0, help="Max frames to process (0 = all).")
    parser.add_argument("--show", action="store_true", help="Show a preview window while processing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.images):
        print(f"Images folder not found: {args.images}")
        return 1

    os.makedirs(args.output, exist_ok=True)
    frames_out = os.path.join(args.output, "frames")
    os.makedirs(frames_out, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        print(f"No images found in {args.images}")
        return 1

    if args.max > 0:
        image_files = image_files[:args.max]

    detect_line = load_detector()
    dummy = DummySelf()

    csv_path = os.path.join(args.output, "steering.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "steer", "status"])

        for idx, fname in enumerate(image_files, start=1):
            img_path = os.path.join(args.images, fname)
            image = cv2.imread(img_path)
            if image is None:
                writer.writerow([fname, "", "read_error"])
                continue

            try:
                steer = detect_line(dummy, image)
                if steer is not None:
                    steer = float(np.clip(float(steer), -1.0, 1.0))
                status = "ok"
            except Exception as exc:
                steer = None
                status = f"error:{exc}"

            writer.writerow([fname, "" if steer is None else f"{steer:.6f}", status])

            annotated = draw_steering_overlay(image, steer)
            cv2.imwrite(os.path.join(frames_out, fname), annotated)

            if args.show:
                cv2.imshow("Steering Visualization", annotated)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(image_files)} frames...")

    if args.show:
        cv2.destroyAllWindows()

    print("Done.")
    print(f"Annotated frames: {frames_out}")
    print(f"Steering CSV: {csv_path}")
    print("Tip: open steering.csv to see whether your code tends to left/right/straight over time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
