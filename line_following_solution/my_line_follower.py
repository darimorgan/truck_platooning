#!/usr/bin/env python3
"""
HSHL Line Following Student Lab — Your Implementation
=====================================================

Implement your line following algorithm by filling in the function below:

    detect_line(image)  — called for every camera frame (~30 fps)

─────────────────────────────────────────────────────────────────────────────
INPUTS  (what you receive from the camera)
─────────────────────────────────────────────────────────────────────────────
Camera frame  →  detect_line(image)
  image         np.ndarray, shape (720, 1280, 3), BGR colour order
                Same convention as OpenCV.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS  (what your function must return)
─────────────────────────────────────────────────────────────────────────────
detect_line(image)  →  float | None
  Return a steering value in range [-1.0, 1.0]:
    -1.0  = steer full left
     0.0  = go straight (line is centered)
    +1.0  = steer full right
        None  = cannot detect line (framework uses neutral steering fallback)

─────────────────────────────────────────────────────────────────────────────
ALGORITHM TIPS
─────────────────────────────────────────────────────────────────────────────
1. The line is painted GREEN on the road (BGR: 0, 255, 0)
2. Use color range thresholding to detect green pixels
3. Find the line center using contour moments
4. Compare line center to image center to get steering offset
5. Use morphological operations to reduce noise
6. Return None if no line is detected

See docs/line_detection_example.py for a complete example implementation.

─────────────────────────────────────────────────────────────────────────────
HELPERS
─────────────────────────────────────────────────────────────────────────────
    self.show_notification(text)  white  — general info
    self.show_warning(text)       yellow — caution
    self.show_alert(text)         red    — critical
    self.current_image            latest camera frame (or None)
"""
from dataclasses import dataclass
from pathlib import Path

import cv2          # type: ignore
import joblib       # type: ignore
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import rclpy        # type: ignore

from .controller import LineFollowerConfig, detect_green_line
from .interface import LineFollowingInterface


@dataclass(frozen=True)
class SvmFeatureConfig:
    """Fixed parameters for SVM diagnostic feature extraction."""

    kernel_size: tuple[int, int] = (7, 7)
    min_green_pixels: int = 150
    min_pixels_per_band: int = 12


@dataclass(frozen=True)
class FallbackConfig:
    """Fixed parameters for short line-loss steering fallback."""

    hold_frames: int = 15
    decay: float = 0.95
    zero_deadband: float = 0.000002


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following.

    Drive from the deterministic contour controller and compare its direction
    against the trained SVM prediction for diagnostics.
    """

    def __init__(self):
        super().__init__("my_line_follower")

        self.config = LineFollowerConfig()
        self.svm_feature_config = SvmFeatureConfig()
        self.fallback_config = FallbackConfig()
        model_path = Path(__file__).resolve().parents[1] / "svm_green_line_model.joblib"
        payload = joblib.load(model_path)

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.hsv_lower = np.array(payload["hsv_lower"], dtype=np.uint8)
        self.hsv_upper = np.array(payload["hsv_upper"], dtype=np.uint8)
        self.roi_start_fraction = float(payload["roi_start_fraction"])

        self._frame_count = 0
        self._last_controller_steering = 0.0
        self._missed_line_frames = 0

        # Register camera callback
        self.on_camera_image(self.detect_line)
        self.get_logger().info("MyLineFollower initialized — controller active, SVM diagnostics loaded")

    def create_green_mask(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Return a denoised binary mask for the green route line in the lower ROI."""
        h, _ = image_bgr.shape[:2]
        roi_start = int(h * self.roi_start_fraction)
        roi = image_bgr[roi_start:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.svm_feature_config.kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask, roi_start

    def center_x_in_band(self, mask: np.ndarray, y0: int, y1: int) -> tuple[float, int]:
        """Return the mean x-position of green pixels in one horizontal band."""
        band = mask[y0:y1, :]
        _, xs = np.nonzero(band)

        if len(xs) < self.svm_feature_config.min_pixels_per_band:
            return np.nan, int(len(xs))

        return float(xs.mean()), int(len(xs))

    def extract_centerline_features(self, image_bgr: np.ndarray):
        """
        Extract the same numeric features used during SVM training.

        Returns:
            (features, mask, roi_start), or (None, mask, roi_start) if the line
            cannot be detected robustly enough for inference.
        """
        _, w = image_bgr.shape[:2]
        mask, roi_start = self.create_green_mask(image_bgr)
        green_pixels = int(np.count_nonzero(mask))

        if green_pixels < self.svm_feature_config.min_green_pixels:
            return None, mask, roi_start

        roi_h = mask.shape[0]
        bands = {
            "near": (int(0.70 * roi_h), roi_h),
            "middle": (int(0.45 * roi_h), int(0.70 * roi_h)),
            "far": (0, int(0.45 * roi_h)),
        }

        centers = {}
        counts = {}
        for name, (y0, y1) in bands.items():
            centers[name], counts[name] = self.center_x_in_band(mask, y0, y1)

        valid_centers = [x for x in centers.values() if not np.isnan(x)]
        if len(valid_centers) < 2:
            return None, mask, roi_start

        fallback_x = float(np.mean(valid_centers))
        near_x = centers["near"] if not np.isnan(centers["near"]) else fallback_x
        middle_x = centers["middle"] if not np.isnan(centers["middle"]) else fallback_x
        far_x = centers["far"] if not np.isnan(centers["far"]) else fallback_x

        near_offset = (near_x - w / 2.0) / (w / 2.0)
        middle_offset = (middle_x - w / 2.0) / (w / 2.0)
        far_offset = (far_x - w / 2.0) / (w / 2.0)

        direction_score = (far_x - near_x) / w
        curvature_proxy = (middle_x - 0.5 * (near_x + far_x)) / (w / 2.0)

        left_pixels = int(np.count_nonzero(mask[:, : w // 2]))
        right_pixels = int(np.count_nonzero(mask[:, w // 2 :]))
        left_right_balance = (right_pixels - left_pixels) / max(1, right_pixels + left_pixels)

        ys, xs = np.nonzero(mask)
        bbox_width = float(xs.max() - xs.min() + 1) if len(xs) else 0.0
        bbox_height = float(ys.max() - ys.min() + 1) if len(ys) else 0.0

        features = {
            "near_x": near_x,
            "middle_x": middle_x,
            "far_x": far_x,
            "near_offset": near_offset,
            "middle_offset": middle_offset,
            "far_offset": far_offset,
            "direction_score": direction_score,
            "curvature_proxy": curvature_proxy,
            "green_pixel_ratio": green_pixels / mask.size,
            "left_right_balance": left_right_balance,
            "bbox_width_norm": bbox_width / w,
            "bbox_height_norm": bbox_height / roi_h,
            "near_pixels": counts["near"],
            "middle_pixels": counts["middle"],
            "far_pixels": counts["far"],
            "roi_start": roi_start,
        }

        return features, mask, roi_start

    def features_to_frame(self, features: dict) -> pd.DataFrame:
        """Convert feature dictionary to the column order expected by the SVM."""
        values = [[features[column] for column in self.feature_columns]]
        return pd.DataFrame(values, columns=self.feature_columns, dtype=np.float64)

    def predict_direction(self, features: dict) -> str:
        """Return the SVM direction label: LEFT, STRAIGHT, or RIGHT."""
        x = self.features_to_frame(features)
        return str(self.model.predict(x)[0])

    def controller_direction(self, steering: float) -> str:
        """Convert deterministic steering into a direction label."""
        if steering < -self.config.steering_deadband:
            return "LEFT"
        if steering > self.config.steering_deadband:
            return "RIGHT"
        return "STRAIGHT"

    def svm_prediction_for_frame(self, image: np.ndarray) -> str | None:
        """Return the SVM direction prediction, or None if features are unavailable."""
        features, _, _ = self.extract_centerline_features(image)
        if features is None:
            return None

        return self.predict_direction(features)

    def detect_line(self, image: np.ndarray) -> float | None:
        """
        Detect the green line and return steering command.

        Args:
            image: BGR image from camera, shape (720, 1280, 3)

        Returns:
            Steering value in [-1.0, 1.0], or None if line not detected.
        """
        detection = detect_green_line(image, self.config)
        if detection.steering is None:
            fallback_steering = self._last_controller_steering
            if self._missed_line_frames >= self.fallback_config.hold_frames:
                fallback_steering *= self.fallback_config.decay

            if abs(fallback_steering) < self.fallback_config.zero_deadband:
                fallback_steering = 0.0

            self._last_controller_steering = fallback_steering
            self._missed_line_frames += 1
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                self.get_logger().info(
                    f"controller={detection.status} fallback_steer={fallback_steering:.6f} "
                    f"missed_frames={self._missed_line_frames} "
                    f"frame={self._frame_count}"
                )
            self.show_notification(f"controller={detection.status} fallback={fallback_steering:.2f}")
            return fallback_steering

        steering = detection.steering
        self._last_controller_steering = steering
        self._missed_line_frames = 0
        controller_prediction = self.controller_direction(steering)
        svm_prediction = self.svm_prediction_for_frame(image)
        if svm_prediction is None:
            svm_status = "unavailable"
        elif svm_prediction == controller_prediction:
            svm_status = "accepted"
        else:
            svm_status = "declined"

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"ctrl={controller_prediction} svm={svm_prediction or 'NONE'} "
                f"svm_status={svm_status} steer={steering:.2f} "
                f"line_x={detection.line_center_x:.1f} frame={self._frame_count}"
            )
        self.show_notification(
            f"ctrl={controller_prediction} svm={svm_prediction or 'NONE'} "
            f"{svm_status} steer={steering:.2f}"
        )

        return steering


def main(args=None):
    """Main entry point for the line follower node."""
    rclpy.init(args=args)
    follower = MyLineFollower()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
