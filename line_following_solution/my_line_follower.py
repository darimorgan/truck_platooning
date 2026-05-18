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
import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore

from .interface import LineFollowingInterface


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following.
    
    Detect a green line and steer to stay centered on it.
    """

    def __init__(self):
        super().__init__("my_line_follower")
        self._frame_count = 0
        
        # Register camera callback
        self.on_camera_image(self.detect_line)
        self.get_logger().info("MyLineFollower initialized — ready to detect green line")

    def detect_line(self, image: np.ndarray) -> float | None:
        """
        Detect the green line and return steering command.
        
        Args:
            image: BGR image from camera, shape (720, 1280, 3)
        
        Returns:
            Steering value in [-1.0, 1.0], or None if line not detected.
        """
        _ = image
        test_steering = -0.75
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(f"TEST MODE steer={test_steering:.2f} frame={self._frame_count}")
        self.show_notification(f"TEST MODE steer={test_steering:.2f}")
        return test_steering


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
