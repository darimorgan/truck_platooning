#!/usr/bin/env python3
"""
LineFollowingInterface — base class for student line following nodes.

Students subclass this and override the callback method:
    detect_line(image)  — called for every camera frame (~30 fps)

The rest of the file is framework code that routes camera frames to your
function and sends steering commands to the simulator.

Quick-start example
-------------------
    class MyLineFollower(LineFollowingInterface):
        def __init__(self):
            super().__init__("my_line_follower")
            self.on_camera_image(self.process_frame)

Inputs
------
Camera image → detect_line(image)
  image       np.ndarray, shape (720, 1280, 3), BGR colour order
              Same convention as OpenCV.

Outputs
-------
detect_line(image) → float or None
  Return steering value in range [-1.0, 1.0]:
    -1.0 = full left turn
     0.0 = go straight (line detected as centered)
    +1.0 = full right turn
        None = cannot detect line, framework falls back to neutral steering

Sending control
---------------
    self.send_steering(steer=0.5)  # Turn right 50%
    # Framework also applies constant forward throttle while connected.
    self.show_notification("Line detected")

Other helpers
-------------
    self.current_image           np.ndarray or None  latest camera frame
    self.create_periodic_task(period_sec, callback)
"""
import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore
import os
from rclpy.node import Node  # type: ignore

try:
    from rclpy.qos import qos_profile_sensor_data  # type: ignore
except Exception:
    qos_profile_sensor_data = 10

from geometry_msgs.msg import Twist  # type: ignore
from sensor_msgs.msg import CompressedImage  # type: ignore
from std_msgs.msg import String  # type: ignore


class LineFollowingInterface(Node):
    """
    Base ROS 2 node for student line following systems.
    
    Wraps all ROS topics so students can focus on vision and steering logic
    without worrying about ROS middleware.
    """

    def __init__(self, node_name: str = "line_follower"):
        super().__init__(node_name)
        
        # Get role name from environment or use default
        self.role_name = self.declare_parameter(
            'role_name',
            'hero'
        ).get_parameter_value().string_value
        
        # ── Topic names ────────────────────────────────────────────────────
        prefix = f"/carla/{self.role_name}"
        camera_topic = f"{prefix}/camera/image/compressed"
        steering_topic = f"{prefix}/cmd_vel_ext"
        hud_topic = f"{prefix}/hud_event"
        
        # ── Publishers ─────────────────────────────────────────────────────
        self._pub_steering = self.create_publisher(Twist, steering_topic, 10)
        self._pub_hud = self.create_publisher(String, hud_topic, 10)
        
        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage,
            camera_topic,
            self._on_camera_msg,
            qos_profile=qos_profile_sensor_data
        )

        throttle_env = os.getenv("DEFAULT_THROTTLE", "0.35")
        try:
            self.default_throttle = max(0.0, min(1.0, float(throttle_env)))
        except ValueError:
            self.default_throttle = 0.35
        
        # ── State ────────────────────────────────────────────────────────
        self.current_image: np.ndarray | None = None
        self._camera_callback = None
        
        # ── Periodic tasks ────────────────────────────────────────────────
        self._timers = []
        
        self.get_logger().info(f"LineFollowingInterface initialized for role '{self.role_name}'")

    def on_camera_image(self, callback):
        """
        Register a callback to be called for each camera frame.
        
        Callback signature:
            callback(image: np.ndarray) → None
        """
        self._camera_callback = callback

    def _on_camera_msg(self, msg: CompressedImage):
        """Internal handler for camera messages."""
        # Decompress JPEG
        try:
            nparr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                self.current_image = image
                
                # Call user callback if registered
                if self._camera_callback:
                    steer_value = self._camera_callback(image)

                    # Always send a control command while active.
                    # If no line is detected, keep car moving with neutral steering.
                    if steer_value is None:
                        self.send_steering(0.0)
                    else:
                        self.send_steering(steer_value)
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")

    def send_steering(self, steer: float):
        """
        Send line-following control to the vehicle.
        
        Args:
            steer: [-1.0, 1.0] steering value
                   -1.0 = full left turn
                    0.0 = go straight
                   +1.0 = full right turn

        Notes:
            - Throttle is kept at a lab-default constant value.
            - Brake is released.
            - Steering comes from the student callback.
        """
        # Clamp to valid range
        steer = max(-1.0, min(1.0, steer))
        
        # Create control message:
        # linear.x = throttle, linear.y = brake, angular.z = steering
        msg = Twist()
        msg.linear.x = float(self.default_throttle)
        msg.linear.y = 0.0
        msg.angular.z = steer
        
        self._pub_steering.publish(msg)

    def show_notification(self, text: str, level: str = "info", duration: float = 3.0):
        """
        Show a notification on the HUD.
        
        Args:
            text: notification message
            level: "info", "warning", or "alert"
            duration: how long to show (seconds)
        """
        import json
        
        hud_msg = json.dumps({
            "text": text,
            "level": level,
            "duration": duration
        })
        
        msg = String()
        msg.data = hud_msg
        self._pub_hud.publish(msg)

    def show_warning(self, text: str, duration: float = 5.0):
        """Show a warning (yellow) on HUD."""
        self.show_notification(text, level="warning", duration=duration)

    def show_alert(self, text: str, duration: float = 5.0):
        """Show an alert (red) on HUD."""
        self.show_notification(text, level="alert", duration=duration)

    def create_periodic_task(self, period_sec: float, callback):
        """
        Create a task that runs periodically.
        
        Args:
            period_sec: period in seconds
            callback: function to call (takes no arguments)
        """
        def timer_callback():
            try:
                callback()
            except Exception as e:
                self.get_logger().error(f"Error in periodic task: {e}")
        
        timer = self.create_timer(period_sec, timer_callback)
        self._timers.append(timer)

    def destroy_node(self):
        """Clean up resources."""
        for timer in self._timers:
            timer.cancel()
        super().destroy_node()
