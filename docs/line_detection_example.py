#!/usr/bin/env python3
"""
Line Detection Example — Complete working implementation

This is a demonstration of how to detect a green line and calculate steering.
You can use this as a starting point for your implementation.
"""
import cv2          # type: ignore
import numpy as np  # type: ignore


def detect_green_line_simple(image: np.ndarray) -> float | None:
    """
    Simple line detection: color threshold → contours → steering.
    
    Args:
        image: BGR image from camera, shape (720, 1280, 3)
    
    Returns:
        Steering value [-1.0, 1.0] or None if line not detected
    """
    
    # Step 1: Create green color mask
    # ────────────────────────────────────
    lower_green = np.array([0, 100, 0])      # Low: [B, G, R]
    upper_green = np.array([100, 255, 100])  # High: [B, G, R]
    
    mask = cv2.inRange(image, lower_green, upper_green)
    
    
    # Step 2: Clean up the mask (reduce noise)
    # ────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    
    # Step 3: Find contours (connected regions)
    # ────────────────────────────────────────
    contours, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL,           # Only outer contours
        cv2.CHAIN_APPROX_SIMPLE      # Simplify contour
    )
    
    if not contours:
        # No green pixels found → no line
        return None
    
    
    # Step 4: Find the largest contour (should be the line)
    # ─────────────────────────────────────────────────────
    largest_contour = max(contours, key=cv2.contourArea)
    
    # If too small, it's probably noise
    if cv2.contourArea(largest_contour) < 100:
        return None
    
    
    # Step 5: Calculate centroid of the contour
    # ────────────────────────────────────────
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None
    
    line_center_x = M["m10"] / M["m00"]  # X coordinate of centroid
    line_center_y = M["m01"] / M["m00"]  # Y coordinate of centroid (not used for steering)
    
    
    # Step 6: Calculate steering based on line position
    # ──────────────────────────────────────────────────
    h, w = image.shape[:2]  # h=720, w=1280
    image_center_x = w / 2.0  # 640
    
    # How far is the line from center? (normalized)
    offset = (line_center_x - image_center_x) / image_center_x
    # offset in [-1, 1]: -1 means far left, +1 means far right
    
    # Convert offset to steering value
    # Multiplier controls sensitivity (adjust to tune)
    steering = np.clip(offset * 0.5, -1.0, 1.0)
    
    return steering


def detect_green_line_advanced(image: np.ndarray, show_debug: bool = False) -> float | None:
    """
    More robust implementation with:
    - HSV color space (better for varying lighting)
    - Fitted line for better center estimation
    - Multiple contour check
    
    Args:
        image: BGR image from camera
        show_debug: if True, returns (steering, debug_image) instead
    
    Returns:
        Steering value or (steering, debug_image) if show_debug=True
    """
    
    # Convert to HSV (more robust to lighting)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green in HSV: H ≈ 60-90 (roughly)
    # Adjust these ranges based on your lighting conditions
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Dilate to connect nearby points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 100:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a line to the contour
    rows, cols = mask.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Line equation: (Y - y) / vy = (X - x) / vx
    # At the center row, what's the X position?
    # X = x + (center_row - y) * (vx / vy)
    
    center_row = rows // 2
    if abs(vy) > 0.01:  # Avoid division by zero
        line_x_at_center = x + (center_row - y) * (vx / vy)
    else:
        # Line is nearly horizontal, use contour centroid
        M = cv2.moments(largest_contour)
        line_x_at_center = M["m10"] / M["m00"] if M["m00"] > 0 else cols // 2
    
    # Calculate steering
    image_center_x = cols / 2.0
    offset = (line_x_at_center - image_center_x) / image_center_x
    steering = np.clip(offset * 0.5, -1.0, 1.0)
    
    if show_debug:
        # Create visualization
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug, [largest_contour], 0, (0, 255, 0), 2)
        cv2.circle(debug, (int(line_x_at_center), center_row), 5, (0, 0, 255), -1)
        cv2.line(debug, (image_center_x, 0), (image_center_x, rows), (255, 0, 0), 1)
        
        return steering, debug
    
    return steering


# ──────────────────────────────────────────────────────────────────────────
# Example usage in a test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a test image with a green line
    test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw a green line slightly to the right
    cv2.line(test_image, (700, 720), (700, 0), (0, 255, 0), thickness=10)
    
    # Test simple version
    steering_simple = detect_green_line_simple(test_image)
    print(f"Simple method: steering = {steering_simple:.3f}")
    
    # Test advanced version with debug
    steering_adv, debug_img = detect_green_line_advanced(test_image, show_debug=True)
    print(f"Advanced method: steering = {steering_adv:.3f}")
    
    # You could save the debug image
    # cv2.imwrite("debug_line_detection.png", debug_img)
