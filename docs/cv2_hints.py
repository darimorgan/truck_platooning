#!/usr/bin/env python3
"""
OpenCV Tips for Line Following

Helpful snippets and explanations for computer vision operations.
"""
import cv2          # type: ignore
import numpy as np  # type: ignore


# ════════════════════════════════════════════════════════════════════════════
# 1. COLOR SPACES
# ════════════════════════════════════════════════════════════════════════════

# BGR vs RGB vs HSV
# ────────────────
#
# Images in OpenCV are BGR by default (Blue, Green, Red)
# NOT RGB like in matplotlib!
#
# For green line detection:
#
#   BGR color space (better for known colors):
#   lower_green = np.array([0, 100, 0])     # [B, G, R]
#   upper_green = np.array([100, 255, 100])
#
#   HSV color space (better for varying lighting):
#   lower_green = np.array([40, 40, 40])    # [H, S, V]
#   upper_green = np.array([90, 255, 255])
#
# To convert: cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ════════════════════════════════════════════════════════════════════════════
# 2. COLOR DETECTION
# ════════════════════════════════════════════════════════════════════════════

def simple_threshold(image, lower, upper):
    """Create binary mask for color range."""
    return cv2.inRange(image, lower, upper)


def adjust_threshold_values():
    """
    How to find the right threshold values:
    
    1. Print pixel values from your image:
        img = cv2.imread("sample_with_line.jpg")
        print( img[350, 640] )  # Print pixel at (640, 350)
        
    2. Add ~/-20 to upper/lower for margin
    
    3. Use a test script to verify
    """
    pass


# ════════════════════════════════════════════════════════════════════════════
# 3. MORPHOLOGICAL OPERATIONS (Noise Reduction)
# ════════════════════════════════════════════════════════════════════════════

def denoise_mask(mask):
    """Remove noise from binary mask."""
    
    # Create a rounded rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Closing: fill small holes inside the object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Opening: remove small noise blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


# Kernel sizes: (5, 5), (7, 7), (9, 9) — larger = more aggressive
# Try different sizes depending on line thickness and image noise


# ════════════════════════════════════════════════════════════════════════════
# 4. FINDING CONTOURS
# ════════════════════════════════════════════════════════════════════════════

def find_line_contour(mask):
    """Find the main line contour in a binary mask."""
    
    # Find all contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,        # Only outer contours (faster)
        cv2.CHAIN_APPROX_SIMPLE   # Simplify contours (less memory)
    )
    
    if not contours:
        return None
    
    # Get the largest contour (should be the line)
    largest = max(contours, key=cv2.contourArea)
    
    # Filter out really small contours (likely noise)
    if cv2.contourArea(largest) < 100:  # Adjust threshold as needed
        return None
    
    return largest


# ════════════════════════════════════════════════════════════════════════════
# 5. CALCULATING CENTROID (Center of Mass)
# ════════════════════════════════════════════════════════════════════════════

def get_contour_center(contour):
    """Calculate the (x, y) center of a contour."""
    
    M = cv2.moments(contour)  # Calculate moments
    
    if M["m00"] == 0:  # Avoid division by zero
        return None
    
    # Centroid coordinates
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    return (cx, cy)


# Explanation:
# - M["m10"] and M["m01"] are spatial moments
# - M["m00"] is the area
# - Centroid = (m10/m00, m01/m00)


# ════════════════════════════════════════════════════════════════════════════
# 6. STEERING CALCULATION
# ════════════════════════════════════════════════════════════════════════════

def offset_to_steering(line_x, image_width, sensitivity=0.5):
    """Convert horizontal line offset to steering value."""
    
    image_center = image_width / 2.0  # 640 for 1280-wide image
    
    # Normalized offset: -1 (far left) to +1 (far right)
    offset = (line_x - image_center) / image_center
    
    # Apply sensitivity multiplier and clamp to [-1, 1]
    steering = np.clip(offset * sensitivity, -1.0, 1.0)
    
    # Sensitivity values to try:
    # - 0.3: smooth, gradual corrections
    # - 0.5: moderate responsiveness
    # - 1.0: aggressive steering
    # - 2.0: very sharp turns
    
    return steering


# ════════════════════════════════════════════════════════════════════════════
# 7. DEBUGGING TIPS
# ════════════════════════════════════════════════════════════════════════════

def visualize_detection(image, mask, contour, center):
    """Create a debug visualization."""
    
    # Start with the binary mask (convert single-channel to 3-channel)
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw the contour in green
    cv2.drawContours(debug, [contour], 0, (0, 255, 0), 2)
    
    # Draw the center as a red circle
    cv2.circle(debug, tuple(map(int, center)), 5, (0, 0, 255), -1)
    
    # Draw image center line for reference
    w = image.shape[1]
    cv2.line(debug, (w//2, 0), (w//2, image.shape[0]), (255, 0, 0), 1)
    
    return debug


# ════════════════════════════════════════════════════════════════════════════
# 8. PERFORMANCE OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════

def optimize_by_roi(image, roi_fraction=0.5):
    """
    Process only the bottom half of the image (where the line is visible).
    Speeds up computation significantly.
    """
    h, w = image.shape[:2]
    
    # Crop to bottom 50% of image
    start_row = int(h * (1 - roi_fraction))
    roi = image[start_row:, :]
    
    # After processing, remember to adjust coordinates back!
    # detected_x stays the same
    # detected_y needs to add start_row
    
    return roi, start_row


# ════════════════════════════════════════════════════════════════════════════
# 9. TEMPLATE FUNCTION — Use This!
# ════════════════════════════════════════════════════════════════════════════

def my_line_detector(image):
    """
    Copy this template and fill in the TODO sections.
    """
    
    # TODO: Define color range for green line (BGR or HSV)
    lower_color = np.array([0, 100, 0])
    upper_color = np.array([100, 255, 100])
    
    # TODO: Create mask
    mask = cv2.inRange(image, lower_color, upper_color)
    
    # TODO: Denoise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # TODO: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # TODO: Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 100:
        return None
    
    # TODO: Get center
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    
    line_center_x = M["m10"] / M["m00"]
    
    # TODO: Calculate steering
    image_center_x = image.shape[1] / 2.0
    offset = (line_center_x - image_center_x) / image_center_x
    steering = np.clip(offset * 0.5, -1.0, 1.0)
    
    return steering
