#!/usr/bin/env python3
"""Deterministic green-line steering controller."""
from __future__ import annotations

from dataclasses import dataclass

import cv2  # type: ignore
import numpy as np


@dataclass(frozen=True)
class LineFollowerConfig:
    lower_hsv: tuple[int, int, int] = (50, 110, 90)
    upper_hsv: tuple[int, int, int] = (70, 255, 220)
    kernel_size: int = 5
    min_contour_area: float = 100.0
    edge_margin_fraction: float = 0.08
    target_center_x: float = 700.0
    steering_gain: float = 0.5
    steering_deadband: float = 0.02


@dataclass(frozen=True)
class LineDetection:
    steering: float | None
    line_center_x: float | None
    mask: np.ndarray
    contours: list[np.ndarray]
    selected_contour: np.ndarray | None
    status: str


def create_green_mask(image_bgr: np.ndarray, config: LineFollowerConfig) -> np.ndarray:
    """Return a denoised binary mask for the configured green route-line color."""
    lower_hsv = np.array(config.lower_hsv, dtype=np.uint8)
    upper_hsv = np.array(config.upper_hsv, dtype=np.uint8)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.kernel_size, config.kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def detect_green_line(image_bgr: np.ndarray, config: LineFollowerConfig) -> LineDetection:
    """Detect the green route line and compute deterministic steering."""
    mask = create_green_mask(image_bgr, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return LineDetection(None, None, mask, contours, None, "no_contours")

    _, w = image_bgr.shape[:2]
    edge_margin_x = config.edge_margin_fraction * w
    candidates: list[tuple[float, np.ndarray, float]] = []
    has_large_contour = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_contour_area:
            continue

        has_large_contour = True
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = float(moments["m10"] / moments["m00"])
        if center_x < edge_margin_x or center_x > w - edge_margin_x:
            continue

        distance = abs(center_x - config.target_center_x)
        candidates.append((distance, contour, center_x))

    if not has_large_contour:
        return LineDetection(None, None, mask, contours, None, "contour_too_small")

    if not candidates:
        return LineDetection(None, None, mask, contours, None, "center_near_edge")

    _, selected_contour, line_center_x = min(candidates, key=lambda item: item[0])
    offset = (line_center_x - config.target_center_x) / (w / 2.0)
    steering = float(np.clip(offset * config.steering_gain, -1.0, 1.0))

    if abs(steering) < config.steering_deadband:
        steering = 0.0

    return LineDetection(steering, line_center_x, mask, contours, selected_contour, "ok")
