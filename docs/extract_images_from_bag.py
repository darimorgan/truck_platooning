#!/usr/bin/env python3
"""
Extract camera images from a ROS 2 bag file.

Usage
-----
    python3 docs/extract_images_from_bag.py ./bags/<session_folder>
    python3 docs/extract_images_from_bag.py ./bags/<session_folder> --out ./frames --every 5

Arguments
---------
    bag_path        Path to the bag session folder (contains metadata.yaml + .db3)
    --out           Output directory for images  [default: ./extracted_frames]
    --every N       Save every N-th frame          [default: 1  (all frames)]
    --topic         Camera topic to read           [default: /carla/hero/camera/image/compressed]
    --format        Image format: jpg or png        [default: jpg]

Output
------
    <out>/frame_000001.jpg
    <out>/frame_000002.jpg
    ...

Requirements
------------
    pip install rosbags opencv-python numpy
    (rosbags is a pure-Python bag reader — no ROS installation needed)
"""

import argparse
import sys
from pathlib import Path

import cv2          # type: ignore
import numpy as np  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract camera images from a ROS 2 bag."
    )
    p.add_argument("bag_path", help="Path to the bag session folder")
    p.add_argument("--out", default="extracted_frames",
                   help="Output directory (default: ./extracted_frames)")
    p.add_argument("--every", type=int, default=1, metavar="N",
                   help="Save every N-th frame (default: 1 = all)")
    p.add_argument("--topic", default="/carla/hero/camera/image/compressed",
                   help="ROS 2 topic to read")
    p.add_argument("--format", choices=["jpg", "png"], default="jpg",
                   help="Output image format (default: jpg)")
    return p.parse_args()


def decode_compressed_image(data: bytes) -> np.ndarray:
    """Decode a CompressedImage payload (JPEG/PNG bytes) to a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — unsupported format?")
    return img


def main() -> None:
    args = parse_args()

    # ── Import rosbags (pure Python, no ROS needed) ────────────────────────
    try:
        from rosbags.rosbag2 import Reader          # type: ignore
        from rosbags.typesys import Stores, get_typestore  # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: 'rosbags' package not found.\n"
            "Install it with:  pip install rosbags"
        )

    bag_path = Path(args.bag_path)
    if not bag_path.exists():
        sys.exit(f"ERROR: Bag path does not exist: {bag_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    saved = 0
    total = 0

    print(f"Opening bag:  {bag_path}")
    print(f"Topic:        {args.topic}")
    print(f"Output dir:   {out_dir}")
    print(f"Every N-th:   {args.every}")
    print()

    with Reader(bag_path) as reader:
        # Verify the topic exists in this bag
        available = [c.topic for c in reader.connections]
        if args.topic not in available:
            print(f"WARNING: Topic '{args.topic}' not found in bag.")
            print("Available topics:")
            for t in sorted(available):
                print(f"  {t}")
            sys.exit(1)

        connections = [c for c in reader.connections if c.topic == args.topic]

        for _connection, _timestamp, rawdata in reader.messages(connections=connections):
            total += 1

            if (total - 1) % args.every != 0:
                continue

            msg = typestore.deserialize_cdr(rawdata, _connection.msgtype)

            # msg.data is the raw compressed bytes
            try:
                img = decode_compressed_image(bytes(msg.data))
            except Exception as e:
                print(f"  [frame {total:06d}] decode error: {e} — skipping")
                continue

            filename = out_dir / f"frame_{saved + 1:06d}.{args.format}"
            cv2.imwrite(str(filename), img)
            saved += 1

            if saved % 50 == 0:
                print(f"  Saved {saved} frames so far...")

    print(f"\nDone. Saved {saved} / {total} frames → {out_dir}/")


if __name__ == "__main__":
    main()
