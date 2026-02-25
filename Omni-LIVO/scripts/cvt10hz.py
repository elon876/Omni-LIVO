#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Frame Rate Converter for Hilti SLAM Challenge Dataset
==============================================================

This script converts Hilti dataset camera images from 40Hz to 10Hz to match
the LiDAR frequency required by Omni-LIVO.

Background:
-----------
Hilti SLAM Challenge datasets have cameras running at 40Hz while the LiDAR
operates at 10Hz. Omni-LIVO requires synchronized multi-sensor data at 10Hz.
This script downsamples camera topics while maintaining temporal synchronization
across all cameras.

Usage:
------
1. Convert all bag files in a directory:
   python cvt10hz.py /path/to/hilti/dataset/

2. Convert specific bag file:
   python cvt10hz.py /path/to/hilti.bag

3. The script will generate output files with "_10hz.bag" suffix

Camera Topics (Hilti 2022/2023):
---------------------------------
- /alphasense_driver_ros/cam0/compressed (Front-left)
- /alphasense_driver_ros/cam1/compressed (Front-right)
- /alphasense_driver_ros/cam3/compressed (Rear-left)
- /alphasense_driver_ros/cam4/compressed (Rear-right)

Output:
-------
- Original: hilti_sequence.bag (40Hz cameras)
- Output: hilti_sequence_10hz.bag (10Hz cameras, synchronized)

Features:
---------
- Multi-process parallel conversion for multiple bag files
- Frame synchronization across all cameras
- Automatic timestamp matching
- Progress monitoring

Example:
--------
# Convert all bags in Hilti 2022 dataset
python cvt10hz.py ~/datasets/hilti2022/

# This will create:
#   exp01_construction_upper_level_10hz.bag
#   exp02_construction_basement_10hz.bag
#   ...

Requirements:
-------------
- Python 3.x
- rosbag
- ROS environment

Author: Omni-LIVO Team
Copyright (c) 2026 Hangzhou Institute for Advanced Study, UCAS
"""

import os
import sys
import rosbag
from concurrent.futures import ProcessPoolExecutor

# Hilti SLAM Challenge camera topics (4 cameras)
TARGET_TOPICS = [
    "/alphasense_driver_ros/cam0/compressed",  # Front-left
    "/alphasense_driver_ros/cam1/compressed",  # Front-right
    "/alphasense_driver_ros/cam3/compressed",  # Rear-left
    "/alphasense_driver_ros/cam4/compressed",  # Rear-right
]

# Downsampling factor: 40Hz → 10Hz
FACTOR = 4


def process_bag(input_bag_path):
    """
    Convert a single bag file from 40Hz to 10Hz

    Args:
        input_bag_path: Path to input bag file

    Returns:
        output_bag_path: Path to generated output bag file
    """
    output_bag_path = input_bag_path.replace(".bag", "_10hz.bag")

    print(f"\n{'='*60}")
    print(f"[PID {os.getpid()}] Processing: {os.path.basename(input_bag_path)}")
    print(f"[PID {os.getpid()}] Output: {os.path.basename(output_bag_path)}")
    print(f"{'='*60}")

    bag_in = rosbag.Bag(input_bag_path, 'r')
    bag_out = rosbag.Bag(output_bag_path, 'w')

    # Buffer for frame synchronization: {topic: (msg, timestamp)}
    buffer = {topic: None for topic in TARGET_TOPICS}

    synced_frame_count = 0  # Counter for synchronized 40Hz frames
    written_frame_count = 0  # Counter for written 10Hz frames

    try:
        for topic, msg, t in bag_in.read_messages(raw=False):

            # Write non-camera topics directly (LiDAR, IMU, etc.)
            if topic not in TARGET_TOPICS:
                bag_out.write(topic, msg, t)
                continue

            # Update/overwrite buffer for this camera
            buffer[topic] = (msg, t)

            # Wait until all cameras have data
            if not all(buffer.values()):
                continue

            # Get timestamps from all cameras
            stamps = [buffer[k][0].header.stamp for k in TARGET_TOPICS]

            # Check if all timestamps are identical (synchronized)
            if all(stamps[i] == stamps[0] for i in range(1, len(stamps))):
                synced_frame_count += 1

                # Downsample: keep every 4th frame (40Hz → 10Hz)
                if synced_frame_count % FACTOR == 1:
                    for k in TARGET_TOPICS:
                        msg_k, t_k = buffer[k]
                        bag_out.write(k, msg_k, t_k)
                    written_frame_count += 1

                # Clear buffer for next synchronized frame
                buffer = {topic: None for topic in TARGET_TOPICS}
                continue

            # Handle unsynchronized frames: discard earliest
            ts_ns = [s.to_nsec() for s in stamps]
            earliest_ns = min(ts_ns)

            # Remove frames with earliest timestamp
            for idx, topic_k in enumerate(TARGET_TOPICS):
                if ts_ns[idx] == earliest_ns:
                    buffer[topic_k] = None

    finally:
        bag_in.close()
        bag_out.close()

    print(f"\n[PID {os.getpid()}] Conversion Summary:")
    print(f"  - Synchronized frames (40Hz): {synced_frame_count}")
    print(f"  - Written frames (10Hz): {written_frame_count}")
    print(f"  - Output file: {output_bag_path}")
    print(f"{'='*60}\n")

    return output_bag_path


def main():
    """
    Main function: process single file or all bag files in directory
    """
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide a bag file or directory path")
        print("\nUsage:")
        print("  python cvt10hz.py /path/to/directory/")
        print("  python cvt10hz.py /path/to/file.bag")
        sys.exit(1)

    input_path = sys.argv[1]

    # Determine if input is file or directory
    if os.path.isfile(input_path):
        # Single file
        if not input_path.endswith(".bag"):
            print(f"Error: {input_path} is not a .bag file")
            sys.exit(1)
        bag_files = [input_path]
    elif os.path.isdir(input_path):
        # Directory: find all .bag files
        bag_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".bag") and not f.endswith("_10hz.bag")
        ]
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    if not bag_files:
        print("No .bag files found.")
        return

    # Print processing information
    print("\n" + "="*60)
    print("Hilti Dataset Camera Frame Rate Converter")
    print("="*60)
    print(f"Found {len(bag_files)} bag file(s) to process:")
    for bag in bag_files:
        print(f"  - {os.path.basename(bag)}")

    # Use parallel processing for multiple files
    workers = min(len(bag_files), os.cpu_count() or 1)
    print(f"\nUsing {workers} parallel worker(s)")
    print("="*60)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_bag, bag) for bag in bag_files]
        results = [f.result() for f in futures]

    print("\n" + "="*60)
    print("All processing completed!")
    print("="*60)
    print(f"Generated {len(results)} output file(s):")
    for result in results:
        print(f"  ✓ {os.path.basename(result)}")
    print("\nYou can now use these bag files with Omni-LIVO:")
    print("  roslaunch omni_livo mapping_Hilti2022.launch")
    print("  rosbag play <output_file>_10hz.bag")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
