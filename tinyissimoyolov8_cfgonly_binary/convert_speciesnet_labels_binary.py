#!/usr/bin/env python3
"""Convert SpeciesNet label files to the batch_results bounding-box format.

Reads a cleaned SpeciesNet label JSON (nested dict keyed by camera array and
image filename), filters out detections whose top-scoring category is blank
("f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"), and writes the surviving
bounding boxes in the same format produced by predict_batch.py:

    [
      {
        "image_path": "/data/.../camera_B1/p_000001.jpg...",
        "image_index": 0,
        "num_detections": 2,
        "boxes": [[x, y, w, h], ...]
      },
      ...
    ]

Usage:
    python convert_speciesnet_labels_binary.py \
        --input  ../speciesnet_labels_cleaned/speciesnet_labels_b1_cleaned.json \
        --output ../speciesnet_labels_cleaned_binary/                     \
        --image_root /data/JR_CAMERA_TRAPS/images

    python convert_speciesnet_labels_binary.py \
        --input  ../speciesnet_labels_cleaned/speciesnet_labels_a2_cleaned.json \
        --output ../speciesnet_labels_cleaned_binary/  \
        --image_root /data/JR_CAMERA_TRAPS/images

    python convert_speciesnet_labels_binary.py \
        --input  ../speciesnet_labels_cleaned/speciesnet_labels_b2_cleaned.json \
        --output ../speciesnet_labels_cleaned_binary/  \
        --image_root /data/JR_CAMERA_TRAPS/images

    python convert_speciesnet_labels_binary.py \
        --input  ../speciesnet_labels_cleaned/speciesnet_labels_full_cleaned.json \
        --output ../speciesnet_labels_cleaned_binary/  \
        --image_root /data/JR_CAMERA_TRAPS/images
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BLANK_CATEGORY = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"


def is_blank(detection: dict) -> bool:
    """Return True when the highest-scoring category for a detection is blank."""
    categories = detection.get("categories", [])
    scores = detection.get("scores", [])
    if not categories or not scores:
        return False
    max_idx = scores.index(max(scores))
    return categories[max_idx] == BLANK_CATEGORY


def convert(labels: dict, image_root: str) -> list[dict]:
    """Convert the nested label dict to a flat list of per-image result dicts."""
    results: list[dict] = []
    image_index = 0

    for array_name, images in labels.items():
        for rel_path, detections in images.items():
            image_path = f"{image_root.rstrip('/')}/{array_name}/{rel_path}"

            valid_boxes = [
                d["bbox"]
                for d in detections
                if not is_blank(d)
            ]

            results.append(
                {
                    "image_path": image_path,
                    "image_index": image_index,
                    "num_detections": len(valid_boxes),
                    "boxes": valid_boxes,
                }
            )
            image_index += 1

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert SpeciesNet labels to batch_results bounding-box format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the cleaned SpeciesNet label JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="speciesnet_labels_cleaned_converted",
        help="Output directory for the converted JSON file",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/data/JR_CAMERA_TRAPS/images",
        help="Root path prepended to array name and relative image path",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    with open(input_path, encoding="utf-8") as f:
        labels = json.load(f)

    results = convert(labels, args.image_root)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem  # e.g. speciesnet_labels_b1_cleaned
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{stem}_boxes_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_images = len(results)
    images_with_detections = sum(1 for r in results if r["num_detections"] > 0)
    total_boxes = sum(r["num_detections"] for r in results)
    discarded = sum(
        1
        for array_images in labels.values()
        for detections in array_images.values()
        for d in detections
        if is_blank(d)
    )

    print(f"Input          : {input_path}")
    print(f"Output         : {output_path}")
    print(f"Total images   : {total_images}")
    print(f"Images w/ ROIs : {images_with_detections}")
    print(f"Total boxes    : {total_boxes}")
    print(f"Blank discarded: {discarded}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
