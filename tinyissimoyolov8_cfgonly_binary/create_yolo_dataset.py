#!/usr/bin/env python3
"""Create a YOLO object-detection dataset from camera_B1 images.

Images are sourced via symbolic links (no copies created).
Bounding-box labels are loaded from a ground-truth JSON file produced by
speciesnet_labels_b1_cleaned_boxes_*.json (list of
{image_path, boxes: [[x, y, w, h], ...]} records, normalized top-left coords).

Output layout
-------------
<output_dir>/
    train/
        images/   <- symlinks to originals
        labels/   <- YOLO .txt files  (class cx cy w h, normalized)
    val/
        images/
        labels/
    test/
        images/
        labels/
    config.yaml
    classes.txt

Split rules
-----------
* Images whose stem starts with "p_000"  ->  test
* Remaining images split train / val via _hash_to_float (seed=123, ratio=0.8)

Usage
-----
    CUDA_VISIBLE_DEVICES=1
    python create_yolo_dataset.py \
    --image_dir /data/JR_CAMERA_TRAPS/images/CAMERA_ARRAY_B/camera_B1 \
    --labels ../speciesnet_labels_cleaned_binary/speciesnet_labels_b1_cleaned_boxes_20260413_143653.json \
    --output ~/jrbp_ct_explorations/tinyissimoyolov8_cfgonly_binary/camera_B1_yolo_25inputs \
    --train_ratio 0.004
        
    #train_ratio=0.004 - 25 images

    CUDA_VISIBLE_DEVICES=1
    python create_yolo_dataset.py \
    --image_dir /data/JR_CAMERA_TRAPS/images/CAMERA_ARRAY_A/camera_A2 \
    --labels ../speciesnet_labels_cleaned_binary/speciesnet_labels_a2_cleaned_boxes_20260414_183450.json \
    --output ~/jrbp_ct_explorations/tinyissimoyolov8_cfgonly_binary/camera_A2_yolo \
    --train_ratio 0.8
   
   CUDA_VISIBLE_DEVICES=1
    python create_yolo_dataset.py \
    --image_dir /data/JR_CAMERA_TRAPS/images/CAMERA_ARRAY_B/camera_B2 \
    --labels ../speciesnet_labels_cleaned_binary/speciesnet_labels_b2_cleaned_boxes_20260415_213737.json \
    --output ~/jrbp_ct_explorations/tinyissimoyolov8_cfgonly_binary/camera_B2_yolo \
    --train_ratio 0.1  #test: ("p_0004", "p_0005")

"""

import argparse
import json
import sys
from pathlib import Path


# ── Splitting helpers (mirrors ctrap_roi/data/loader.py) ─────────────────────

def _hash_to_float(s: str, seed: int = 0) -> float:
    """Deterministic float in [0, 1) from a string, matching loader.py."""
    import hashlib
    h = hashlib.md5((str(seed) + "|" + s).encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return (v % 10_000_000) / 10_000_000.0


def assign_split(path: str, seed: int = 123, train_ratio: float = 0.8) -> str:
    """Return 'test', 'train', or 'val' for a given image path."""
    stem = Path(path).name  # use filename so hash is path-independent
    if stem.startswith(("p_0004", "p_0005")): # "p_000", 
        return "test"
    val = _hash_to_float(path, seed=seed)
    return "train" if val < train_ratio else "val"


# ── YOLO label conversion ─────────────────────────────────────────────────────

def boxes_to_yolo(boxes: list[list[float]]) -> list[str]:
    """Convert [x, y, w, h] top-left normalized boxes to YOLO label lines.

    YOLO format: <class_id> <cx> <cy> <w> <h>  (all normalized, center-based)
    """
    lines = []
    for x, y, w, h in boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ── Dataset creation ──────────────────────────────────────────────────────────

def create_directory_structure(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def write_config(output_dir: Path) -> None:
    config = (
        f"path: {output_dir.resolve()}  # absolute path to dataset\n"
        "test: test/images\n"
        "train: train/images\n"
        "val: val/images\n"
        "\n"
        "# classes\n"
        "names:\n"
        "  0: object\n"
        "nc: 1\n"
    )
    (output_dir / "config.yaml").write_text(config)


def write_classes(output_dir: Path) -> None:
    (output_dir / "classes.txt").write_text("0\tobject\n")


def create_symlink(src: Path, dst: Path) -> None:
    """Create symlink dst -> src, skipping if dst already exists."""
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def write_label(label_path: Path, lines: list[str]) -> None:
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build YOLO dataset from camera_B1 with symlinked images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/JR_CAMERA_TRAPS/images/CAMERA_ARRAY_B/camera_B1",
        help="Directory containing source .jpg images",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to ground-truth JSON file (list of {image_path, boxes} records)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "camera_B1_yolo"),
        help="Output dataset directory",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of non-test images assigned to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Hash seed for train/val split",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output).expanduser().resolve()
    labels_path = Path(args.labels)

    if not image_dir.is_dir():
        print(f"Error: image_dir not found: {image_dir}")
        return 1

    if not labels_path.exists():
        print(f"Error: labels JSON not found: {labels_path}")
        return 1

    # ── Collect images ────────────────────────────────────────────────────────
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        print(f"Error: no .jpg files found in {image_dir}")
        return 1

    print("=" * 60)
    print("CTrap YOLO Dataset Builder")
    print("=" * 60)
    print(f"\nSource dir  : {image_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Total images: {len(image_paths)}")

    # ── Assign splits ─────────────────────────────────────────────────────────
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for p in image_paths:
        splits[assign_split(str(p), seed=args.seed, train_ratio=args.train_ratio)].append(p)

    print(f"  train : {len(splits['train'])}")
    print(f"  val   : {len(splits['val'])}")
    print(f"  test  : {len(splits['test'])}")

    # ── Create directories + config ───────────────────────────────────────────
    create_directory_structure(output_dir)
    write_config(output_dir)
    write_classes(output_dir)
    print(f"\nCreated directory structure under {output_dir}")

    # ── Load ground-truth boxes from JSON ────────────────────────────────────
    print(f"\nLoading ground-truth labels from {labels_path} ...")
    with open(labels_path, encoding="utf-8") as f:
        label_records = json.load(f)

    detection_map: dict[str, list[list[float]]] = {
        r["image_path"]: r.get("boxes", [])
        for r in label_records
        if r.get("image_path")
    }

    total_boxes = sum(len(v) for v in detection_map.values())
    images_with_boxes = sum(1 for v in detection_map.values() if v)
    print(f"  Loaded {total_boxes} boxes for {images_with_boxes} / "
          f"{len(detection_map)} images")

    # ── Create symlinks + write label files ───────────────────────────────────
    print("\nCreating symlinks and writing labels ...")
    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, paths in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"

        for src in paths:
            stem = src.stem  # filename without extension, e.g. p_007949.jpg20230402 -> strip only last .jpg
            # Path.stem strips exactly one suffix, so for "p_000001.jpg20180327.jpg"
            # stem = "p_000001.jpg20180327" — use that as the label stem too.
            dst_img = img_dir / src.name
            dst_lbl = lbl_dir / f"{stem}.txt"

            create_symlink(src, dst_img)

            boxes = detection_map.get(str(src), [])
            write_label(dst_lbl, boxes_to_yolo(boxes))

        counts[split_name] = len(paths)

    print(f"  train : {counts['train']} images")
    print(f"  val   : {counts['val']} images")
    print(f"  test  : {counts['test']} images")

    print(f"\nconfig.yaml : {output_dir / 'config.yaml'}")
    print(f"classes.txt : {output_dir / 'classes.txt'}")

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
