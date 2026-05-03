#!/usr/bin/env python3
"""Create a YOLO object-detection dataset from ALL cameras under an image root.

Recursively finds every .jpg beneath --image_root, assigns each image to
train / val / test via a deterministic hash split (default 80 / 10 / 10),
and creates symbolic links (never copies).

Symlink names are prefixed with the camera path components so filenames
remain unique across cameras:
    CAMERA_ARRAY_A__camera_A2__p_000001.jpg20180327.jpg

Optional ground-truth JSON files (same {image_path, boxes} format as
create_yolo_dataset.py) can be supplied via --labels.  Images without
an entry get an empty label file (background / no-detection images).

Usage
-----
    # No labels yet — image structure only:
    python create_yolo_dataset_all.py \\
        --image_root /data/JR_CAMERA_TRAPS/images \\
        --output ~/jrbp_ct_explorations/tinyissimoyolov8_cfgonly_binary/all_cameras_yolo

    # With label JSON files:
    python create_yolo_dataset_all.py \\
        --image_root /data/JR_CAMERA_TRAPS/images \\
        --output ~/jrbp_ct_explorations/tinyissimoyolov8_cfgonly_binary/all_cameras_yolo \\
        --labels ../speciesnet_labels_cleaned_binary/speciesnet_labels_a2_cleaned_boxes_*.json \\
                 ../speciesnet_labels_cleaned_binary/speciesnet_labels_b1_cleaned_boxes_*.json

    # Custom ratios:
    python create_yolo_dataset_all.py \\
        --image_root /data/JR_CAMERA_TRAPS/images \\
        --output ~/jrbp_ct_explorations/tinyissimoyolov8_cfgonly_binary/all_cameras_yolo \\
        --train_ratio 0.8 --val_ratio 0.1
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Split helpers ─────────────────────────────────────────────────────────────

def _hash_to_float(s: str, seed: int = 0) -> float:
    """Deterministic float in [0, 1) from a string (matches loader.py)."""
    import hashlib
    h = hashlib.md5((str(seed) + "|" + s).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 10_000_000) / 10_000_000.0


def assign_split(path: str, seed: int = 123,
                 train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """Return 'train', 'val', or 'test' for a given image path."""
    v = _hash_to_float(path, seed=seed)
    if v < train_ratio:
        return "train"
    if v < train_ratio + val_ratio:
        return "val"
    return "test"


# ── YOLO label conversion ─────────────────────────────────────────────────────

def boxes_to_yolo(boxes: list[list[float]]) -> str:
    """Convert [x, y, w, h] top-left normalized boxes to YOLO label text."""
    lines = []
    for x, y, w, h in boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + "\n" if lines else ""


# ── Dataset helpers ───────────────────────────────────────────────────────────

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


def camera_id(img_path: Path, image_root: Path) -> str:
    """Return '__'-joined relative path components above the image filename.

    E.g. CAMERA_ARRAY_A/camera_A2/p_...jpg  →  'CAMERA_ARRAY_A__camera_A2'
    """
    rel = img_path.relative_to(image_root)
    return "__".join(rel.parts[:-1])


def symlink_name(img_path: Path, image_root: Path) -> str:
    """Unique filename: camera prefix + original filename.

    E.g. 'CAMERA_ARRAY_A__camera_A2__p_000001.jpg20180327.jpg'
    """
    cam = camera_id(img_path, image_root)
    return f"{cam}__{img_path.name}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build YOLO dataset from all cameras under image_root",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/data/JR_CAMERA_TRAPS/images",
        help="Root directory containing all camera sub-folders",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[
            "/home/lanmei/jrbp_ct_explorations/speciesnet_labels_cleaned_binary/"
            "speciesnet_labels_full_cleaned_boxes_20260502_200512.json"
        ],
        metavar="JSON",
        help=(
            "Ground-truth JSON files ({image_path, boxes} list format). "
            "Multiple files accepted. Images without entries get empty label files."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset directory",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument(
        "--seed", type=int, default=123, help="Hash seed for split"
    )
    args = parser.parse_args()

    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output).expanduser().resolve()
    test_ratio  = 1.0 - args.train_ratio - args.val_ratio

    if not image_root.is_dir():
        print(f"Error: image_root not found: {image_root}")
        return 1
    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be < 1.0")
        return 1

    print("=" * 70)
    print("CTrap YOLO Dataset Builder — All Cameras")
    print("=" * 70)
    print(f"Image root   : {image_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Split        : train={args.train_ratio:.0%}  val={args.val_ratio:.0%}  test={test_ratio:.0%}")
    print(f"Hash seed    : {args.seed}")

    # ── Load labels ──────────────────────────────────────────────────────────
    detection_map: dict[str, list[list[float]]] = {}
    for label_file in args.labels:
        path = Path(label_file)
        if not path.exists():
            print(f"Warning: label file not found, skipping: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        count_before = len(detection_map)
        for r in records:
            if r.get("image_path"):
                detection_map[r["image_path"]] = r.get("boxes", [])
        print(f"  Loaded {len(detection_map) - count_before} entries from {path.name}")

    if detection_map:
        total_boxes = sum(len(v) for v in detection_map.values())
        images_with_boxes = sum(1 for v in detection_map.values() if v)
        print(f"  Total: {total_boxes} boxes across {images_with_boxes} / {len(detection_map)} images")
    else:
        print("  No label files provided — all images will have empty label files.")

    # ── Discover all images ──────────────────────────────────────────────────
    print("\nScanning for images (this may take a moment) …")
    all_images = sorted(image_root.rglob("*.jpg"))
    print(f"Found {len(all_images):,} images total.")

    # ── Assign splits + group by camera ──────────────────────────────────────
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    per_camera: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for img in all_images:
        sp = assign_split(str(img), seed=args.seed,
                          train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        splits[sp].append(img)
        cam = camera_id(img, image_root)
        per_camera[cam][sp] += 1

    # ── Create output structure ──────────────────────────────────────────────
    create_directory_structure(output_dir)
    write_config(output_dir)
    (output_dir / "classes.txt").write_text("0\tobject\n")

    # ── Create symlinks + label files ─────────────────────────────────────────
    print("\nCreating symlinks and label files …")
    label_counts = {"with_boxes": 0, "empty": 0}
    skipped_collisions = 0

    for split_name, paths in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"

        for i, src in enumerate(paths):
            if (i + 1) % 50_000 == 0:
                print(f"  [{split_name}] {i + 1:,} / {len(paths):,} …")

            link_name = symlink_name(src, image_root)
            dst_img   = img_dir / link_name
            stem      = Path(link_name).stem  # strip only last suffix (.jpg)
            dst_lbl   = lbl_dir / f"{stem}.txt"

            # Symlink
            if not (dst_img.exists() or dst_img.is_symlink()):
                dst_img.symlink_to(src)
            else:
                skipped_collisions += 1

            # Label
            if not dst_lbl.exists():
                boxes = detection_map.get(str(src), [])
                dst_lbl.write_text(boxes_to_yolo(boxes))
                if boxes:
                    label_counts["with_boxes"] += 1
                else:
                    label_counts["empty"] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\n{'Split':<8}  {'Images':>8}  {'%':>6}")
    print("-" * 28)
    total = len(all_images)
    for sp in ("train", "val", "test"):
        n = len(splits[sp])
        print(f"  {sp:<6}  {n:>8,}  {n/total:>5.1%}")
    print(f"  {'TOTAL':<6}  {total:>8,}  {'100.0%':>6}")

    print(f"\n{'Camera':<45}  {'train':>7}  {'val':>7}  {'test':>7}  {'total':>7}")
    print("-" * 72)
    for cam in sorted(per_camera):
        c = per_camera[cam]
        t = c["train"] + c["val"] + c["test"]
        print(f"  {cam:<43}  {c['train']:>7,}  {c['val']:>7,}  {c['test']:>7,}  {t:>7,}")

    print(f"\nLabel files :")
    print(f"  With boxes : {label_counts['with_boxes']:,}")
    print(f"  Empty      : {label_counts['empty']:,}")
    if skipped_collisions:
        print(f"  Skipped (already exist): {skipped_collisions}")

    print(f"\nOutput      : {output_dir}")
    print(f"config.yaml : {output_dir / 'config.yaml'}")
    print("\n" + "=" * 70)
    print("Dataset creation complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
