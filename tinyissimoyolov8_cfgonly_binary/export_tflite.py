"""export_tflite.py

Export a trained YOLO .pt model to TFLite (fp32 / fp16 / int8) and
evaluate each format on a dataset split.

TFLite runs on CPU (no GPU required).  INT8 calibration uses the
training split of the supplied YAML.

Usage
-----
  # INT8 only (default):
  python export_tflite.py \\
      --weights runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt \\
      --data    camera_A2_yolo/config.yaml \\
      --imgsz   256

  # All three TFLite formats:
  python export_tflite.py \\
      --weights runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt \\
      --data    camera_A2_yolo/config.yaml \\
      --formats fp32 fp16 int8 \\
      --imgsz   256

 python export_tflite.py \
    --weights runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt \
    --data    camera_A2_yolo/config.yaml \
    --formats int8 \
    --imgsz   256 \
    --calib_split    train \
    --calib_fraction 0.02       # use only 20% of val images for calibration

"""

import argparse
import json
import os
import sys
from pathlib import Path

# Suppress TF spam before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Tee stdout/stderr to a log file if --log_file is given
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--log_file", type=str, default=None)
_pre_early, _ = _pre.parse_known_args()

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self._streams: s.flush()
    def fileno(self):
        return self._streams[0].fileno()

if _pre_early.log_file:
    _log_path = Path(_pre_early.log_file)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

from ultralytics import YOLO
import torch; torch.cuda.init()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLO TFLite export + eval")
    p.add_argument(
        "--weights",
        type=str,
        default="runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt",
    )
    p.add_argument(
        "--data",
        type=str,
        default="camera_A2_yolo/config.yaml",
        help="Dataset YAML (used for calibration and eval)",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["int8"],
        choices=["fp32", "fp16", "int8", "full_int8"],
        help=(
            "TFLite formats to export and evaluate. "
            "'int8' = dynamic-range quant (float32 I/O, renamed from _dynamic_range_quant). "
            "'full_int8' = full integer quant with INT8 I/O — use this for MCU/edge deployment."
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Image size for export and eval (default: 256 for edge deployment)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for eval (TFLite is typically batch=1)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for val (default: 0.001, ultralytics standard for mAP)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS during val (default: 0.7, ultralytics standard)",
    )
    p.add_argument(
        "--calib_split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split used for INT8 calibration (default: val, ultralytics default)",
    )
    p.add_argument(
        "--calib_fraction",
        type=float,
        default=0.1,
        help="Fraction of calib_split images to use for INT8 calibration, e.g. 0.1 for 10% (default: 1.0 = all)",
    )
    p.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Ignore existing checkpoint and re-run all formats",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

# Maps format name → (export kwargs, tflite filename suffix)
# NOTE on int8 variants produced by onnx2tf (all generated in one export pass):
#   int8          → renamed from _dynamic_range_quant  — weight-only int8, float32 I/O
#   full_int8     → _full_integer_quant                — full int8 weights+activations, INT8 I/O
#                   ↑ use this for MCU/edge deployment (what tools mean by "quantized model")
_FORMAT_SPEC = {
    "fp32":      (dict(),          "float32.tflite"),
    "fp16":      (dict(half=True), "float16.tflite"),
    "int8":      (dict(int8=True), "int8.tflite"),
    "full_int8": (dict(int8=True), "full_integer_quant.tflite"),  # int8 I/O; same export pass as int8
}


def _saved_model_dir(weights_path: Path, imgsz: int) -> Path:
    """Return (and create if needed) the per-imgsz saved_model directory."""
    stem = weights_path.stem
    d = weights_path.parent / f"{stem}_saved_model_tflite_{imgsz}"
    d.mkdir(exist_ok=True)
    return d


def _export(weights_path: Path, fmt: str, args) -> Path:
    """Export to TFLite if not already present; return path to .tflite file."""
    stem = weights_path.stem
    smd = _saved_model_dir(weights_path, args.imgsz)
    suffix = _FORMAT_SPEC[fmt][1]
    tflite_path = smd / f"{stem}_{suffix}"

    if tflite_path.exists():
        print(f"[{fmt}] Reusing existing: {tflite_path}")
        return tflite_path

    print(f"[{fmt}] Exporting TFLite {fmt.upper()} …")
    extra_kwargs = _FORMAT_SPEC[fmt][0].copy()
    if extra_kwargs.get("int8"):
        extra_kwargs["data"]     = args.data
        extra_kwargs["split"]    = args.calib_split
        extra_kwargs["fraction"] = args.calib_fraction

    src = YOLO(str(weights_path))
    exported_dir = src.export(
        format="tflite",
        imgsz=args.imgsz,
        **extra_kwargs,
    )

    # Ultralytics creates {stem}_saved_model/ — rename to our per-imgsz dir if it's different
    ul_dir = weights_path.parent / f"{stem}_saved_model"
    if ul_dir.is_dir() and ul_dir != smd:
        import shutil
        if smd.exists():
            shutil.rmtree(smd)
        ul_dir.rename(smd)

    if not tflite_path.exists():
        candidates = list(smd.glob(f"*{suffix}"))
        if candidates:
            tflite_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Export finished but {tflite_path} not found. "
                f"Contents of {smd}: {list(smd.iterdir())}"
            )

    size_kb = tflite_path.stat().st_size / 1024
    print(f"[{fmt}] Saved: {tflite_path}  ({size_kb:.0f} KB)")
    return tflite_path


def _val_model(tflite_path: Path, args, run_name: str) -> dict:
    # tflite_path is inside {stem}_saved_model_tflite_{imgsz}/ which is inside weights/
    # so .parent.parent.parent goes up to the run root (alongside runs/)
    project = str(tflite_path.parent.parent.parent.parent / "runs_tflite")
    m = YOLO(str(tflite_path))
    metrics = m.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device="cpu",
        project=project,
        name=run_name,
        verbose=False,
    )
    return {
        "map50":     metrics.box.map50,
        "map":       metrics.box.map,
        "precision": metrics.box.mp,
        "recall":    metrics.box.mr,
    }


def _print_row(label, m, size_kb=None):
    size_str = f"  ({size_kb:.0f} KB)" if size_kb else ""
    print(f"  {label:<12}  mAP@50={m['map50']:.4f}  mAP@50:95={m['map']:.4f}"
          f"  P={m['precision']:.4f}  R={m['recall']:.4f}{size_str}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    weights_path = Path(args.weights).resolve()
    weights_dir  = weights_path.parent
    stem         = weights_path.stem

    smd = _saved_model_dir(weights_path, args.imgsz)

    print(f"\nWeights       : {weights_path}")
    print(f"Data          : {args.data}")
    print(f"Eval split    : {args.split}")
    print(f"Formats       : {args.formats}")
    print(f"ImgSz         : {args.imgsz}  (TFLite runs on CPU)")
    print(f"INT8 calib    : split={args.calib_split}, fraction={args.calib_fraction:.2f}")
    print(f"Output dir    : {smd}\n")

    # Checkpoint lives inside the per-imgsz saved_model dir
    checkpoint_path = smd / "tflite_eval.json"
    if checkpoint_path.exists() and not args.no_checkpoint:
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results   = ckpt.get("results", {})
        file_sizes = ckpt.get("file_sizes", {})
        skipped = [fmt for fmt in results if "error" not in results[fmt]]
        if skipped:
            print(f"Resuming from checkpoint — already done: {skipped}")
    else:
        results    = {}
        file_sizes = {}

    tflite_paths = {}

    # ── Export phase ──────────────────────────────────────────────────────────
    for fmt in args.formats:
        try:
            p = _export(weights_path, fmt, args)
            tflite_paths[fmt] = p
            file_sizes[fmt] = p.stat().st_size / 1024  # KB
        except Exception as e:
            print(f"[{fmt}] Export ERROR: {e}")

    # ── Eval phase ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Evaluating on split='{args.split}' (CPU) …")
    print("=" * 60)

    run_base = weights_path.parent.parent.name

    for fmt in args.formats:
        if fmt in results and "error" not in results[fmt] and not args.no_checkpoint:
            print(f"\n[{fmt}] Skipping (checkpoint exists)")
            _print_row(fmt, results[fmt], file_sizes.get(fmt))
            continue
        if fmt not in tflite_paths:
            print(f"[{fmt}] No model file — skipping eval")
            continue
        run_name = f"{run_base}-tflite-{fmt}-{args.split}"
        print(f"\n[{fmt}] {tflite_paths[fmt].name}")
        try:
            m = _val_model(tflite_paths[fmt], args, run_name)
            results[fmt] = m
            _print_row(fmt, m, file_sizes.get(fmt))
        except Exception as e:
            print(f"[{fmt}] Eval ERROR: {e}")
            results[fmt] = {"error": str(e)}
        with open(checkpoint_path, "w") as f:
            json.dump({"weights": str(weights_path), "data": args.data,
                       "split": args.split, "imgsz": args.imgsz,
                       "results": results, "file_sizes": file_sizes}, f, indent=2)
        print("  [checkpoint saved]")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Format':<12}  {'mAP@50':>8}  {'mAP@50:95':>10}  {'Precision':>10}  {'Recall':>8}  {'Size':>10}")
    print("-" * 60)
    baseline = results.get("fp32", {})
    for fmt in args.formats:
        m = results.get(fmt)
        size_str = f"{file_sizes[fmt]:.0f} KB" if fmt in file_sizes else "?"
        if m is None or "error" in m:
            print(f"  {fmt:<12}  {'ERROR':>8}  {size_str:>10}")
            continue
        delta = ""
        if fmt != "fp32" and "map50" in baseline:
            d = m["map50"] - baseline["map50"]
            delta = f"  ({d:+.4f} vs fp32)"
        print(f"  {fmt:<12}  {m['map50']:>8.4f}  {m['map']:>10.4f}  "
              f"{m['precision']:>10.4f}  {m['recall']:>8.4f}  {size_str:>10}{delta}")

    out_json = checkpoint_path
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
