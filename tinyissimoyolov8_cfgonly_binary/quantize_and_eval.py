"""quantize_and_eval.py

Export a trained YOLO .pt model to FP16 and/or INT8 quantized formats,
then evaluate each format on a test split and print a comparison table.

Supported export formats:
  fp16-onnx   — ONNX graph with FP16 weights (CPU/GPU, no TensorRT required)
  fp16-engine — TensorRT FP16 engine (requires TensorRT)
  int8-engine — TensorRT INT8 engine with calibration (requires TensorRT)

Usage examples
--------------
  # FP32 baseline + ONNX FP16 only (no TensorRT needed):
  python quantize_and_eval.py \\
      --weights runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt \\
      --data    camera_B1_yolo/config.yaml \\
      --formats fp32 fp16-onnx \\
      --device  1

  # All formats including TensorRT:
  python quantize_and_eval.py \\
      --weights runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt \\
      --data    camera_B1_yolo/config.yaml \\
      --formats fp32 fp16-onnx fp16-engine int8-engine \\
      --device  1 \\
      --imgsz   640

  # All formats including TensorRT:
  python quantize_and_eval.py \\
      --weights runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt \\
      --data    camera_A2_yolo/config.yaml \\
      --formats fp32 fp16-onnx fp16-engine int8-engine \\
      --device  1 \\
      --imgsz   640
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Parse --device and --log_file early (before torch import)
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device",   type=str, default="1")
_pre.add_argument("--log_file", type=str, default=None)
_pre_early, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_early.device

# Tee stdout/stderr to a log file if requested, with line-buffering
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()
    def fileno(self):
        return self._streams[0].fileno()

if _pre_early.log_file:
    _log_path = Path(_pre_early.log_file)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

from ultralytics import YOLO
import torch; torch.cuda.init()  # force CUDA init before onnxruntime deferred-check interferes


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLO quantization + eval script")
    p.add_argument(
        "--weights",
        type=str,
        default="runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-300eps/weights/best.pt",
        help="Path to the source .pt model weights",
    )
    p.add_argument(
        "--data",
        type=str,
        default="camera_B1_yolo/config.yaml",
        help="Path to dataset YAML (used for val split and INT8 calibration)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    p.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["fp32", "fp16-onnx"],
        choices=["fp32", "fp16-onnx", "fp16-engine", "int8-engine"],
        help="Quantization formats to export and evaluate",
    )
    p.add_argument(
        "--device",
        type=str,
        default="1",
        help="CUDA device index or 'cpu'",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export and eval (default: 640)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for val runs",
    )
    p.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project dir for val outputs (defaults to runs/ next to weights)",
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
        "--int8_calib_batches",
        type=int,
        default=30,
        help="Number of calibration batches for INT8 (default: 30)",
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
        help="Path to log file; stdout+stderr are tee'd there in addition to console",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _val_model(model_path: str, args, run_name: str, device: str) -> dict:
    project = args.project or str(Path(args.weights).parent.parent.parent / "runs_quantized")
    m = YOLO(model_path)
    metrics = m.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=device,
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


def _print_row(label, m):
    print(f"  {label:<20}  mAP@50={m['map50']:.4f}  mAP@50:95={m['map']:.4f}"
          f"  P={m['precision']:.4f}  R={m['recall']:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # CUDA_VISIBLE_DEVICES was already set at module load time (before torch import),
    # so ultralytics sees the selected GPU as index 0.
    device = "1"

    weights_path = Path(args.weights).resolve()
    weights_dir  = weights_path.parent          # …/weights/
    stem         = weights_path.stem             # e.g. "best"

    print(f"\nWeights : {weights_path}")
    print(f"Data    : {args.data}")
    print(f"Split   : {args.split}")
    print(f"Formats : {args.formats}")
    print(f"Device  : CUDA:{args.device} (remapped to index 0)")
    print(f"ImgSz   : {args.imgsz}\n")

    results = {}
    model_paths = {}  # format → path to evaluate

    # ── FP32 baseline ─────────────────────────────────────────────────────────
    if "fp32" in args.formats:
        model_paths["fp32"] = str(weights_path)

    # ── FP16 ONNX export ──────────────────────────────────────────────────────
    if "fp16-onnx" in args.formats:
        onnx_fp16_path = weights_dir / f"{stem}_fp16.onnx"
        if onnx_fp16_path.exists():
            print(f"[fp16-onnx] Reusing existing: {onnx_fp16_path}")
        else:
            print("[fp16-onnx] Exporting FP16 ONNX …")
            src = YOLO(str(weights_path))
            exported = src.export(
                format="onnx",
                half=True,
                imgsz=args.imgsz,
                device=device,
            )
            # Ultralytics writes to the same dir as the .pt; rename to mark FP16
            default_onnx = weights_dir / f"{stem}.onnx"
            if default_onnx.exists() and not onnx_fp16_path.exists():
                default_onnx.rename(onnx_fp16_path)
            elif Path(str(exported)).exists():
                Path(str(exported)).rename(onnx_fp16_path)
            print(f"[fp16-onnx] Saved: {onnx_fp16_path}")
        model_paths["fp16-onnx"] = str(onnx_fp16_path)

    # ── FP16 TensorRT engine ──────────────────────────────────────────────────
    if "fp16-engine" in args.formats:
        engine_fp16_path = weights_dir / f"{stem}_fp16.engine"
        if engine_fp16_path.exists():
            print(f"[fp16-engine] Reusing existing: {engine_fp16_path}")
        else:
            print("[fp16-engine] Exporting FP16 TensorRT engine (may take a few minutes) …")
            src = YOLO(str(weights_path))
            src.export(
                format="engine",
                half=True,
                imgsz=args.imgsz,
                device=device,
            )
            default_engine = weights_dir / f"{stem}.engine"
            if default_engine.exists():
                default_engine.rename(engine_fp16_path)
            print(f"[fp16-engine] Saved: {engine_fp16_path}")
        model_paths["fp16-engine"] = str(engine_fp16_path)

    # ── INT8 TensorRT engine ──────────────────────────────────────────────────
    if "int8-engine" in args.formats:
        engine_int8_path = weights_dir / f"{stem}_int8.engine"
        if engine_int8_path.exists():
            print(f"[int8-engine] Reusing existing: {engine_int8_path}")
        else:
            print("[int8-engine] Exporting INT8 TensorRT engine (calibration required, slow) …")
            src = YOLO(str(weights_path))
            src.export(
                format="engine",
                int8=True,
                dynamic=True,
                data=args.data,            # calibration dataset
                imgsz=args.imgsz,
                device=device,
                batch=args.batch,          # sets engine max-batch profile; must cover val batch size
            )
            default_engine = weights_dir / f"{stem}.engine"
            if default_engine.exists():
                default_engine.rename(engine_int8_path)
            print(f"[int8-engine] Saved: {engine_int8_path}")
        model_paths["int8-engine"] = str(engine_int8_path)

    # ── Evaluate each format ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Evaluating on split='{args.split}' …")
    print("=" * 60)

    run_base = Path(args.weights).parent.parent.name  # e.g. run folder name

    # Load checkpoint so a restarted run skips already-completed formats
    checkpoint_path = weights_dir / "quantization_eval.json"
    if checkpoint_path.exists() and not args.no_checkpoint:
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", {})
        skipped = [fmt for fmt in results if "error" not in results[fmt]]
        if skipped:
            print(f"Resuming from checkpoint — already done: {skipped}")
    else:
        results = {}

    for fmt in args.formats:
        if fmt in results and "error" not in results[fmt] and not args.no_checkpoint:
            print(f"\n[{fmt}] Skipping (checkpoint exists)")
            _print_row(fmt, results[fmt])
            continue
        path = model_paths.get(fmt)
        if path is None or not Path(path).exists():
            print(f"[{fmt}] Model file not found: {path} — skipping")
            continue
        run_name = f"{run_base}-quant-{fmt}-{args.split}"
        print(f"\n[{fmt}] {Path(path).name}")
        try:
            m = _val_model(path, args, run_name, device)
            results[fmt] = m
            _print_row(fmt, m)
        except Exception as e:
            print(f"[{fmt}] ERROR: {e}")
            results[fmt] = {"error": str(e)}
        # Save checkpoint after each format
        with open(checkpoint_path, "w") as f:
            json.dump({"weights": str(weights_path), "data": args.data,
                       "split": args.split, "results": results}, f, indent=2)
        print(f"  [checkpoint saved]")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Format':<20}  {'mAP@50':>8}  {'mAP@50:95':>10}  {'Precision':>10}  {'Recall':>8}")
    print("-" * 60)
    baseline = results.get("fp32", {})
    for fmt in args.formats:
        m = results.get(fmt)
        if m is None or "error" in m:
            print(f"  {fmt:<20}  {'ERROR':>8}")
            continue
        delta = ""
        if fmt != "fp32" and "map50" in baseline:
            d = m["map50"] - baseline["map50"]
            delta = f"  ({d:+.4f} vs fp32)"
        print(f"  {fmt:<20}  {m['map50']:>8.4f}  {m['map']:>10.4f}  "
              f"{m['precision']:>10.4f}  {m['recall']:>8.4f}{delta}")

    # ── Save JSON summary ────────────────────────────────────────────────────
    out_json = weights_dir / "quantization_eval.json"
    with open(out_json, "w") as f:
        json.dump({"weights": str(weights_path), "data": args.data,
                   "split": args.split, "results": results}, f, indent=2)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
