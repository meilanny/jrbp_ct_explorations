"""train_encoder.py

Knowledge-distillation training of a lightweight image encoder for camera-trap
species recognition.

Architecture
------------
Teacher : DINOv2 ViT-S/14 (frozen)
          → L2-normalised 384-dim CLS embedding

Student : lightweight backbone  (default: MobileNetV2; MCUNet when available)
          + linear projection head  feat_dim → 384   (distillation target)
          + linear classifier       feat_dim → num_classes
          All three components trained jointly.

Loss
----
  total = alpha * L_distill + (1 - alpha) * L_cls

  L_distill = 1 - cosine_similarity(student_proj, teacher_emb).mean()
              (both sides L2-normalised before comparison)
  L_cls     = cross_entropy(student_logits, label_idx)

Training data
-------------
Cropped images derived from bounding boxes in speciesnet cleaned JSON files.
JSON format:
  { "CAMERA_ARRAY_X": { "camera_Y/filename.jpg": [
        { "bbox": [x, y, w, h],            # normalised top-left
          "categories": ["uuid;...;species", ...],
          "scores": [0.95, ...] }
  ] } }
Image root: /data/JR_CAMERA_TRAPS/images/{ARRAY_KEY}/{rel_path}

Usage
-----
  # Single label file:
  python train_encoder.py \\
      --labels ../speciesnet_labels_cleaned/speciesnet_labels_b1_cleaned.json \\
      --image_root /data/JR_CAMERA_TRAPS/images \\
      --output runs_encoder/exp1 --device 1 --epochs 100

  # Multiple label files (all cameras):
  python train_encoder.py \\
      --labels ../speciesnet_labels_cleaned/speciesnet_labels_a2_cleaned.json \\
               ../speciesnet_labels_cleaned/speciesnet_labels_b1_cleaned.json \\
               ../speciesnet_labels_cleaned/speciesnet_labels_b2_cleaned.json \\
      --image_root /data/JR_CAMERA_TRAPS/images \\
      --output runs_encoder/all_cameras --device 1 --epochs 100

  # Resume:
  python train_encoder.py ... --resume runs_encoder/exp1/ckpt_last.pt

Student backbone options (--student_arch):
  mobilenet_v2        torchvision MobileNetV2 (always available, feat_dim=1280)
  mcunet-512kB        MCUNet 512KB (requires MIT server; falls back to mobilenet_v2)
  mcunet-320kB        MCUNet 320KB
  mcunet-256kB        MCUNet 256KB
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

# ── Early log-file tee (same pattern as other scripts in this project) ────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device",   type=str, default="1")
_pre.add_argument("--log_file", type=str, default=None)
_pre_early, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_early.device


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

import torch
torch.cuda.init()  # prevent deferred CUDA check interference from onnxruntime

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated camera-trap images

from transformers import AutoModel, AutoImageProcessor


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DINOv2 → MCUNet/MobileNetV2 knowledge distillation encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--labels", nargs="+", required=True,
        help="Speciesnet cleaned JSON label file(s)",
    )
    p.add_argument(
        "--image_root", type=str,
        default="/data/JR_CAMERA_TRAPS/images",
        help="Root directory containing CAMERA_ARRAY_* subdirs",
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output directory for checkpoints and logs",
    )
    p.add_argument(
        "--student_arch", type=str, default="mcunet-in4",
        choices=["mobilenet_v2",
                 "mcunet-in4", "mcunet-in3", "mcunet-in2", "mcunet-in1", "mcunet-in0"],
        help=(
            "Student backbone. mcunet-in4 = 512KB/160px, in3 = 320KB/176px, "
            "in2 = 256KB, in1 = 5fps, in0 = 10fps. "
            "mobilenet_v2 = always available, feat_dim=1280, 224px."
        ),
    )
    p.add_argument("--teacher_model", type=str, default="facebook/dinov2-small")
    p.add_argument("--teacher_dim",   type=int, default=384)
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--wd",       type=float, default=1e-4,  help="Weight decay")
    p.add_argument("--alpha",    type=float, default=0.5,
                   help="Distillation loss weight; CE weight = 1 - alpha")
    p.add_argument("--imgsz",    type=int,   default=0,
                   help="Override input size (0 = use model's native resolution: "
                        "160 for mcunet-in4, 224 for mobilenet_v2)")
    p.add_argument("--bbox_pad", type=float, default=0.1,
                   help="Fractional padding added around each crop bbox")
    p.add_argument(
        "--min_bbox_side", type=float, default=0.02,
        help="Skip crops whose shorter normalised side is below this threshold",
    )
    p.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction of data held out for validation")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs (last is always saved)")
    p.add_argument("--resume",  type=str,   default=None,
                   help="Path to a checkpoint .pt file to resume from")
    p.add_argument("--device",  type=str,   default="1")
    p.add_argument("--log_file",type=str,   default=None)
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def _load_crops(label_files: list[str], image_root: Path,
                min_bbox_side: float) -> tuple[list[dict], list[str]]:
    """Parse cleaned JSON files → list of crop records, sorted class list."""
    records = []
    for lf in label_files:
        data = json.load(open(lf, encoding="utf-8"))
        for array_key, imgs in data.items():
            base = image_root / array_key
            for rel_path, detections in imgs.items():
                img_path = base / rel_path
                for det in detections:
                    x, y, w, h = det["bbox"]
                    if w < min_bbox_side or h < min_bbox_side:
                        continue
                    label = det["categories"][0].split(";")[-1]
                    records.append({
                        "path":  img_path,
                        "bbox":  (x, y, w, h),
                        "label": label,
                    })

    classes = sorted({r["label"] for r in records})
    return records, classes


class CropDataset(Dataset):
    """Yields (student_tensor, teacher_tensor, label_idx) for each bbox crop."""

    def __init__(self, records: list[dict], class_to_idx: dict[str, int],
                 student_tf, teacher_tf, bbox_pad: float):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.student_tf   = student_tf
        self.teacher_tf   = teacher_tf
        self.bbox_pad     = bbox_pad

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        W, H = img.size
        x, y, w, h = rec["bbox"]

        # Add padding and clamp to [0, 1]
        pad_x = w * self.bbox_pad
        pad_y = h * self.bbox_pad
        x1 = max(0.0, x - pad_x)
        y1 = max(0.0, y - pad_y)
        x2 = min(1.0, x + w + pad_x)
        y2 = min(1.0, y + h + pad_y)

        crop = img.crop((int(x1 * W), int(y1 * H),
                         int(x2 * W), int(y2 * H)))

        return (
            self.student_tf(crop),
            self.teacher_tf(crop),
            self.class_to_idx[rec["label"]],
        )


# ── Student model ─────────────────────────────────────────────────────────────

class StudentModel(nn.Module):
    """Backbone + projection head + classifier.

    backbone(x) → feat  (feat_dim,)
    proj(feat)  → emb   (teacher_dim,)  — distillation head
    head(feat)  → logits (num_classes,) — classification head
    """

    def __init__(self, backbone: nn.Module, feat_dim: int,
                 teacher_dim: int, num_classes: int):
        super().__init__()
        self.backbone   = backbone
        self.proj       = nn.Linear(feat_dim, teacher_dim, bias=False)
        self.head       = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat   = self.backbone(x)                        # (B, feat_dim)
        emb    = F.normalize(self.proj(feat), dim=-1)    # (B, teacher_dim)
        logits = self.head(feat)                         # (B, num_classes)
        return emb, logits


def _build_backbone_mobilenet_v2(pretrained: bool = True):
    """MobileNetV2 backbone: features + global pool → 1280-dim vector."""
    base = tvm.mobilenet_v2(
        weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    )
    # Replace classifier with identity; keep features + pool
    base.classifier = nn.Identity()
    feat_dim = 1280
    return base, feat_dim


def _build_backbone_mcunet(net_id: str, pretrained: bool = True):
    """MCUNet backbone via the mcunet pip package (requires fixed URL in model_zoo.py).

    ProxylessNASNets forward (from source):
      first_conv → blocks → feature_mix_layer (optional) → mean(H,W) → classifier
    We wrap everything up to and including the spatial mean as the backbone.

    Returns (backbone_module, feat_dim, input_resolution).
    """
    from mcunet.model_zoo import build_model

    model, resolution, _ = build_model(net_id, pretrained=pretrained)
    feat_dim = model.classifier.in_features  # LinearLayer wrapper exposes in_features

    class _MCUBackbone(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.first_conv        = m.first_conv
            self.blocks            = m.blocks
            self.feature_mix_layer = m.feature_mix_layer  # may be None

        def forward(self, x):
            x = self.first_conv(x)
            for block in self.blocks:
                x = block(x)
            if self.feature_mix_layer is not None:
                x = self.feature_mix_layer(x)
            x = x.mean(3).mean(2)   # global average pool over H and W
            return x                # (B, feat_dim)

    return _MCUBackbone(model), feat_dim, resolution


def build_student(arch: str, teacher_dim: int, num_classes: int,
                  imgsz: int | None) -> tuple["StudentModel", int]:
    """Return (StudentModel, input_resolution)."""
    if arch == "mobilenet_v2":
        backbone, feat_dim = _build_backbone_mobilenet_v2(pretrained=True)
        res = imgsz if imgsz else 224
        return StudentModel(backbone, feat_dim, teacher_dim, num_classes), res

    # MCUNet path
    try:
        backbone, feat_dim, native_res = _build_backbone_mcunet(arch, pretrained=True)
        res = imgsz if imgsz else native_res
        print(f"[student] Loaded {arch}  feat_dim={feat_dim}  resolution={res}")
        return StudentModel(backbone, feat_dim, teacher_dim, num_classes), res
    except Exception as e:
        print(f"[student] WARNING: {arch} unavailable ({e}). "
              f"Falling back to mobilenet_v2.")
        backbone, feat_dim = _build_backbone_mobilenet_v2(pretrained=True)
        res = imgsz if imgsz else 224
        return StudentModel(backbone, feat_dim, teacher_dim, num_classes), res


# ── Transforms ───────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def student_train_transform(imgsz: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def student_val_transform(imgsz: int) -> T.Compose:
    return T.Compose([
        T.Resize(imgsz + 32),
        T.CenterCrop(imgsz),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def teacher_transform(imgsz: int) -> T.Compose:
    # DINOv2 uses the same ImageNet normalisation; no augmentation (teacher is deterministic)
    return T.Compose([
        T.Resize(imgsz + 32),
        T.CenterCrop(imgsz),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ── Utilities ─────────────────────────────────────────────────────────────────

def _split_records(records: list[dict], val_frac: float,
                   seed: int) -> tuple[list[dict], list[dict]]:
    import hashlib
    def h(r):
        key = str(seed) + "|" + str(r["path"]) + str(r["bbox"])
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    val = [r for r in records if h(r) < val_frac]
    trn = [r for r in records if h(r) >= val_frac]
    return trn, val


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, val, n=1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1e-8)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CUDA_VISIBLE_DEVICES was set at module load time → always index 0

    print("=" * 70)
    print("Encoder Knowledge Distillation Training")
    print("=" * 70)
    print(f"  Teacher      : {args.teacher_model}  (frozen)")
    print(f"  Student      : {args.student_arch}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Device       : CUDA:{args.device} (mapped to cuda:0)")
    print(f"  Epochs       : {args.epochs}  |  Batch: {args.batch}")
    print(f"  LR: {args.lr}  WD: {args.wd}  Alpha (distill): {args.alpha}\n")

    # ── Load and split data ───────────────────────────────────────────────────
    print("Loading label files …")
    records, classes = _load_crops(
        args.labels, Path(args.image_root), args.min_bbox_side
    )
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"  Crop samples : {len(records):,}")
    print(f"  Classes      : {num_classes}")
    top5 = Counter(r["label"] for r in records).most_common(5)
    print(f"  Top-5 classes: {top5}")

    json.dump({"classes": classes},
              open(output_dir / "classes.json", "w"), indent=2)

    train_records, val_records = _split_records(records, args.val_split, args.seed)
    print(f"  Train / Val  : {len(train_records):,} / {len(val_records):,}\n")

    # ── Build teacher (frozen) ────────────────────────────────────────────────
    print("Loading teacher …")
    teacher_proc  = AutoImageProcessor.from_pretrained(args.teacher_model)
    teacher_model = AutoModel.from_pretrained(args.teacher_model).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    print(f"  {args.teacher_model}  →  {args.teacher_dim}-dim CLS embedding\n")

    # ── Build student ─────────────────────────────────────────────────────────
    print("Building student …")
    student, imgsz = build_student(
        args.student_arch, args.teacher_dim, num_classes,
        args.imgsz if args.imgsz > 0 else None   # None → use model's native resolution
    )
    student = student.to(device)
    feat_dim = student.proj.in_features
    print(f"  backbone feat_dim={feat_dim}  →  proj(384)  |  head({num_classes})\n")

    # ── Datasets + loaders ────────────────────────────────────────────────────
    s_train_tf = student_train_transform(imgsz)
    s_val_tf   = student_val_transform(imgsz)
    t_tf       = teacher_transform(imgsz)

    train_ds = CropDataset(train_records, class_to_idx, s_train_tf, t_tf, args.bbox_pad)
    val_ds   = CropDataset(val_records,   class_to_idx, s_val_tf,   t_tf, args.bbox_pad)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    ce_loss = nn.CrossEntropyLoss()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["student"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']})\n")

    # ── Helper: teacher forward ───────────────────────────────────────────────
    @torch.no_grad()
    def teacher_embed(teacher_crops: torch.Tensor) -> torch.Tensor:
        # teacher_crops: (B, 3, H, W) already normalised
        out = teacher_model(pixel_values=teacher_crops)
        cls = out.last_hidden_state[:, 0]       # (B, 384)
        return F.normalize(cls, dim=-1)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"{'Epoch':>6}  {'loss':>8}  {'l_dist':>8}  {'l_cls':>8}  "
          f"{'acc_tr':>7}  {'acc_val':>8}  {'lr':>9}  {'time':>7}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        t0 = time.time()
        m_loss = AverageMeter()
        m_dist = AverageMeter()
        m_cls  = AverageMeter()
        m_acc  = AverageMeter()

        for s_imgs, t_imgs, labels in train_loader:
            s_imgs = s_imgs.to(device, non_blocking=True)
            t_imgs = t_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Teacher embedding (no grad)
            t_emb = teacher_embed(t_imgs)

            # Student forward
            s_emb, logits = student(s_imgs)

            # Losses
            l_dist  = (1.0 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean()
            l_cls   = ce_loss(logits, labels)
            loss    = args.alpha * l_dist + (1.0 - args.alpha) * l_cls

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            bs = labels.size(0)
            m_loss.update(loss.item(),   bs)
            m_dist.update(l_dist.item(), bs)
            m_cls.update(l_cls.item(),   bs)
            m_acc.update(accuracy(logits, labels), bs)

        scheduler.step()

        # Validation
        student.eval()
        mv_acc  = AverageMeter()
        mv_dist = AverageMeter()
        with torch.no_grad():
            for s_imgs, t_imgs, labels in val_loader:
                s_imgs = s_imgs.to(device)
                t_imgs = t_imgs.to(device)
                labels = labels.to(device)
                t_emb = teacher_embed(t_imgs)
                s_emb, logits = student(s_imgs)
                mv_acc.update(accuracy(logits, labels),  labels.size(0))
                mv_dist.update(
                    (1.0 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean().item(),
                    labels.size(0),
                )

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        is_best = mv_acc.avg > best_val_acc
        if is_best:
            best_val_acc = mv_acc.avg

        print(f"{epoch:6d}  {m_loss.avg:8.4f}  {m_dist.avg:8.4f}  {m_cls.avg:8.4f}  "
              f"{m_acc.avg:7.4f}  {mv_acc.avg:8.4f}  {lr_now:9.2e}  {elapsed:6.0f}s"
              + ("  ★" if is_best else ""))

        # Checkpoint
        ckpt = {
            "epoch":        epoch,
            "student":      student.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "args":         vars(args),
            "classes":      classes,
        }
        torch.save(ckpt, output_dir / "ckpt_last.pt")
        if is_best:
            torch.save(ckpt, output_dir / "ckpt_best.pt")
        if epoch % args.save_every == 0:
            torch.save(ckpt, output_dir / f"ckpt_epoch{epoch:04d}.pt")

    print("-" * 70)
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
