"""bg_subtract.py — Background subtraction preprocessing for YOLO datasets.

Core API
--------
BgSubtractConfig          dataclass of all tunable knobs
preprocess()              grayscale + optional GaussianBlur
compute_diff()            absdiff or SSIM difference image
build_mask()              threshold + optional morph-open + dilation
apply_subtraction()       full pipeline → masked BGR image

YOLO integration (optional)
----------------------------
BgSubtractDataset         YOLODataset subclass; overrides load_image() to
                          apply subtraction on-the-fly (no disk writes).
                          Each frame is diffed against the preceding frame
                          in sorted filename order.  The first frame in a
                          sequence falls back to the original image.

make_bg_subtract_trainer  Factory that returns a DetectionTrainer subclass
                          wired to BgSubtractDataset with a given config.

Usage in train / finetune scripts
----------------------------------
    from bg_subtract import BgSubtractConfig, make_bg_subtract_trainer

    bg_cfg  = BgSubtractConfig()   # or override fields
    Trainer = make_bg_subtract_trainer(bg_cfg)

    model.train(
        data    = "...",
        trainer = Trainer,         # ← drop-in replacement
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BgSubtractConfig:
    # Reference mode
    # "temporal" : diff each frame against its filename-sorted predecessor
    # "fixed"    : diff every frame against a single fixed reference image
    ref_mode: str          = "fixed"
    fixed_ref_path: str | None = None   # required when ref_mode == "fixed"

    # Preprocessing
    use_blur: bool               = True
    blur_kernel: tuple[int, int] = (21, 21)
    blur_sigma: int              = 0        # 0 → auto-computed by cv2

    # Difference mode
    diff_mode: str               = "ssim"   # "absdiff" | "ssim"

    # absdiff: pixel diff > threshold → foreground
    absdiff_thresh: int          = 25

    # ssim: applied as THRESH_BINARY_INV on uint8 diff map
    ssim_thresh: int             = 180

    # Morphological opening (noise removal, applied before dilation)
    use_morph_open: bool               = True
    open_kernel_size: tuple[int, int]  = (5, 5)

    # Dilation (fill gaps in foreground mask)
    dilate_kernel: tuple[int, int]     = (7, 7)
    dilate_iters: int                  = 2

    def __post_init__(self):
        if self.ref_mode not in ("temporal", "fixed"):
            raise ValueError(
                f"ref_mode must be 'temporal' or 'fixed', got '{self.ref_mode}'"
            )
        # fixed_ref_path is validated only when the config is used, not here,
        # so that a placeholder config can be defined while USE_BG_SUBTRACTION=False.


# ── Core helpers ──────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray, cfg: BgSubtractConfig) -> np.ndarray:
    """Convert BGR image to grayscale and optionally apply GaussianBlur."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cfg.use_blur:
        gray = cv2.GaussianBlur(gray, cfg.blur_kernel, cfg.blur_sigma)
    return gray


def compute_diff(ref: np.ndarray, target: np.ndarray,
                 cfg: BgSubtractConfig) -> np.ndarray:
    """Return a uint8 difference image (brighter = more change).

    Parameters
    ----------
    ref, target : grayscale uint8 arrays of identical shape
    cfg         : BgSubtractConfig
    """
    if cfg.diff_mode == "absdiff":
        return cv2.absdiff(ref, target)

    elif cfg.diff_mode == "ssim":
        _score, diff = ssim(ref, target, full=True)
        return (diff * 255).astype(np.uint8)

    else:
        raise ValueError(
            f"Unknown diff_mode '{cfg.diff_mode}'. Choose 'absdiff' or 'ssim'."
        )


def build_mask(diff: np.ndarray, cfg: BgSubtractConfig) -> np.ndarray:
    """Threshold the diff image into a binary foreground mask."""
    if cfg.diff_mode == "absdiff":
        _, thresh = cv2.threshold(diff, cfg.absdiff_thresh, 255, cv2.THRESH_BINARY)
    else:  # ssim: high score → similar → background
        _, thresh = cv2.threshold(diff, cfg.ssim_thresh, 255, cv2.THRESH_BINARY_INV)

    if cfg.use_morph_open:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.open_kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)

    dilate_k = np.ones(cfg.dilate_kernel, np.uint8)
    return cv2.dilate(thresh, dilate_k, iterations=cfg.dilate_iters)


def apply_subtraction(ref_img: np.ndarray, target_img: np.ndarray,
                      cfg: BgSubtractConfig) -> np.ndarray:
    """Full pipeline: preprocess → diff → mask → masked target (BGR).

    Parameters
    ----------
    ref_img, target_img : BGR uint8 arrays (target is resized to ref if shapes differ)
    cfg                 : BgSubtractConfig
    """
    if ref_img.shape != target_img.shape:
        target_img = cv2.resize(target_img, (ref_img.shape[1], ref_img.shape[0]))

    ref_gray    = preprocess(ref_img,    cfg)
    target_gray = preprocess(target_img, cfg)

    diff = compute_diff(ref_gray, target_gray, cfg)
    mask = build_mask(diff, cfg)
    return cv2.bitwise_and(target_img, target_img, mask=mask)


# ── YOLO integration ──────────────────────────────────────────────────────────

def _make_bg_subtract_dataset_class(bg_cfg: BgSubtractConfig):
    """Return a YOLODataset subclass with bg_cfg baked in."""
    from ultralytics.data import YOLODataset

    # For "fixed" mode, load the reference once here so it is shared across
    # all load_image() calls without repeated disk reads.
    if bg_cfg.ref_mode == "fixed":
        _fixed_ref = cv2.imread(bg_cfg.fixed_ref_path)
        if _fixed_ref is None:
            raise FileNotFoundError(
                f"Cannot read fixed reference image: {bg_cfg.fixed_ref_path}"
            )
    else:
        _fixed_ref = None

    class BgSubtractDataset(YOLODataset):
        """YOLODataset that applies background subtraction in load_image().

        ref_mode="temporal":
            Images are sorted by filename so consecutive indices are
            consecutive camera-trap frames.  Index 0 falls back to the
            original image (no predecessor available).  The reference is
            loaded directly via cv2.imread to avoid recursive subtraction.

        ref_mode="fixed":
            Every image is diffed against the same fixed reference loaded
            once at class-construction time.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bg_cfg.ref_mode == "temporal":
                # Sort so index i−1 is always the temporal predecessor
                self.im_files = sorted(self.im_files)

        def load_image(self, i, rect_mode=True):
            im, hw0, hw = super().load_image(i, rect_mode)

            if bg_cfg.ref_mode == "fixed":
                try:
                    im = apply_subtraction(_fixed_ref, im, bg_cfg)
                except Exception:
                    pass  # fallback: return original image
                return im, hw0, hw

            else:  # temporal
                if i == 0:
                    return im, hw0, hw  # no predecessor for first frame

                # Load reference directly (raw, no caching) to avoid recursion
                ref_im = cv2.imread(self.im_files[i - 1])
                if ref_im is None:
                    return im, hw0, hw

                try:
                    im = apply_subtraction(ref_im, im, bg_cfg)
                except Exception:
                    pass  # fallback: return original image

                return im, hw0, hw

    return BgSubtractDataset


def make_bg_subtract_trainer(bg_cfg: BgSubtractConfig | None = None):
    """Return a DetectionTrainer subclass that uses background subtraction.

    Parameters
    ----------
    bg_cfg : BgSubtractConfig, or None to use defaults

    Returns
    -------
    A DetectionTrainer subclass; pass it as the ``trainer`` argument to
    ``model.train()``.

    Example
    -------
        Trainer = make_bg_subtract_trainer(BgSubtractConfig(diff_mode="ssim"))
        model.train(data="config.yaml", trainer=Trainer, ...)
    """
    if bg_cfg is None:
        bg_cfg = BgSubtractConfig()

    BgSubtractDataset = _make_bg_subtract_dataset_class(bg_cfg)

    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class BgSubtractDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            from ultralytics.utils import colorstr

            gs = max(
                int(unwrap_model(self.model).stride.max() if self.model else 0),
                32,
            )
            return BgSubtractDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or (mode == "val"),
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=int(gs),
                pad=0.0 if mode == "train" else 0.5,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
            )

    return BgSubtractDetectionTrainer
