import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

USE_BLUR  = True          # apply GaussianBlur before computing diff
DIFF_MODE = "absdiff"     # "absdiff" | "ssim"

BLUR_KERNEL   = (21, 21)  # GaussianBlur kernel size (odd values only)
BLUR_SIGMA    = 0         # 0 → auto-computed by cv2

# absdiff threshold: pixel diff > value → foreground
ABSDIFF_THRESH = 25

# ssim threshold: ssim score < (1 - value/255) → foreground
# ssim diff image is in [0, 1]; we scale to uint8 and threshold
SSIM_THRESH    = 180      # applied as THRESH_BINARY_INV on uint8 ssim diff

DILATE_KERNEL = (7, 7)
DILATE_ITERS  = 2

# Morphological opening: erode then dilate to remove small noise specks.
# Applied *before* dilation when USE_MORPH_OPEN=True.
# Useful when the diff/threshold produces many tiny isolated blobs
# (common with SSIM on compressed JPEGs or absdiff on textured backgrounds).
USE_MORPH_OPEN   = True
OPEN_KERNEL_SIZE = (5, 5)   # smaller than DILATE_KERNEL to preserve real regions


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray, use_blur: bool) -> np.ndarray:
    """Convert to grayscale and optionally apply GaussianBlur."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_blur:
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL, BLUR_SIGMA)
    return gray


def compute_diff(ref: np.ndarray, target: np.ndarray, mode: str) -> np.ndarray:
    """Return a uint8 difference image using the selected mode.

    Parameters
    ----------
    ref, target : grayscale uint8 arrays of the same shape
    mode        : "absdiff" or "ssim"

    Returns
    -------
    diff : uint8 array; brighter pixels = more change
    """
    if mode == "absdiff":
        return cv2.absdiff(ref, target)

    elif mode == "ssim":
        score, diff = ssim(ref, target, full=True)
        print(f"SSIM score: {score:.4f}")
        # diff is in [-1, 1] (typically [0, 1] for similar images);
        # convert to uint8 so standard cv2 thresholding applies
        diff_uint8 = (diff * 255).astype(np.uint8)
        return diff_uint8

    else:
        raise ValueError(f"Unknown diff mode '{mode}'. Choose 'absdiff' or 'ssim'.")


def build_mask(diff: np.ndarray, mode: str) -> np.ndarray:
    """Threshold the diff image into a binary foreground mask."""
    if mode == "absdiff":
        _, thresh = cv2.threshold(diff, ABSDIFF_THRESH, 255, cv2.THRESH_BINARY)
    else:  # ssim: high ssim score (close to 255) → similar → background
        _, thresh = cv2.threshold(diff, SSIM_THRESH, 255, cv2.THRESH_BINARY_INV)

    # Optional opening: remove isolated noise specks before growing regions
    if USE_MORPH_OPEN:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KERNEL_SIZE)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)

    dilate_kernel = np.ones(DILATE_KERNEL, np.uint8)
    mask = cv2.dilate(thresh, dilate_kernel, iterations=DILATE_ITERS)
    return mask


# ── Main ──────────────────────────────────────────────────────────────────────

model = YOLO('runs/tinyissimoYOLOv8-binary-best_500eps_on_A1-preprocessBG_abs_all-ref37-finetune_on_B1_82split-200eps'
            '/weights/best.pt') 

# model = YOLO('runs/tinyissimoYOLOv8-binary-fulldataset-82split-excludetest-500eps'
#             '/weights/best_100.pt')

# model = YOLO('best_tinyissimoyolo_v8_on_A2_subset.pt')

ref_img    = cv2.imread('/home/lanmei/jrbp_ct_explorations/'
                        'tinyissimoyolov8_cfgonly_binary/'
                        'camera_B1_yolo/test/images/p_000037.jpg20180327.jpg') # p_000002 # p_000037
target_img = cv2.imread('/home/lanmei/jrbp_ct_explorations/'
                        'tinyissimoyolov8_cfgonly_binary/'
                        'camera_B1_yolo/test/images/p_000025.jpg20180327.jpg') # p_000003

'''
ref_img    = cv2.imread('/home/lanmei/jrbp_ct_explorations/'
                        'tinyissimoyolov8_cfgonly_binary/'
                        'camera_B1_yolo/test/images/p_000045.jpg20180327.jpg') # p_000037
target_img = cv2.imread('/home/lanmei/jrbp_ct_explorations/'
                        'tinyissimoyolov8_cfgonly_binary/'
                        'camera_B1_yolo/test/images/p_000044.jpg20180327.jpg')
'''                     


if ref_img.shape != target_img.shape:
    target_img = cv2.resize(target_img, (ref_img.shape[1], ref_img.shape[0]))

# Preprocess
ref_gray    = preprocess(ref_img,    use_blur=USE_BLUR)
target_gray = preprocess(target_img, use_blur=USE_BLUR)

# Compute difference and mask
diff       = compute_diff(ref_gray, target_gray, mode=DIFF_MODE)
clean_mask = build_mask(diff, mode=DIFF_MODE)

# Apply mask and run YOLO
masked_target = cv2.bitwise_and(target_img, target_img, mask=clean_mask)
results = model([masked_target, target_img], save=True)