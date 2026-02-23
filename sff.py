from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

from utils_io import list_images


FocusMeasureName = Literal["laplacian", "tenengrad"]
ColormapName = Literal["TURBO", "JET", "VIRIDIS"]


def load_stack(folder: Path) -> List[np.ndarray]:
    """Load focal stack images from a folder (PNG/JPG), sorted by filename.

    Images are loaded in grayscale (`uint8`) because focus measures are computed
    from intensity gradients.

    Args:
        folder: Directory containing focal stack images.

    Returns:
        List of grayscale images as 2D `uint8` arrays.
    """
    images: List[np.ndarray] = []
    for image_path in list_images(folder):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def focus_measure_laplacian(img: np.ndarray, blur_ksize: int, lap_ksize: int) -> np.ndarray:
    """Compute Laplacian energy focus measure.

    Steps:
    1) Optional Gaussian pre-blur to reduce sensor noise.
    2) Laplacian response (`ksize` in {1, 3}).
    3) Energy = squared response.

    Args:
        img: 2D grayscale image (`uint8` or float).
        blur_ksize: 0 (disabled) or odd kernel size (e.g., 3/5).
        lap_ksize: Laplacian kernel size, expected 1 or 3.

    Returns:
        Focus energy map as `float32`.
    """
    work = img.astype(np.float32)
    if blur_ksize > 0:
        work = cv2.GaussianBlur(work, (blur_ksize, blur_ksize), 0)

    lap = cv2.Laplacian(work, cv2.CV_32F, ksize=lap_ksize)
    return lap * lap


def focus_measure_tenengrad(img: np.ndarray, blur_ksize: int) -> np.ndarray:
    """Compute Tenengrad focus measure using Sobel gradient magnitude squared.

    Steps:
    1) Optional Gaussian pre-blur.
    2) Sobel X and Y with kernel size 3.
    3) Tenengrad energy = Gx^2 + Gy^2.

    Args:
        img: 2D grayscale image (`uint8` or float).
        blur_ksize: 0 (disabled) or odd kernel size (e.g., 3/5).

    Returns:
        Focus energy map as `float32`.
    """
    work = img.astype(np.float32)
    if blur_ksize > 0:
        work = cv2.GaussianBlur(work, (blur_ksize, blur_ksize), 0)

    grad_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    return (grad_x * grad_x) + (grad_y * grad_y)


def aggregate_window(fm: np.ndarray, ksize: int) -> np.ndarray:
    """Aggregate focus measure spatially via normalized box filter.

    A larger window improves robustness in low-texture regions but reduces spatial
    detail in the depth index.

    Args:
        fm: Focus measure image (`float32`).
        ksize: Odd window size in [3, 15].

    Returns:
        Smoothed focus measure image as `float32`.
    """
    if ksize <= 1:
        return fm.astype(np.float32)
    return cv2.blur(fm.astype(np.float32), (ksize, ksize))


def compute_depth_index(
    stack: List[np.ndarray],
    measure: FocusMeasureName = "laplacian",
    window_size: int = 7,
    preblur: int = 3,
    lap_ksize: int = 3,
    median_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SFF depth index and confidence from a focal stack.

    Algorithm:
    - Compute focus measure per frame.
    - Aggregate with local normalized box filter.
    - Select per-pixel best frame via argmax over stack dimension.
    - Confidence = (top1 - top2) / (top1 + 1e-6).
    - Optionally median-filter depth index to denoise label noise.

    Args:
        stack: List of 2D grayscale frames with equal size.
        measure: "laplacian" or "tenengrad".
        window_size: Odd integer, typically 3-15.
        preblur: 0 or odd Gaussian kernel size.
        lap_ksize: Laplacian kernel size (1 or 3).
        median_size: 0 (disabled) or odd median kernel size.

    Returns:
        depth_idx: `uint8` depth index map in [0, n_frames-1].
        confidence: `float32` normalized confidence in [0, 1] (approx).

    Raises:
        ValueError: If fewer than 2 images are provided.
    """
    num_frames = len(stack)
    if num_frames < 2:
        raise ValueError("Need at least 2 images for shape-from-focus.")

    focus_volume: List[np.ndarray] = []
    for img in stack:
        if measure == "laplacian":
            fm = focus_measure_laplacian(img, blur_ksize=preblur, lap_ksize=lap_ksize)
        elif measure == "tenengrad":
            fm = focus_measure_tenengrad(img, blur_ksize=preblur)
        else:
            raise ValueError(f"Unsupported focus measure: {measure}")

        fm_agg = aggregate_window(fm, ksize=window_size)
        focus_volume.append(fm_agg)

    volume = np.stack(focus_volume, axis=0).astype(np.float32)  # [N, H, W]

    depth_idx = np.argmax(volume, axis=0).astype(np.uint8)

    sorted_vals = np.partition(volume, kth=num_frames - 2, axis=0)
    top1 = sorted_vals[-1]
    top2 = sorted_vals[-2]
    confidence = (top1 - top2) / (top1 + 1e-6)
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

    if median_size and median_size > 1:
        depth_idx = cv2.medianBlur(depth_idx, median_size)

    return depth_idx, confidence


def depth_index_to_heatmap(
    depth_idx: np.ndarray,
    cmap: ColormapName = "TURBO",
    invert: bool = False,
    num_levels: Optional[int] = None,
) -> np.ndarray:
    """Convert depth index map to a color heatmap using OpenCV colormaps.

    Args:
        depth_idx: 2D `uint8` map containing depth indices.
        cmap: Color map name in {"TURBO", "JET", "VIRIDIS"}.
        invert: If True, invert normalized depth mapping.
        num_levels: Optional number of valid depth levels (e.g., stack length).
            If provided, depth values are scaled by this range for better contrast.

    Returns:
        3-channel BGR heatmap (`uint8`) from `cv2.applyColorMap`.
    """
    if depth_idx.dtype != np.uint8:
        depth_work = depth_idx.astype(np.uint8)
    else:
        depth_work = depth_idx.copy()

    if num_levels is not None and num_levels > 1:
        max_index = float(max(1, num_levels - 1))
        depth_norm = np.clip((depth_work.astype(np.float32) / max_index) * 255.0, 0, 255).astype(np.uint8)
    else:
        depth_norm = cv2.normalize(depth_work, None, 0, 255, cv2.NORM_MINMAX)

    if invert:
        depth_norm = 255 - depth_norm

    cmap_map = {
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }
    return cv2.applyColorMap(depth_norm, cmap_map[cmap])
