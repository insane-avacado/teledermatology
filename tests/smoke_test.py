from pathlib import Path

import cv2
import numpy as np

from sff import compute_depth_index, depth_index_to_heatmap, load_stack


def _make_synthetic_stack(n: int = 5, h: int = 96, w: int = 128) -> list[np.ndarray]:
    """Create a simple synthetic stack with varying blur strengths."""
    base = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(base, (16, 16), (w - 16, h - 16), 180, 2)
    cv2.line(base, (0, h // 2), (w - 1, h // 2), 255, 1)
    cv2.circle(base, (w // 2, h // 2), 18, 220, 1)

    stack = []
    sigmas = np.linspace(0.2, 3.0, n)
    for sigma in sigmas:
        if sigma < 0.5:
            stack.append(base.copy())
        else:
            blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
            stack.append(blurred)
    return stack


def test_sff_shapes() -> None:
    stack = _make_synthetic_stack(n=6)
    depth_idx, confidence = compute_depth_index(stack, measure="laplacian")

    assert depth_idx.ndim == 2
    assert confidence.ndim == 2
    assert depth_idx.shape == confidence.shape == stack[0].shape
    assert depth_idx.dtype == np.uint8
    assert confidence.dtype == np.float32

    heatmap = depth_index_to_heatmap(depth_idx, cmap="TURBO", invert=False, num_levels=len(stack))
    assert heatmap.shape[:2] == depth_idx.shape
    assert heatmap.shape[2] == 3
    assert heatmap.dtype == np.uint8


def test_load_stack(tmp_path: Path) -> None:
    for i in range(3):
        img = np.full((20, 30), 30 + i * 10, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), img)

    loaded = load_stack(tmp_path)
    assert len(loaded) == 3
    assert loaded[0].shape == (20, 30)
