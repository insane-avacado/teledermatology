from pathlib import Path
from typing import List

import cv2
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def ensure_dir(path: Path) -> None:
    """Create directory (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> List[Path]:
    """Return sorted PNG/JPG image paths under a folder.

    Args:
        path: Folder containing image files.

    Returns:
        Sorted list of image paths.
    """
    if not path.exists() or not path.is_dir():
        return []
    files = [p for p in path.iterdir() if p.suffix in VALID_EXTENSIONS and p.is_file()]
    return sorted(files, key=lambda p: p.name)


def save_image(path: Path, img: np.ndarray) -> bool:
    """Save an image with OpenCV and return success status."""
    ensure_dir(path.parent)
    return bool(cv2.imwrite(str(path), img))
