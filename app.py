from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sff import compute_depth_index, depth_index_to_heatmap, load_stack
from utils_io import list_images, save_image


def _resize_stack_if_needed(stack: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
    """Resize all images to first frame size if needed.

    Returns:
        (resized_stack, had_resize)
    """
    if not stack:
        return stack, False

    h0, w0 = stack[0].shape[:2]
    resized = []
    changed = False
    for img in stack:
        h, w = img.shape[:2]
        if (h, w) != (h0, w0):
            img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
            changed = True
        resized.append(img)
    return resized, changed


def _brightness_warning(stack: List[np.ndarray], threshold_ratio: float = 0.08) -> bool:
    """Detect notable global brightness variation across the stack."""
    if len(stack) < 2:
        return False
    means = np.array([float(np.mean(img)) for img in stack], dtype=np.float32)
    mean_global = float(np.mean(means)) + 1e-6
    rel_span = float((np.max(means) - np.min(means)) / mean_global)
    return rel_span > threshold_ratio


def _to_display_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _next_stack_image_path(stack_folder: Path, image_files: List[Path]) -> Path:
    """Generate next sequential filename in the stack folder.

    Uses pattern `img_XXX.png` with zero-padded 3+ digits.
    """
    if not image_files:
        return stack_folder / "img_000.png"

    max_idx = -1
    for file_path in image_files:
        match = re.search(r"(\d+)$", file_path.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1 if max_idx >= 0 else len(image_files)
    return stack_folder / f"img_{next_idx:03d}.png"


def _estimate_skin_reference_index(depth_idx: np.ndarray) -> float:
    """Estimate skin reference index from border pixels (robust baseline)."""
    h, w = depth_idx.shape
    pad_h = max(1, h // 10)
    pad_w = max(1, w // 10)

    top = depth_idx[:pad_h, :]
    bottom = depth_idx[-pad_h:, :]
    left = depth_idx[:, :pad_w]
    right = depth_idx[:, -pad_w:]
    border_vals = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()]).astype(np.float32)
    return float(np.median(border_vals))


def _signed_height_map(depth_idx: np.ndarray, invert: bool, skin_reference: float) -> np.ndarray:
    """Convert depth index into signed relative height.

    Positive values are treated as above skin surface.
    Negative values are treated as deeper than skin surface.
    """
    depth_f = depth_idx.astype(np.float32)
    if invert:
        return depth_f - skin_reference
    return skin_reference - depth_f


def _lesion_facecolors_plotly(height_map: np.ndarray, depth_min: Optional[float] = None, depth_max: Optional[float] = None) -> np.ndarray:
    """Create normalized depth values for Plotly colorscale (red-blue only).

    Color semantics:
    - bright blue (0.0): shallow / near skin surface
    - dark red (1.0): deep / lesion interior

    Args:
        height_map: Depth values (can be relative or calibrated in mm).
        depth_min: Minimum depth value for color mapping (optional).
        depth_max: Maximum depth value for color mapping (optional).

    Returns:
        Normalized values in [0, 1] for use with Plotly colorscale.
    """
    h_flat = height_map.astype(np.float32)
    if depth_min is None or depth_max is None:
        h_min = float(np.nanmin(h_flat))
        h_max = float(np.nanmax(h_flat))
    else:
        h_min = depth_min
        h_max = depth_max
    norm = (h_flat - h_min) / (h_max - h_min + 1e-6)
    return np.clip(norm, 0.0, 1.0)


def _render_3d_lesion_plot_plotly(
    height_map: np.ndarray,
    sample_step: int,
    z_scale: float,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    is_calibrated: bool = False,
) -> go.Figure:
    """Render an interactive 3D surface plot using Plotly (red-blue colorscale).

    Click and drag to rotate, scroll to zoom. Colors: blue (shallow) → red (deep).

    Args:
        height_map: Depth values (relative or calibrated in mm).
        sample_step: Downsampling factor.
        z_scale: Vertical exaggeration factor.
        depth_min: Minimum depth for color scaling (optional).
        depth_max: Maximum depth for color scaling (optional).
        is_calibrated: If True, use calibrated depth scaling for colors.
    """
    z = height_map[::sample_step, ::sample_step] * z_scale
    h_colored = height_map[::sample_step, ::sample_step]

    color_vals = _lesion_facecolors_plotly(h_colored, depth_min=depth_min, depth_max=depth_max)

    z_label = "Depth (mm)" if is_calibrated else "Relative Height"
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                surfacecolor=color_vals,
                colorscale=[[0, "rgb(38, 191, 255)"], [1, "rgb(115, 13, 13)"]],
                colorbar=dict(title="Depth"),
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title="3D Lesion Depth Map (Interactive - Drag to Rotate)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        width=900,
        height=700,
        hovermode="closest",
    )
    return fig


def _render_3d_lesion_plot(
    height_map: np.ndarray,
    sample_step: int,
    z_scale: float,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    is_calibrated: bool = False,
) -> go.Figure:
    """Render a 3D surface plot using Plotly with interactive rotation.
    
    Delegates to _render_3d_lesion_plot_plotly for interactive visualization.
    """
    return _render_3d_lesion_plot_plotly(
        height_map=height_map,
        sample_step=sample_step,
        z_scale=z_scale,
        depth_min=depth_min,
        depth_max=depth_max,
        is_calibrated=is_calibrated,
    )


def _calibrate_depth_index(
    depth_idx: np.ndarray,
    skin_ref_idx: float,
    scale_mm_per_step: float,
) -> np.ndarray:
    """Convert depth index to physical depth in millimeters.

    Args:
        depth_idx: Raw depth index map (uint8).
        skin_ref_idx: Depth index value representing skin surface (zero depth).
        scale_mm_per_step: Millimeters per unit change in focus step.

    Returns:
        Calibrated depth map in mm (float32). Positive = deeper, negative = shallower.
    """
    depth_f = depth_idx.astype(np.float32)
    relative_depth = depth_f - skin_ref_idx
    calibrated_mm = relative_depth * scale_mm_per_step
    return calibrated_mm


def _format_depth_display(depth_map_mm: np.ndarray, num_decimals: int = 2) -> str:
    """Create a text summary of calibrated depth statistics."""
    valid = np.isfinite(depth_map_mm)
    if not np.any(valid):
        return "No valid depth data."
    vals = depth_map_mm[valid]
    return (
        f"Min: {np.min(vals):.{num_decimals}f} mm | "
        f"Max: {np.max(vals):.{num_decimals}f} mm | "
        f"Mean: {np.mean(vals):.{num_decimals}f} mm"
    )


def main() -> None:
    st.set_page_config(page_title="Local SFF Depth Heatmap", layout="wide")
    st.title("Local Shape-from-Focus (SFF)")
    st.caption("Manual focal stack in, depth index + confidence + heatmap out (fully local).")

    st.sidebar.header("Controls")
    stack_folder_text = st.sidebar.text_input(
        "Stack folder",
        value="data/focal_stack",
        help="Path to folder containing sequential PNG/JPG focal stack images.",
    )

    measure_ui = st.sidebar.selectbox(
        "Focus measure",
        options=["Laplacian", "Tenengrad"],
        index=0,
        help="Laplacian: second-derivative edge energy. Tenengrad: Sobel gradient energy.",
    )
    preblur = st.sidebar.selectbox(
        "Pre-blur (Gaussian KSIZE)",
        options=[0, 3, 5],
        index=1,
        help="Small blur can suppress noise before focus measurement.",
    )
    lap_ksize = st.sidebar.selectbox(
        "Laplacian ksize",
        options=[1, 3],
        index=1,
        help="Laplacian kernel size (used only for Laplacian measure).",
    )
    window_size = st.sidebar.slider(
        "Local window size",
        min_value=3,
        max_value=15,
        value=7,
        step=2,
        help="Odd box-filter window for focus aggregation. Larger = smoother, less detail.",
    )
    median_size = st.sidebar.selectbox(
        "Median filter size",
        options=[0, 3, 5, 7],
        index=2,
        help="Optional denoising on final depth index map. 0 disables.",
    )
    cmap = st.sidebar.selectbox(
        "Heatmap colormap",
        options=["TURBO", "JET", "VIRIDIS"],
        index=0,
        help="Color map used to visualize relative depth index.",
    )
    invert = st.sidebar.checkbox(
        "Invert depth mapping",
        value=False,
        help="Invert near/far color assignment for the heatmap.",
    )
    show_3d = st.sidebar.checkbox(
        "Show 3D lesion map",
        value=True,
        help="Display a 3D depth visualization using lesion-specific color semantics.",
    )
    sample_step = st.sidebar.selectbox(
        "3D downsample step",
        options=[1, 2, 4, 8],
        index=1,
        help="Larger step renders faster for large images.",
    )
    z_scale = st.sidebar.slider(
        "3D height scale",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.5,
        help="Vertical exaggeration of the 3D depth surface.",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Calibration")
    enable_cal = st.sidebar.checkbox(
        "Enable depth calibration",
        value=False,
        help="Convert depth index to physical depth (mm) using reference points.",
    )
    skin_ref_idx = 128
    scale_mm_per_step = 0.05
    if enable_cal:
        skin_ref_idx = st.sidebar.slider(
            "Skin surface index",
            min_value=0,
            max_value=255,
            value=128,
            step=1,
            help="Depth index value representing the skin surface (zero depth).",
        )
        scale_mm_per_step = st.sidebar.number_input(
            "Scale (mm per step)",
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Millimeters per unit change in focus index (e.g., 0.05 mm/step).",
        )
        if st.sidebar.button("Reset calibration", type="secondary"):
            st.session_state.pop("calibration", None)
            st.rerun()

    run_clicked = st.sidebar.button("Run SFF", type="primary")
    save_clicked = st.sidebar.button("Save Outputs")

    stack_folder = Path(stack_folder_text).expanduser()
    image_files = list_images(stack_folder)

    st.subheader("Camera capture")
    st.caption("Live view is provided by your browser camera permissions. Use Brio + Logi Tune manual focus while capturing.")

    with st.expander("Focus guidance", expanded=True):
        st.markdown(
            """
1. Lock camera position and keep scene static.
2. In Logi Tune, turn autofocus OFF.
3. Start near focus, capture one frame.
4. Move focus slider a small step toward far focus.
5. Capture again and repeat until you have 20–60 frames.
6. Keep exposure and white-balance stable.
            """.strip()
        )

    captured = st.camera_input(
        "Live camera feed (click camera button to capture)",
        help="Uses browser camera stream. After capture, click 'Save captured image to stack'.",
    )

    if captured is not None:
        captured_bytes = captured.getvalue()
        arr = np.frombuffer(captured_bytes, dtype=np.uint8)
        captured_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if captured_bgr is not None:
            st.image(cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2RGB), caption="Captured frame preview", use_column_width=True)

            if st.button("Save captured image to stack"):
                stack_folder.mkdir(parents=True, exist_ok=True)
                latest_files = list_images(stack_folder)
                out_path = _next_stack_image_path(stack_folder, latest_files)
                ok = save_image(out_path, captured_bgr)
                if ok:
                    st.success(f"Saved: {out_path}")
                    st.rerun()
                else:
                    st.error("Failed to save captured image.")

    st.subheader("Stack preview")
    if not image_files:
        st.info("No PNG/JPG files found yet. Set a valid stack folder in the sidebar.")
    else:
        st.write(f"Detected **{len(image_files)}** frames in: `{stack_folder}`")
        preview_idx = st.slider("Preview frame", 0, len(image_files) - 1, 0, 1)
        selected_path = image_files[preview_idx]
        preview_bgr = cv2.imread(str(image_files[preview_idx]), cv2.IMREAD_COLOR)
        if preview_bgr is not None:
            st.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), caption=image_files[preview_idx].name, use_column_width=True)

        if st.button("Delete selected image", type="secondary", help="Removes the currently previewed frame so you can retake it."):
            try:
                selected_path.unlink(missing_ok=False)
                st.success(f"Deleted: {selected_path.name}")
                st.rerun()
            except FileNotFoundError:
                st.warning("Selected image no longer exists.")
                st.rerun()
            except OSError as err:
                st.error(f"Failed to delete image: {err}")

        with st.expander("Thumbnail grid (first 12 frames)", expanded=False):
            cols = st.columns(4)
            for idx, file_path in enumerate(image_files[:12]):
                img_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                cols[idx % 4].image(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    caption=file_path.name,
                    use_column_width=True,
                )

    if len(image_files) > 200:
        st.warning("Large stack (>200 frames): processing can be slow and memory-heavy. Consider downsampling or fewer frames.")

    measure = "laplacian" if measure_ui == "Laplacian" else "tenengrad"

    if run_clicked:
        if len(image_files) < 2:
            st.error("Need at least 2 images to run SFF.")
            st.stop()

        t0 = perf_counter()
        stack = load_stack(stack_folder)
        if len(stack) < 2:
            st.error("Failed to load enough valid images from the folder.")
            st.stop()

        stack, resized = _resize_stack_if_needed(stack)
        if resized:
            st.warning("Input image sizes differed. All frames were resized to match the first frame.")

        if _brightness_warning(stack):
            st.warning("Brightness varied across frames. Lock exposure/white-balance in Logi Tune for best results.")

        depth_idx, confidence = compute_depth_index(
            stack,
            measure=measure,
            window_size=window_size,
            preblur=preblur,
            lap_ksize=lap_ksize,
            median_size=median_size,
        )
        heatmap_bgr = depth_index_to_heatmap(depth_idx, cmap=cmap, invert=invert, num_levels=len(stack))

        elapsed = perf_counter() - t0
        h, w = depth_idx.shape[:2]

        st.session_state["depth_idx"] = depth_idx
        st.session_state["confidence"] = confidence
        st.session_state["heatmap_bgr"] = heatmap_bgr

        st.success("SFF completed.")
        st.write(f"Dimensions: **{w} x {h}** | Frames: **{len(stack)}** | Time: **{elapsed:.3f} s**")
        
        if enable_cal:
            st.session_state["calibration"] = {"skin_ref_idx": skin_ref_idx, "scale_mm_per_step": scale_mm_per_step}

    if "depth_idx" in st.session_state:
        depth_idx = st.session_state["depth_idx"]
        confidence = st.session_state["confidence"]
        heatmap_bgr = st.session_state["heatmap_bgr"]

        st.subheader("Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            depth_vis = _to_display_gray_u8(depth_idx)
            st.image(depth_vis, caption="Depth index (grayscale)", clamp=True, use_column_width=True)

        with col2:
            conf_vis = (np.clip(confidence, 0.0, 1.0) * 255.0).astype(np.uint8)
            st.image(conf_vis, caption="Confidence map", clamp=True, use_column_width=True)

        with col3:
            st.image(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB), caption="Color heatmap", use_column_width=True)

        if enable_cal and "calibration" in st.session_state:
            cal_params = st.session_state["calibration"]
            calibrated_mm = _calibrate_depth_index(
                depth_idx,
                skin_ref_idx=cal_params["skin_ref_idx"],
                scale_mm_per_step=cal_params["scale_mm_per_step"],
            )
            st.session_state["calibrated_depth_mm"] = calibrated_mm
            st.write(f"**Calibrated depth (mm):** {_format_depth_display(calibrated_mm)}")

        skin_ref = _estimate_skin_reference_index(depth_idx)
        signed_height = _signed_height_map(depth_idx, invert=invert, skin_reference=skin_ref)

        if show_3d:
            st.subheader("3D Lesion Depth Heatmap")
            st.caption("Dark red = deepest, bright blue = near skin surface, green = above skin surface.")
            use_calibrated_3d = enable_cal and "calibrated_depth_mm" in st.session_state
            if use_calibrated_3d:
                calibrated_mm = st.session_state["calibrated_depth_mm"]
                z_min = float(np.nanmin(calibrated_mm))
                z_max = float(np.nanmax(calibrated_mm))
                fig_3d = _render_3d_lesion_plot(
                    calibrated_mm,
                    sample_step=sample_step,
                    z_scale=z_scale,
                    depth_min=z_min,
                    depth_max=z_max,
                    is_calibrated=True,
                )
            else:
                fig_3d = _render_3d_lesion_plot(signed_height, sample_step=sample_step, z_scale=z_scale, is_calibrated=False)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.session_state["fig_3d"] = fig_3d
            st.session_state["signed_height"] = signed_height

    if save_clicked:
        if "depth_idx" not in st.session_state:
            st.error("Run SFF first before saving outputs.")
        else:
            depth_idx = st.session_state["depth_idx"]
            confidence = st.session_state["confidence"]
            heatmap_bgr = st.session_state["heatmap_bgr"]

            outputs = Path("outputs")
            confidence_u8 = (np.clip(confidence, 0.0, 1.0) * 255.0).astype(np.uint8)

            ok1 = save_image(outputs / "depth_index.png", depth_idx)
            ok2 = save_image(outputs / "confidence.png", confidence_u8)
            ok3 = save_image(outputs / "heatmap.png", heatmap_bgr)

            fig_3d = st.session_state.get("fig_3d")
            ok4 = True
            if fig_3d is not None:
                try:
                    fig_3d.write_image(outputs / "heatmap_3d.png", width=1200, height=900)
                except Exception:
                    ok4 = False

            ok5 = True
            if "calibrated_depth_mm" in st.session_state:
                try:
                    cal_map = st.session_state["calibrated_depth_mm"]
                    cal_uint8 = np.clip((cal_map + 10) / 20 * 255, 0, 255).astype(np.uint8)
                    ok5 = save_image(outputs / "calibrated_depth_mm.png", cal_uint8)
                except OSError:
                    ok5 = False

            if ok1 and ok2 and ok3 and ok4 and ok5:
                st.success("Saved outputs to ./outputs/")
            else:
                st.error("Failed to save one or more output images.")


if __name__ == "__main__":
    main()
