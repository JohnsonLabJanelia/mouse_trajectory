#!/usr/bin/env python3
"""
Overlay 2D snout trajectory on a video frame image.

Design:
  1. Load 3D points (x, y, z, confidence) from data3D.csv for Snout.
  2. Use camera calibration to project 3D → 2D for this camera (JARVIS formula).
  3. Filter points by confidence (default 0.15 for plotting; arena/regions use 0.3 elsewhere).
  4. Plot the filtered 2D trajectory on the frame image and save.

Arena mask and start/end regions are built separately (using confidence 0.3); this script
only plots the trajectory with confidence 0.15. Further filtering (arena, start/end) can
be added later.
"""

from pathlib import Path
import argparse
import re
import sys
import yaml

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir / "predictions3D"))
from plot_trajectory_xy import parse_data3d_csv


def get_trial_frame_and_recording(trial_dir: Path) -> tuple[int, Path | None]:
    """
    Parse frame_start from prediction folder name (e.g. ..._11572-20491 -> 11572)
    and recording_path from info.yaml.
    """
    match = re.search(r"_(\d+)-(\d+)$", trial_dir.name)
    if not match:
        raise ValueError(f"Cannot parse frame range from folder name: {trial_dir.name}")
    frame_start = int(match.group(1))

    info_path = trial_dir / "info.yaml"
    recording_path = None
    if info_path.exists():
        with open(info_path) as f:
            info = yaml.safe_load(f)
            if "recording_path" in info:
                recording_path = Path(info["recording_path"])
    return frame_start, recording_path


def load_frame_from_camera(recording_path: Path, camera: str, frame_index: int) -> np.ndarray:
    """Read a single frame from the camera video (e.g. recording_path/Cam2005325.mp4)."""
    video_path = recording_path / f"{camera}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Camera video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    # cv2 returns BGR; convert to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_continuous_segments(
    u: np.ndarray,
    v: np.ndarray,
    frame_indices: np.ndarray,
    jump_threshold_px: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split trajectory at large frame-to-frame steps (jumps).
    Returns list of (u_seg, v_seg, frame_indices_seg), each segment has no internal jump.
    """
    if len(u) < 2:
        return [(u, v, frame_indices)] if len(u) > 0 else []
    dist = np.sqrt(np.diff(u.astype(float)) ** 2 + np.diff(v.astype(float)) ** 2)
    is_jump = dist > jump_threshold_px
    segments = []
    start = 0
    for i in range(len(is_jump)):
        if is_jump[i]:
            # segment from start to i+1 (inclusive of point i, exclusive of i+1)
            segments.append((u[start : i + 1], v[start : i + 1], frame_indices[start : i + 1]))
            start = i + 1
    segments.append((u[start:], v[start:], frame_indices[start:]))
    return segments


def load_arena_mask(mask_path: Path) -> np.ndarray:
    """Load arena mask from .npz (keys: mask, shape_hw). Returns (H, W) uint8."""
    data = np.load(mask_path)
    mask = data["mask"]
    return mask


def load_arena_start_end(npz_path: Path) -> dict:
    """
    Load start/end regions from .npz (average_start, average_end, radius_start, radius_end).
    Returns dict with (u_center, v_center) and radius for each; points inside circle are in the region.
    """
    data = np.load(npz_path)
    return {
        "average_start": np.atleast_1d(data["average_start"]).astype(float),
        "average_end": np.atleast_1d(data["average_end"]).astype(float),
        "radius_start": float(data["radius_start"]),
        "radius_end": float(data["radius_end"]),
    }


def point_in_start_region(u: np.ndarray, v: np.ndarray, start_end: dict) -> np.ndarray:
    """Boolean array: True where (u, v) is inside the start circle (image coords)."""
    c = start_end["average_start"]
    r = start_end["radius_start"]
    d = np.sqrt((u - c[0]) ** 2 + (v - c[1]) ** 2)
    return d <= r


def point_in_end_region(u: np.ndarray, v: np.ndarray, start_end: dict) -> np.ndarray:
    """Boolean array: True where (u, v) is inside the end circle (image coords)."""
    c = start_end["average_end"]
    r = start_end["radius_end"]
    d = np.sqrt((u - c[0]) ** 2 + (v - c[1]) ** 2)
    return d <= r


def segment_has_points_in_start_or_end_region(
    u_seg: np.ndarray,
    v_seg: np.ndarray,
    start_end: dict,
) -> bool:
    """True if at least one point in the segment lies in the start circle or in the end circle (image coords u, v)."""
    c_start = start_end["average_start"]
    c_end = start_end["average_end"]
    r_start = start_end["radius_start"]
    r_end = start_end["radius_end"]
    d_to_start = np.sqrt((u_seg - c_start[0]) ** 2 + (v_seg - c_start[1]) ** 2)
    d_to_end = np.sqrt((u_seg - c_end[0]) ** 2 + (v_seg - c_end[1]) ** 2)
    in_start = np.any(d_to_start <= r_start)
    in_end = np.any(d_to_end <= r_end)
    return in_start or in_end


def load_calib(calib_path: Path) -> dict:
    """Load JARVIS-format calibration (intrinsicMatrix, distortionCoefficients, R, T)."""
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"Could not open calibration: {calib_path}")
    calib = {
        "intrinsicMatrix": fs.getNode("intrinsicMatrix").mat(),
        "distortionCoefficients": fs.getNode("distortionCoefficients").mat(),
        "R": fs.getNode("R").mat(),
        "T": fs.getNode("T").mat(),
    }
    fs.release()
    return calib


def project_3d_to_2d(points_3d: np.ndarray, calib: dict) -> np.ndarray:
    """
    Project Nx3 world points to Nx2 image (u, v) using JARVIS ReprojectionTool formula.
    Same as ReprojectionTool.reprojectPoint() in JARVIS-HybridNet/jarvis/utils/reprojection.py.
    """
    K = calib["intrinsicMatrix"]
    R = calib["R"]
    T = calib["T"]
    dist = calib["distortionCoefficients"].flatten()

    # cameraMatrix = transpose([R; T] @ K) -> (3, 4)
    RT = np.vstack([R, T.reshape(1, 3)])
    cameraMatrix = (RT @ K).T

    n = len(points_3d)
    points_hom = np.hstack([points_3d, np.ones((n, 1))])
    projected = points_hom @ cameraMatrix.T  # (N, 3): (u*z, v*z, z)

    z = projected[:, 2]
    z = np.where(z == 0, 1e-10, z)

    u_norm = projected[:, 0] / z - K[0, 2]
    v_norm = projected[:, 1] / z - K[1, 2]

    fx, fy = K[0, 0], K[1, 1]
    r2 = (u_norm / fx) ** 2 + (v_norm / fy) ** 2
    k1 = dist[0] if len(dist) > 0 else 0
    k2 = dist[1] if len(dist) > 1 else 0
    distort = 1 + (k1 + k2 * r2) * r2

    u = u_norm * distort + K[0, 2]
    v = v_norm * distort + K[1, 2]
    return np.column_stack([u, v])


def run(
    frame: np.ndarray,
    csv_path: Path,
    out_path: Path,
    calib_path: Path,
    *,
    arena_mask: np.ndarray | None = None,
    start_end: dict | None = None,
    jump_threshold_px: float = 100.0,
    min_segment_points: int = 50,
    min_confidence: float = 0.15,
    verbose: bool = True,
) -> None:
    """
    Load 3D snout points, project to 2D, filter by confidence, arena mask, jump segments, and start/end region; plot on frame, save.
    frame: RGB image (H, W, 3) from the camera for this trial.
    arena_mask: optional (H, W) uint8; only points inside mask (non-zero) are plotted.
    start_end: optional dict from load_arena_start_end(); only segments that start in start circle and end in end circle are kept.
    jump_threshold_px: split trajectory when consecutive points are farther than this (pixels).
    min_segment_points: drop segments with fewer points than this (drops short/jumpy fragments).
    """
    calib = load_calib(calib_path)
    img_h, img_w = frame.shape[:2]

    body_parts = parse_data3d_csv(csv_path)
    if "Snout" not in body_parts:
        if verbose:
            print("No 'Snout' in data3D.csv")
        Image.fromarray(frame).save(out_path)
        return

    snout = body_parts["Snout"]
    points_3d = snout[["x", "y", "z"]].values.astype(float)
    conf = snout["confidence"].values.astype(float)
    n_orig = len(points_3d)
    if n_orig == 0:
        if verbose:
            print("No Snout points in data3D.csv")
        Image.fromarray(frame).save(out_path)
        return

    # Project all 3D → 2D first (needed for start-region trim)
    points_2d = project_3d_to_2d(points_3d, calib)
    u = points_2d[:, 0]
    v = points_2d[:, 1]

    # Trim at start: drop leading points until first point in start region (trajectory must start there)
    if start_end is not None:
        in_start = point_in_start_region(u, v, start_end)
        first_in_start = np.where(in_start)[0]
        if len(first_in_start) == 0:
            if verbose:
                print("No point falls in start region; trajectory cannot start there. Saving frame only.")
            Image.fromarray(frame).save(out_path)
            return
        first_idx = int(first_in_start[0])
        points_3d = points_3d[first_idx:]
        conf = conf[first_idx:]
        u = u[first_idx:]
        v = v[first_idx:]
        if verbose:
            print(f"Start trim: dropped first {first_idx} points (not in start area), kept {len(points_3d)}")

        # Trim at end: keep only up to and including last point in end region
        in_end = point_in_end_region(u, v, start_end)
        last_in_end = np.where(in_end)[0]
        if len(last_in_end) > 0:
            last_idx = int(last_in_end[-1])
            n_before = len(points_3d)
            points_3d = points_3d[: last_idx + 1]
            conf = conf[: last_idx + 1]
            u = u[: last_idx + 1]
            v = v[: last_idx + 1]
            if verbose:
                print(f"End trim: dropped {n_before - (last_idx + 1)} points after last in end region, kept {len(points_3d)}")

    # Retain points with confidence >= min_confidence
    keep = conf >= min_confidence
    points_3d = points_3d[keep]
    u = u[keep]
    v = v[keep]
    conf = conf[keep]
    if verbose:
        n_drop = (~keep).sum()
        print(f"Confidence >= {min_confidence}: kept {len(points_3d)} points, dropped {n_drop}")

    if len(points_3d) == 0:
        if verbose:
            print("No points after confidence filter; saving frame only.")
        Image.fromarray(frame).save(out_path)
        return

    frame_indices = np.arange(len(points_3d))

    # Filter by arena mask: keep only points inside image and inside mask
    if arena_mask is not None:
        in_bounds = (
            (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        )
        u_safe = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v_safe = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        ui = np.clip(u_safe.astype(np.int32), 0, img_w - 1)
        vi = np.clip(v_safe.astype(np.int32), 0, img_h - 1)
        in_mask = (arena_mask[vi, ui] != 0) & in_bounds
        u = u[in_mask]
        v = v[in_mask]
        frame_indices = frame_indices[in_mask]
        if verbose:
            n_removed = np.sum(~in_mask)
            print(f"Arena mask: kept {len(u)} points, removed {n_removed}")
        if len(u) == 0:
            if verbose:
                print("No points inside arena; saving frame only.")
            Image.fromarray(frame).save(out_path)
            return

    # Split at jumps and keep only long enough segments (no jump artifacts)
    segments = get_continuous_segments(u, v, frame_indices, jump_threshold_px)
    kept = [s for s in segments if len(s[0]) >= min_segment_points]
    if not kept:
        if verbose:
            print(
                f"No segment with >={min_segment_points} points after jump filtering; saving frame only."
            )
            print(f"  Segments: {[len(s[0]) for s in segments]}")
        Image.fromarray(frame).save(out_path)
        return

    # Keep only segments that have at least one point in the start region or in the end region
    if start_end is not None:
        before = len(kept)
        kept = [s for s in kept if segment_has_points_in_start_or_end_region(s[0], s[1], start_end)]
        if verbose and len(kept) != before:
            print(f"Start/end filter: kept {len(kept)} segment(s), dropped {before - len(kept)} (segment has no points in start/end region)")
        if not kept:
            if verbose:
                print("No segment has any point in start or end region; saving frame only.")
            Image.fromarray(frame).save(out_path)
            return

    u = np.concatenate([s[0] for s in kept])
    v = np.concatenate([s[1] for s in kept])
    frame_indices = np.concatenate([s[2] for s in kept])
    if verbose:
        n_dropped = sum(len(s[0]) for s in segments) - len(u)
        print(
            f"Jump filter: kept {len(u)} points in {len(kept)} segment(s), dropped {n_dropped} (segments with <{min_segment_points} pts or jumps >{jump_threshold_px}px)"
        )

    # Matplotlib: image in pixel coords, trajectory overlaid (original style)
    fig, ax = plt.subplots(1, 1, figsize=(img_w / 100, img_h / 100))
    ax.imshow(frame, extent=[0, img_w, img_h, 0])  # (0,0) top-left, so y down
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    scatter = ax.scatter(u, v, c=frame_indices, cmap="viridis", s=2, alpha=0.7)
    for u_seg, v_seg, _ in kept:
        ax.plot(u_seg, v_seg, color="yellow", alpha=0.25, linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label="Frame index")
    ax.set_title("Snout trajectory (confidence >= " + str(min_confidence) + ")")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    if verbose:
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot snout trajectory on frame: 3D → 2D with calibration, filter by confidence, overlay."
    )
    parser.add_argument("trial_dir", type=Path, help="Trial folder (e.g. Predictions_3D_trial_0000_11572-20491)")
    parser.add_argument("--camera", type=str, required=True, help="Camera name (e.g. Cam2005325)")
    parser.add_argument("--calib-path", type=Path, default=None, help="Calibration YAML; default from trial info.yaml")
    parser.add_argument("--frame-path", type=Path, default=None, help="Frame image; default trial_dir/frame.png")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output image; default trial_dir/trajectory_on_frame.png")
    parser.add_argument("--arena-mask", type=Path, default=None, help="Arena mask .npz; default predictions3D/arena_mask.npz")
    parser.add_argument("--no-arena-mask", action="store_true", help="Do not filter by arena mask")
    parser.add_argument("--arena-start-end", type=Path, default=None, help="Start/end regions .npz; default predictions3D/arena_start_end.npz")
    parser.add_argument("--no-arena-start-end", action="store_true", help="Do not filter segments by start/end regions")
    parser.add_argument("--jump-threshold", type=float, default=100.0, help="Split at consecutive points farther than this (px); default 100")
    parser.add_argument("--min-segment-points", type=int, default=50, help="Drop segments with fewer points; default 50")
    parser.add_argument("--min-confidence", type=float, default=0.15, help="Min confidence to retain points (default 0.15)")
    args = parser.parse_args()

    trial_dir = args.trial_dir.resolve()
    if not trial_dir.is_dir():
        raise SystemExit(f"Not a directory: {trial_dir}")

    # Calibration
    calib_path = args.calib_path
    if calib_path is None:
        info_path = trial_dir / "info.yaml"
        if info_path.exists():
            import yaml
            with open(info_path) as f:
                info = yaml.safe_load(f)
                if "dataset_name" in info:
                    calib_path = Path(info["dataset_name"]) / f"{args.camera}.yaml"
    if calib_path is None or not Path(calib_path).exists():
        raise SystemExit("Calibration file not found; set --calib-path")
    calib_path = Path(calib_path).resolve()

    # Frame: --frame-path or trial_dir/frame.png (plot on top of that image)
    frame_path = args.frame_path or trial_dir / "frame.png"
    frame_path = Path(frame_path).resolve()
    if not frame_path.exists():
        raise SystemExit(f"Frame not found: {frame_path}")
    frame = np.array(Image.open(frame_path))

    # Output
    out_path = args.output or trial_dir / "trajectory_on_frame.png"
    out_path = Path(out_path)

    csv_path = trial_dir / "data3D.csv"
    if not csv_path.exists():
        raise SystemExit(f"data3D.csv not found: {csv_path}")

    arena_mask = None
    if not args.no_arena_mask:
        mask_path = args.arena_mask or _script_dir / "predictions3D" / "arena_mask.npz"
        mask_path = Path(mask_path).resolve()
        if mask_path.exists():
            arena_mask = load_arena_mask(mask_path)
            if arena_mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                raise SystemExit(
                    f"Arena mask shape {arena_mask.shape[:2]} does not match frame shape {frame.shape[:2]}"
                )
        elif args.arena_mask is not None:
            raise SystemExit(f"Arena mask not found: {mask_path}")

    start_end = None
    if not args.no_arena_start_end:
        start_end_path = args.arena_start_end or _script_dir / "predictions3D" / "arena_start_end.npz"
        start_end_path = Path(start_end_path).resolve()
        if start_end_path.exists():
            start_end = load_arena_start_end(start_end_path)
        elif args.arena_start_end is not None:
            raise SystemExit(f"Arena start/end not found: {start_end_path}")

    run(
        frame=frame,
        csv_path=csv_path,
        out_path=out_path,
        calib_path=calib_path,
        arena_mask=arena_mask,
        start_end=start_end,
        jump_threshold_px=args.jump_threshold,
        min_segment_points=args.min_segment_points,
        min_confidence=args.min_confidence,
        verbose=True,
    )


if __name__ == "__main__":
    main()
