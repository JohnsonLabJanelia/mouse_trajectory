#!/usr/bin/env python3
"""
Head direction analysis from snout + left/right ears (data3D.csv).

Uses the triangle (Snout, EarL, EarR) to compute head direction angle in the camera (u,v) plane.
Outputs: trajectory_analysis/<animal>/head_direction/ including average head-direction heatmaps
by phase (early, mid, late).

Requires data3D.csv per trial (Snout, EarL, EarR) and calibration for 3D→2D projection.
"""

from pathlib import Path
import argparse
import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir / "predictions3D"))

import analyze_trajectories as at
from plot_trajectory_on_frame import load_calib, project_3d_to_2d
from plot_trajectory_xy import parse_data3d_csv

# Trial folder name pattern
TRIAL_PATTERN = re.compile(r"^Predictions_3D_trial_(\d+)_(\d+)-(\d+)$")

# Reward frame lookup: import from midline-and-goals pipeline when available
def _get_reward_frame_by_trial(
    trials_list: list[tuple[Path, str, str | None]],
    reward_times_path: Path,
    logs_dir: Path | None,
    animal: str,
) -> dict[str, int]:
    """Return dict trial_id -> reward_frame. Uses plot_flow_field_rory helpers if available."""
    reward_frame_by_trial: dict[str, int] = {}
    try:
        from plot_flow_field_rory import _get_reward_frame_from_csv, _get_reward_frame_per_trial
    except ImportError:
        return reward_frame_by_trial
    reward_times_path = Path(reward_times_path)
    if reward_times_path.is_file():
        reward_frame_by_trial = _get_reward_frame_from_csv(trials_list, reward_times_path)
    if not reward_frame_by_trial and logs_dir is not None and Path(logs_dir).is_dir():
        reward_frame_by_trial = _get_reward_frame_per_trial(trials_list, Path(logs_dir), animal)
    return reward_frame_by_trial


def get_animal_trials(predictions_dir: Path, animal: str) -> list[tuple[Path, str, str | None]]:
    """Return list of (csv_path, trial_id, session_folder) for the given animal."""
    animal_lower = animal.strip().lower()
    all_trials = at.find_trajectory_csvs(predictions_dir)
    return [t for t in all_trials if t[2] and t[2].lower().startswith(f"{animal_lower}_")]


def _split_phase_trials(
    trials_list: list[tuple[Path, str, str | None]],
) -> tuple[list[tuple[Path, str, str | None]], list[tuple[Path, str, str | None]], list[tuple[Path, str, str | None]]]:
    """Return (early, mid, late) thirds when ordered chronologically (session_folder, trial_id)."""
    if not trials_list:
        return [], [], []
    ordered = sorted(trials_list, key=lambda t: ((t[2] or ""), t[1]))
    n = len(ordered)
    k = max(1, n // 3)
    early = ordered[:k]
    mid = ordered[k : 2 * k] if 2 * k <= n else []
    late = ordered[2 * k :] if 2 * k < n else []
    return (early, mid, late)


def _resolve_calib_path(trial_dir: Path, calib_root: Path | None, camera: str) -> Path | None:
    """Resolve calibration path for a trial: info.yaml dataset_name under calib_root, or direct path."""
    info_path = trial_dir / "info.yaml"
    if not info_path.exists():
        return None
    try:
        import yaml
        with open(info_path) as f:
            info = yaml.safe_load(f)
    except Exception:
        return None
    dataset_name = info.get("dataset_name")
    if not dataset_name:
        return None
    # If dataset_name is already an absolute path, use it
    base = Path(dataset_name)
    if base.is_absolute():
        calib_path = base / f"{camera}.yaml"
    elif calib_root is not None:
        calib_path = Path(calib_root) / dataset_name / f"{camera}.yaml"
    else:
        calib_path = base / f"{camera}.yaml"
    return calib_path.resolve() if calib_path.exists() else None


def _head_angle_deg_from_triangle(snout_uv: np.ndarray, earL_uv: np.ndarray, earR_uv: np.ndarray) -> np.ndarray:
    """Head direction angle in degrees (0 = right/u+, 90 = down/v+ in image coords).
    Angle from ear midpoint toward snout in the u-v plane."""
    mid_u = (earL_uv[:, 0] + earR_uv[:, 0]) / 2
    mid_v = (earL_uv[:, 1] + earR_uv[:, 1]) / 2
    du = snout_uv[:, 0] - mid_u
    dv = snout_uv[:, 1] - mid_v
    rad = np.arctan2(dv, du)
    return np.degrees(rad)


def _body_angle_deg_from_tail_ear_mid(tail_uv: np.ndarray, earL_uv: np.ndarray, earR_uv: np.ndarray) -> np.ndarray:
    """Body direction angle in degrees (0 = right, 90 = down). From tail toward ear midpoint."""
    mid_u = (earL_uv[:, 0] + earR_uv[:, 0]) / 2
    mid_v = (earL_uv[:, 1] + earR_uv[:, 1]) / 2
    du = mid_u - tail_uv[:, 0]
    dv = mid_v - tail_uv[:, 1]
    rad = np.arctan2(dv, du)
    return np.degrees(rad)


def load_head_direction_per_trial(
    csv_path: Path,
    trial_dir: Path,
    calib: dict,
    frame_start: int,
    min_confidence: float = 0.15,
) -> pd.DataFrame | None:
    """
    Load data3D.csv for trial, project Snout/EarL/EarR to uv, compute head angle per frame.
    Returns DataFrame with columns: frame_number, u, v, head_angle_deg (only rows with valid triangle).
    """
    data3d_path = trial_dir / "data3D.csv"
    if not data3d_path.exists():
        return None
    body_parts = parse_data3d_csv(data3d_path)
    for key in ["Snout", "EarL", "EarR"]:
        if key not in body_parts:
            return None
    n_rows = len(body_parts["Snout"])
    frame_numbers = frame_start + np.arange(n_rows, dtype=np.int64)

    snout_xyz = body_parts["Snout"][["x", "y", "z"]].values.astype(float)
    earL_xyz = body_parts["EarL"][["x", "y", "z"]].values.astype(float)
    earR_xyz = body_parts["EarR"][["x", "y", "z"]].values.astype(float)
    conf_s = body_parts["Snout"]["confidence"].values.astype(float)
    conf_l = body_parts["EarL"]["confidence"].values.astype(float)
    conf_r = body_parts["EarR"]["confidence"].values.astype(float)

    keep = (conf_s >= min_confidence) & (conf_l >= min_confidence) & (conf_r >= min_confidence)
    if not np.any(keep):
        return None
    snout_uv = project_3d_to_2d(snout_xyz[keep], calib)
    earL_uv = project_3d_to_2d(earL_xyz[keep], calib)
    earR_uv = project_3d_to_2d(earR_xyz[keep], calib)
    head_angle = _head_angle_deg_from_triangle(snout_uv, earL_uv, earR_uv)
    frames = frame_numbers[keep]
    return pd.DataFrame({
        "frame_number": frames,
        "u": snout_uv[:, 0],
        "v": snout_uv[:, 1],
        "head_angle_deg": head_angle,
    })


def load_body_direction_per_trial(
    csv_path: Path,
    trial_dir: Path,
    calib: dict,
    frame_start: int,
    min_confidence: float = 0.15,
) -> pd.DataFrame | None:
    """
    Load data3D.csv for trial, project Tail/EarL/EarR to uv; body direction = tail → ear midpoint.
    Returns DataFrame with columns: frame_number, u, v, body_angle_deg. (u,v) = tail position.
    """
    data3d_path = trial_dir / "data3D.csv"
    if not data3d_path.exists():
        return None
    body_parts = parse_data3d_csv(data3d_path)
    for key in ["Tail", "EarL", "EarR"]:
        if key not in body_parts:
            return None
    n_rows = len(body_parts["Tail"])
    frame_numbers = frame_start + np.arange(n_rows, dtype=np.int64)
    tail_xyz = body_parts["Tail"][["x", "y", "z"]].values.astype(float)
    earL_xyz = body_parts["EarL"][["x", "y", "z"]].values.astype(float)
    earR_xyz = body_parts["EarR"][["x", "y", "z"]].values.astype(float)
    conf_t = body_parts["Tail"]["confidence"].values.astype(float)
    conf_l = body_parts["EarL"]["confidence"].values.astype(float)
    conf_r = body_parts["EarR"]["confidence"].values.astype(float)
    keep = (conf_t >= min_confidence) & (conf_l >= min_confidence) & (conf_r >= min_confidence)
    if not np.any(keep):
        return None
    tail_uv = project_3d_to_2d(tail_xyz[keep], calib)
    earL_uv = project_3d_to_2d(earL_xyz[keep], calib)
    earR_uv = project_3d_to_2d(earR_xyz[keep], calib)
    body_angle = _body_angle_deg_from_tail_ear_mid(tail_uv, earL_uv, earR_uv)
    frames = frame_numbers[keep]
    return pd.DataFrame({
        "frame_number": frames,
        "u": tail_uv[:, 0],
        "v": tail_uv[:, 1],
        "body_angle_deg": body_angle,
    })


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees; returns NaN if empty."""
    if len(angles_deg) == 0:
        return np.nan
    rad = np.deg2rad(angles_deg)
    return np.degrees(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad))))


def compute_phase_head_direction_grid(
    trials_list: list[tuple[Path, str, str | None]],
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    calib_root: Path | None,
    camera: str,
    min_confidence: float = 0.15,
    reward_frame_by_trial: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For all trials, load (u, v, head_angle) restricted to trajectory_filtered frames; optionally
    restrict to start→reward (frame_number <= reward_frame) when reward_frame_by_trial is provided.
    Returns (mean_angle_grid_deg, count_grid). Grid shape (len(v_edges)-1, len(u_edges)-1).
    """
    n_u, n_v = len(u_edges) - 1, len(v_edges) - 1
    sum_sin = np.zeros((n_v, n_u))
    sum_cos = np.zeros((n_v, n_u))
    count = np.zeros((n_v, n_u))

    for csv_path, trial_id, session_folder in trials_list:
        trial_dir = csv_path.parent
        m = TRIAL_PATTERN.match(trial_id)
        frame_start = int(m.group(2)) if m else 0
        calib_path = _resolve_calib_path(trial_dir, calib_root, camera)
        if calib_path is None or not calib_path.exists():
            continue
        try:
            calib = load_calib(calib_path)
        except Exception:
            continue
        hd_df = load_head_direction_per_trial(csv_path, trial_dir, calib, frame_start, min_confidence)
        if hd_df is None or len(hd_df) == 0:
            continue
        # Restrict to frames that appear in trajectory_filtered (valid path)
        traj = at.load_trajectory_csv(csv_path)
        if len(traj) == 0 or "u" not in traj.columns or "v" not in traj.columns:
            continue
        valid_frames = set(traj["frame_number"].astype(int).tolist())
        hd_df = hd_df[hd_df["frame_number"].isin(valid_frames)]
        # Optionally restrict to start → reward only
        if reward_frame_by_trial is not None:
            reward_frame = reward_frame_by_trial.get(trial_id)
            if reward_frame is None:
                continue
            hd_df = hd_df[hd_df["frame_number"] <= reward_frame]
        if len(hd_df) == 0:
            continue
        u_vals = hd_df["u"].values
        v_vals = hd_df["v"].values
        rad = np.deg2rad(hd_df["head_angle_deg"].values)
        iu = np.searchsorted(u_edges, u_vals, side="right") - 1
        iv = np.searchsorted(v_edges, v_vals, side="right") - 1
        iu = np.clip(iu, 0, n_u - 1)
        iv = np.clip(iv, 0, n_v - 1)
        for idx in range(len(hd_df)):
            jv, ju = iv[idx], iu[idx]
            sum_sin[jv, ju] += np.sin(rad[idx])
            sum_cos[jv, ju] += np.cos(rad[idx])
            count[jv, ju] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_angle_rad = np.arctan2(sum_sin, sum_cos)
    mean_angle_deg = np.degrees(mean_angle_rad)
    mean_angle_deg[count == 0] = np.nan
    return mean_angle_deg, count


def compute_phase_body_direction_grid(
    trials_list: list[tuple[Path, str, str | None]],
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    calib_root: Path | None,
    camera: str,
    min_confidence: float = 0.15,
    reward_frame_by_trial: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as compute_phase_head_direction_grid but for body (tail → ear mid). Returns (mean_angle_deg, count)."""
    n_u, n_v = len(u_edges) - 1, len(v_edges) - 1
    sum_sin = np.zeros((n_v, n_u))
    sum_cos = np.zeros((n_v, n_u))
    count = np.zeros((n_v, n_u))
    for csv_path, trial_id, session_folder in trials_list:
        trial_dir = csv_path.parent
        m = TRIAL_PATTERN.match(trial_id)
        frame_start = int(m.group(2)) if m else 0
        calib_path = _resolve_calib_path(trial_dir, calib_root, camera)
        if calib_path is None or not calib_path.exists():
            continue
        try:
            calib = load_calib(calib_path)
        except Exception:
            continue
        bd_df = load_body_direction_per_trial(csv_path, trial_dir, calib, frame_start, min_confidence)
        if bd_df is None or len(bd_df) == 0:
            continue
        traj = at.load_trajectory_csv(csv_path)
        if len(traj) == 0 or "u" not in traj.columns or "v" not in traj.columns:
            continue
        valid_frames = set(traj["frame_number"].astype(int).tolist())
        bd_df = bd_df[bd_df["frame_number"].isin(valid_frames)]
        if reward_frame_by_trial is not None:
            reward_frame = reward_frame_by_trial.get(trial_id)
            if reward_frame is None:
                continue
            bd_df = bd_df[bd_df["frame_number"] <= reward_frame]
        if len(bd_df) == 0:
            continue
        u_vals = bd_df["u"].values
        v_vals = bd_df["v"].values
        rad = np.deg2rad(bd_df["body_angle_deg"].values)
        iu = np.searchsorted(u_edges, u_vals, side="right") - 1
        iv = np.searchsorted(v_edges, v_vals, side="right") - 1
        iu = np.clip(iu, 0, n_u - 1)
        iv = np.clip(iv, 0, n_v - 1)
        for idx in range(len(bd_df)):
            jv, ju = iv[idx], iu[idx]
            sum_sin[jv, ju] += np.sin(rad[idx])
            sum_cos[jv, ju] += np.cos(rad[idx])
            count[jv, ju] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_angle_rad = np.arctan2(sum_sin, sum_cos)
    mean_angle_deg = np.degrees(mean_angle_rad)
    mean_angle_deg[count == 0] = np.nan
    return mean_angle_deg, count


def _cyclic_cmap_for_angle():
    """Cyclic colormap: 90° (down) = red, -90° (up) = blue; 0° and ±180° = yellow/cyan for clarity."""
    from matplotlib.colors import LinearSegmentedColormap
    # With norm vmin=-180, vmax=180: norm gives 0=-180°, 0.25=-90°, 0.5=0°, 0.75=90°, 1=180°
    # 5 evenly spaced stops: -90° = blue, 90° = red
    cmap = LinearSegmentedColormap.from_list("head_angle_rb", [
        (0.2, 0.8, 0.8),   # -180° cyan
        (0.0, 0.2, 1.0),   # -90° blue
        (1.0, 1.0, 0.2),   # 0° yellow
        (1.0, 0.0, 0.0),   # 90° red
        (0.2, 0.8, 0.8),   # 180° cyan (wraps)
    ], N=256)
    return cmap


def plot_head_direction_by_phase(
    early_trials: list[tuple[Path, str, str | None]],
    mid_trials: list[tuple[Path, str, str | None]],
    late_trials: list[tuple[Path, str, str | None]],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    calib_root: Path | None,
    camera: str,
    out_path: Path,
    animal: str = "",
    params: dict | None = None,
    n_bins: int = 40,
    reward_frame_by_trial: dict[str, int] | None = None,
    title_suffix: str = "",
) -> None:
    """
    Three panels: early, mid, late. Each shows (u,v) heatmap colored by mean head direction (cyclic).
    If reward_frame_by_trial is provided, only frames from start to reward are included; otherwise full path.
    title_suffix: e.g. " (full path)" or " (start to reward)".
    """
    u_edges = np.linspace(u_min, u_max, n_bins + 1)
    v_edges = np.linspace(v_min, v_max, n_bins + 1)
    # Larger figure and fonts
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    font_title = 16
    font_axis = 14
    font_ticks = 12
    phase_lists = [early_trials, mid_trials, late_trials]
    phase_names = ["Early", "Mid", "Late"]
    norm = plt.Normalize(vmin=-180, vmax=180)
    cmap = _cyclic_cmap_for_angle()

    for ax, trials, name in zip(axes, phase_lists, phase_names):
        ax.tick_params(axis="both", labelsize=font_ticks)
        if not trials:
            ax.set_xlim(u_min, u_max)
            ax.set_ylim(v_max, v_min)
            ax.set_aspect("equal")
            ax.set_title(f"{name} (n=0)", fontsize=font_title)
            ax.set_xlabel("u (px)", fontsize=font_axis)
            ax.set_ylabel("v (px)", fontsize=font_axis)
            continue
        mean_angle, count = compute_phase_head_direction_grid(
            trials, u_edges, v_edges, calib_root, camera,
            reward_frame_by_trial=reward_frame_by_trial,
        )
        count_min = max(3, np.nanpercentile(count[count > 0], 5) if (count > 0).any() else 3)
        mean_angle_masked = np.where(count >= count_min, mean_angle, np.nan)
        ax.pcolormesh(
            u_edges, v_edges, mean_angle_masked,
            cmap=cmap, norm=norm, shading="flat", alpha=0.9,
        )
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)", fontsize=font_axis)
        ax.set_ylabel("v (px)", fontsize=font_axis)
        ax.set_title(f"{name} (n={len(trials)} trials)", fontsize=font_title)
        if params:
            v_mid = params.get("v_mid")
            if v_mid is not None:
                ax.axhline(float(v_mid), color="black", linewidth=1.5, zorder=5)
            for key, color in [("goal1", "#e63946"), ("goal2", "#1d3557")]:
                gu, gv = params.get(f"{key}_u"), params.get(f"{key}_v")
                if gu is not None and gv is not None:
                    ax.plot(float(gu), float(gv), "o", color=color, markersize=10, markeredgecolor="white", zorder=6)

    # Leave space on the right for colorbar so it doesn't overlap the plot
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    cax = fig.add_axes([0.90, 0.12, 0.022, 0.76])  # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Head direction (°)")
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.ax.tick_params(labelsize=font_ticks)
    cbar.set_label("Head direction (°)", fontsize=font_axis)
    title = "Head direction by phase (snout–ear triangle)"
    if animal:
        title += f" — {animal}"
    if title_suffix:
        title += title_suffix
    fig.suptitle(title, fontsize=font_title + 2, y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_body_direction_by_phase(
    early_trials: list[tuple[Path, str, str | None]],
    mid_trials: list[tuple[Path, str, str | None]],
    late_trials: list[tuple[Path, str, str | None]],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    calib_root: Path | None,
    camera: str,
    out_path: Path,
    animal: str = "",
    params: dict | None = None,
    n_bins: int = 40,
    reward_frame_by_trial: dict[str, int] | None = None,
    title_suffix: str = "",
) -> None:
    """Three panels: early, mid, late. Each shows (u,v) heatmap colored by mean body direction (tail → ear mid)."""
    u_edges = np.linspace(u_min, u_max, n_bins + 1)
    v_edges = np.linspace(v_min, v_max, n_bins + 1)
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    font_title, font_axis, font_ticks = 16, 14, 12
    phase_lists = [early_trials, mid_trials, late_trials]
    phase_names = ["Early", "Mid", "Late"]
    norm = plt.Normalize(vmin=-180, vmax=180)
    cmap = _cyclic_cmap_for_angle()
    for ax, trials, name in zip(axes, phase_lists, phase_names):
        ax.tick_params(axis="both", labelsize=font_ticks)
        if not trials:
            ax.set_xlim(u_min, u_max)
            ax.set_ylim(v_max, v_min)
            ax.set_aspect("equal")
            ax.set_title(f"{name} (n=0)", fontsize=font_title)
            ax.set_xlabel("u (px)", fontsize=font_axis)
            ax.set_ylabel("v (px)", fontsize=font_axis)
            continue
        mean_angle, count = compute_phase_body_direction_grid(
            trials, u_edges, v_edges, calib_root, camera,
            reward_frame_by_trial=reward_frame_by_trial,
        )
        count_min = max(3, np.nanpercentile(count[count > 0], 5) if (count > 0).any() else 3)
        mean_angle_masked = np.where(count >= count_min, mean_angle, np.nan)
        ax.pcolormesh(
            u_edges, v_edges, mean_angle_masked,
            cmap=cmap, norm=norm, shading="flat", alpha=0.9,
        )
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)", fontsize=font_axis)
        ax.set_ylabel("v (px)", fontsize=font_axis)
        ax.set_title(f"{name} (n={len(trials)} trials)", fontsize=font_title)
        if params:
            v_mid = params.get("v_mid")
            if v_mid is not None:
                ax.axhline(float(v_mid), color="black", linewidth=1.5, zorder=5)
            for key, color in [("goal1", "#e63946"), ("goal2", "#1d3557")]:
                gu, gv = params.get(f"{key}_u"), params.get(f"{key}_v")
                if gu is not None and gv is not None:
                    ax.plot(float(gu), float(gv), "o", color=color, markersize=10, markeredgecolor="white", zorder=6)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    cax = fig.add_axes([0.90, 0.12, 0.022, 0.76])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Body direction (°)")
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.ax.tick_params(labelsize=font_ticks)
    cbar.set_label("Body direction (°)", fontsize=font_axis)
    title = "Body direction by phase (tail → ear midpoint)"
    if animal:
        title += f" — {animal}"
    if title_suffix:
        title += title_suffix
    fig.suptitle(title, fontsize=font_title + 2, y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_head_direction_angle_reference(
    out_path: Path,
    direction_label: str = "Head direction",
    sublabel: str = "snout–ear triangle",
) -> None:
    """
    Draw a reference diagram: which angle (degrees) corresponds to which direction in the image.
    direction_label and sublabel allow reuse for body (tail → ear midpoint).
    Convention: 0° = right (u+), 90° = down (v+).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.set_aspect("equal")
    # Image convention: u horizontal (right = +), v vertical (down = +)
    # So in math coords we plot u → x, v → y with y inverted for "image": y_down = -v so down = +v in image = -y in plot
    # Actually for the reference we just draw a circle and show angles; no need to invert. Use standard math: angle 0 = right (1,0), 90 = up (0,1).
    # Our head angle: atan2(dv, du) so 0° = (du>0, dv=0) = right, 90° = (du=0, dv>0) = down in image.
    # In a typical plot, x = u (right), y = v. If we use ax.invert_yaxis() then up = small v, down = large v. So 90° (down) = arrow pointing in +v = downward in plot after invert = upward in data. So let's not invert: just use a compass where 0° is right, 90° is downward in the figure (positive y).
    center = 0.0
    radius = 1.0
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k-", linewidth=1)
    ax.plot(center, center, "ko", markersize=6)
    # Arrows: angle in degrees -> direction. Our convention: head_angle = atan2(dv, du), so 0° = (1,0), 90° = (0,1)
    # In plot coordinates: x = u, y = v. So 0° -> (1,0) right; 90° -> (0,1) up in plot. But in image v increases downward! So 90° in our convention = (0,1) in (du,dv) = down in image = (0, +1) in (u,v). So in plot (x,y)=(u,v), 90° is positive y = upward in plot. So "down in image" = "up in plot" if we use y=v. So we need to say "90° = down (in image)" and draw the arrow in the direction of (0, +1) in (u,v) = (0, +1) in plot = upward. So arrow at 90° goes upward in the figure, and we label it "90° = down (v+ in image)".
    arrow_length = 0.85
    for deg, label, color in [
        (0, "0°  right (u+)", "green"),
        (90, "90°  down (v+)", "blue"),
        (180, "±180°  left", "red"),
        (-90, "-90°  up (v−)", "purple"),
    ]:
        rad = np.deg2rad(deg)
        dx = arrow_length * np.cos(rad)
        dy = arrow_length * np.sin(rad)
        ax.arrow(center, center, dx, dy, head_width=0.12, head_length=0.08, fc=color, ec=color, linewidth=2)
        # Label at arrow tip (slightly beyond)
        text_dist = 1.05
        ax.text(text_dist * np.cos(rad), text_dist * np.sin(rad), label, ha="center", va="center", fontsize=12, color=color)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("u (pixels) →", fontsize=12)
    ax.set_ylabel("v (pixels) →", fontsize=12)
    ax.set_title(f"{direction_label} angle reference (camera image coordinates)", fontsize=14)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
    note = (
        f"{direction_label} = {sublabel}.\n"
        "0° = points right (increasing u); 90° = points down (increasing v).\n"
        "Angles in [-180°, 180°]. Same convention used in heatmaps and analyses."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_head_body_vector_diagram(out_path: Path) -> None:
    """
    Draw a schematic of head vs body direction vectors from one frame.
    Shows: Tail, Ear midpoint, Snout; body vector (tail → ear mid); head vector (ear mid → snout).
    Uses example positions so body ≈ 0°, head ≈ 53° (camera convention: 0° = right, 90° = down).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect("equal")
    # Example positions in (u, v) style: u = x (right), v = y (down in image)
    tail_u, tail_v = 0.0, 0.0
    ear_mid_u, ear_mid_v = 2.0, 0.0
    snout_u, snout_v = 2.6, 0.85
    # Body vector: tail → ear mid
    body_du = ear_mid_u - tail_u
    body_dv = ear_mid_v - tail_v
    body_angle_deg = float(np.degrees(np.arctan2(body_dv, body_du)))
    # Head vector: ear mid → snout
    head_du = snout_u - ear_mid_u
    head_dv = snout_v - ear_mid_v
    head_angle_deg = float(np.degrees(np.arctan2(head_dv, head_du)))
    # Points
    ax.plot(tail_u, tail_v, "ko", markersize=12, zorder=5)
    ax.plot(ear_mid_u, ear_mid_v, "ko", markersize=10, zorder=5)
    ax.plot(snout_u, snout_v, "ko", markersize=12, zorder=5)
    ax.text(tail_u - 0.35, tail_v, "Tail", fontsize=12, ha="right", va="center")
    ax.text(ear_mid_u, ear_mid_v - 0.35, "Ear mid\n(EarL+EarR)/2", fontsize=11, ha="center", va="top")
    ax.text(snout_u + 0.25, snout_v, "Snout", fontsize=12, ha="left", va="center")
    # Body direction vector (tail → ear mid)
    ax.arrow(
        tail_u, tail_v, body_du * 0.85, body_dv * 0.85,
        head_width=0.15, head_length=0.1, fc="#1d3557", ec="#1d3557", linewidth=3, zorder=4,
    )
    ax.text(tail_u + body_du * 0.45, tail_v + body_dv * 0.45 + 0.2, "Body direction\n(tail → ear mid)", fontsize=11, ha="center", color="#1d3557", fontweight="bold")
    # Head direction vector (ear mid → snout)
    ax.arrow(
        ear_mid_u, ear_mid_v, head_du * 0.95, head_dv * 0.95,
        head_width=0.15, head_length=0.1, fc="#e63946", ec="#e63946", linewidth=3, zorder=4,
    )
    ax.text(ear_mid_u + head_du * 0.6, ear_mid_v + head_dv * 0.6 + 0.15, "Head direction\n(ear mid → snout)", fontsize=11, ha="center", color="#e63946", fontweight="bold")
    # Angle labels
    ax.text(1.0, -0.5, f"Body angle = {body_angle_deg:.0f}° (0° = right)", fontsize=10, color="#1d3557")
    ax.text(2.4, 0.6, f"Head angle = {head_angle_deg:.0f}°", fontsize=10, color="#e63946")
    ax.set_xlabel("u (pixels) →", fontsize=12)
    ax.set_ylabel("v (pixels) →", fontsize=12)
    ax.set_title("Head direction vs body direction (example frame)", fontsize=14)
    ax.set_xlim(-0.8, 3.2)
    ax.set_ylim(-0.9, 1.8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    note = (
        "Body = angle from tail to ear midpoint. Head = angle from ear midpoint to snout.\n"
        "Same convention: 0° = right (u+), 90° = down (v+). |head − body| = head–body alignment."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------- Start-to-reward analyses (require reward_frame_by_trial and params) ----------

def _p_to_stars(p: float, bonferroni_alpha: float | None = None) -> str:
    """Return *, **, *** for p < 0.05, 0.01, 0.001 (vs alpha or bonferroni_alpha)."""
    alpha = bonferroni_alpha if bonferroni_alpha is not None else 0.05
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""


def _annotate_boxplot_significance(
    ax: plt.Axes,
    groups: list[np.ndarray],
    alpha: float = 0.05,
) -> None:
    """
    If significant, draw asterisks on the boxplot. groups = list of 1d arrays (2, 3, or 4).
    For 2 groups: Mann-Whitney; one asterisk centered above. For 3: Kruskal-Wallis + post-hoc
    Mann-Whitney with Bonferroni; bracket + asterisk for each significant pair.
    For 4+ groups: Kruskal-Wallis only; one asterisk centered above if significant.
    """
    groups = [np.asarray(g).flatten() for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    if any(len(g) < 2 for g in groups):
        return
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_annot = ylim[1] + 0.02 * y_range
    if len(groups) == 2:
        try:
            _, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            if p < alpha:
                stars = _p_to_stars(p)
                ax.text(1.5, y_annot, stars, ha="center", va="bottom", fontsize=12)
                ax.set_ylim(ylim[0], ylim[1] + 0.08 * y_range)
        except Exception:
            pass
        return
    if len(groups) > 3:
        try:
            stat, p_kw = stats.kruskal(*groups)
            if p_kw < alpha:
                stars = _p_to_stars(p_kw)
                n = len(groups)
                ax.text((1 + n) / 2, y_annot, stars, ha="center", va="bottom", fontsize=12)
                ax.set_ylim(ylim[0], ylim[1] + 0.08 * y_range)
        except Exception:
            pass
        return
    if len(groups) != 3:
        return
    try:
        stat, p_kw = stats.kruskal(*groups)
        if p_kw >= alpha:
            return
        bonferroni_alpha = alpha / 3
        pairs_sig: list[tuple[int, int, float]] = []
        for i in range(3):
            for j in range(i + 1, 3):
                if len(groups[i]) < 2 or len(groups[j]) < 2:
                    continue
                _, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                if p < bonferroni_alpha:
                    pairs_sig.append((i, j, p))
        if not pairs_sig:
            return
        # Draw bracket and asterisk for each significant pair; stack vertically if multiple
        n_pairs = len(pairs_sig)
        step = 0.04 * y_range
        for k, (i, j, p) in enumerate(pairs_sig):
            y = y_annot + k * step
            ax.plot([i + 1, i + 1, j + 1, j + 1], [y, y + step * 0.5, y + step * 0.5, y], "k-", lw=1)
            stars = _p_to_stars(p, bonferroni_alpha)
            ax.text((i + 1 + j + 1) / 2, y + step * 0.5, stars, ha="center", va="bottom", fontsize=10)
        ax.set_ylim(ylim[0], ylim[1] + 0.08 * y_range + n_pairs * step)
    except Exception:
        pass


def _parse_trial_frames(trial_id: str) -> tuple[int, int] | None:
    """Parse trial_id (e.g. Predictions_3D_trial_0000_11572-20491) -> (frame_start, frame_end)."""
    m = TRIAL_PATTERN.match(trial_id)
    return (int(m.group(2)), int(m.group(3))) if m else None


def _point_goal_region_local(u: float, v: float, params: dict) -> int:
    """Return 1 if (u,v) in goal1 rect, 2 if in goal2 rect, 0 otherwise. Uses params (half_u, top_bottom, etc.)."""
    v_mid = params.get("v_mid")
    g1_u, g1_v = params.get("goal1_u"), params.get("goal1_v")
    g2_u, g2_v = params.get("goal2_u"), params.get("goal2_v")
    if v_mid is None or g1_u is None or g2_u is None:
        return 0
    half_u = float(params.get("half_u", (params.get("goal2_u", 0) - params.get("goal1_u", 0)) or 50))
    top_bottom = float(params.get("top_bottom", v_mid - 100))
    top_top = float(params.get("top_top", v_mid - 5))
    bottom_bottom = float(params.get("bottom_bottom", v_mid + 5))
    bottom_top = float(params.get("bottom_top", v_mid + 100))
    if g1_v < v_mid:
        if top_bottom <= v <= top_top and abs(u - g1_u) <= half_u:
            return 1
        if bottom_bottom <= v <= bottom_top and abs(u - g2_u) <= half_u:
            return 2
    else:
        if bottom_bottom <= v <= bottom_top and abs(u - g1_u) <= half_u:
            return 1
        if top_bottom <= v <= top_top and abs(u - g2_u) <= half_u:
            return 2
    return 0


def _load_trial_start_to_reward_with_head(
    csv_path: Path,
    trial_id: str,
    reward_frame: int,
    trial_dir: Path,
    calib: dict,
    frame_start: int,
) -> pd.DataFrame | None:
    """Load trajectory start→reward and merge with head direction. Returns df with frame_number, u, v, head_angle_deg, movement_angle_deg (deg)."""
    df = at.load_trajectory_csv(csv_path)
    if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
        return None
    mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
    df = df.loc[mask].sort_values("frame_number").reset_index(drop=True)
    if len(df) < 2:
        return None
    hd_df = load_head_direction_per_trial(csv_path, trial_dir, calib, frame_start)
    if hd_df is None or len(hd_df) == 0:
        return None
    merged = df.merge(hd_df[["frame_number", "head_angle_deg"]], on="frame_number", how="inner")
    if len(merged) < 2:
        return None
    du = np.diff(merged["u"].values.astype(float))
    dv = np.diff(merged["v"].values.astype(float))
    movement_deg = np.degrees(np.arctan2(dv, du))
    movement_deg = np.concatenate([[movement_deg[0]], movement_deg])
    merged = merged.copy()
    merged["movement_angle_deg"] = movement_deg
    return merged


def _load_trial_start_to_reward_with_head_and_body(
    csv_path: Path,
    trial_id: str,
    reward_frame: int,
    trial_dir: Path,
    calib: dict,
    frame_start: int,
) -> pd.DataFrame | None:
    """Load trajectory start→reward with both head and body direction. Returns df with frame_number, u, v (snout), head_angle_deg, body_angle_deg."""
    df = at.load_trajectory_csv(csv_path)
    if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
        return None
    mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
    df = df.loc[mask].sort_values("frame_number").reset_index(drop=True)
    hd_df = load_head_direction_per_trial(csv_path, trial_dir, calib, frame_start)
    bd_df = load_body_direction_per_trial(csv_path, trial_dir, calib, frame_start)
    if hd_df is None or len(hd_df) == 0 or bd_df is None or len(bd_df) == 0:
        return None
    merged = df[["frame_number", "u", "v"]].merge(
        hd_df[["frame_number", "head_angle_deg"]], on="frame_number", how="inner"
    ).merge(
        bd_df[["frame_number", "body_angle_deg"]], on="frame_number", how="inner"
    )
    if len(merged) < 2:
        return None
    return merged


def _load_trial_start_to_reward_with_body(
    csv_path: Path,
    trial_id: str,
    reward_frame: int,
    trial_dir: Path,
    calib: dict,
    frame_start: int,
) -> pd.DataFrame | None:
    """Load trajectory start→reward and merge with body direction. Returns df with frame_number, u, v (snout from traj), body_angle_deg, movement_angle_deg."""
    df = at.load_trajectory_csv(csv_path)
    if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
        return None
    mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
    df = df.loc[mask].sort_values("frame_number").reset_index(drop=True)
    if len(df) < 2:
        return None
    bd_df = load_body_direction_per_trial(csv_path, trial_dir, calib, frame_start)
    if bd_df is None or len(bd_df) == 0:
        return None
    merged = df.merge(bd_df[["frame_number", "body_angle_deg"]], on="frame_number", how="inner")
    if len(merged) < 2:
        return None
    du = np.diff(merged["u"].values.astype(float))
    dv = np.diff(merged["v"].values.astype(float))
    movement_deg = np.degrees(np.arctan2(dv, du))
    movement_deg = np.concatenate([[movement_deg[0]], movement_deg])
    merged = merged.copy()
    merged["movement_angle_deg"] = movement_deg
    return merged


def _crossing_events_with_frame(
    df: pd.DataFrame,
    params: dict,
) -> list[tuple[int, float, float, int]]:
    """Return list of (frame_number, u, v, toward_goal) at each midline crossing. toward_goal 1 or 2."""
    v_mid = params["v_mid"]
    g1_v, g2_v = params.get("goal1_v"), params.get("goal2_v")
    if g1_v is None or g2_v is None:
        return []
    v_above = min(g1_v, g2_v)
    v_below = max(g1_v, g2_v)
    v_thresh_top = (v_mid + v_above) / 2.0
    v_thresh_bottom = (v_mid + v_below) / 2.0
    g1_is_top = g1_v < v_mid

    def side(v: float) -> int:
        if v <= v_thresh_top:
            return -1
        if v >= v_thresh_bottom:
            return 1
        return 0

    out: list[tuple[int, float, float, int]] = []
    v_vals = df["v"].values.astype(float)
    u_vals = df["u"].values.astype(float)
    frames = df["frame_number"].values.astype(int)
    last_side = side(float(v_vals[0]))
    for i in range(1, len(v_vals)):
        s = side(float(v_vals[i]))
        if s != 0 and last_side != 0 and s != last_side:
            toward = 1 if (s == -1 and g1_is_top) or (s == 1 and not g1_is_top) else 2
            out.append((int(frames[i]), float(u_vals[i]), float(v_vals[i]), toward))
        if s != 0:
            last_side = s
    return out


def _first_goal_entries(
    df: pd.DataFrame,
    params: dict,
) -> tuple[int | None, float, float, int | None, float, float]:
    """Return (frame_g1, u_g1, v_g1, frame_g2, u_g2, v_g2) for first entry to each goal. None if never."""
    f1, u1, v1 = None, np.nan, np.nan
    f2, u2, v2 = None, np.nan, np.nan
    for _, row in df.iterrows():
        lab = _point_goal_region_local(float(row["u"]), float(row["v"]), params)
        if lab == 1 and f1 is None:
            f1, u1, v1 = int(row["frame_number"]), float(row["u"]), float(row["v"])
        elif lab == 2 and f2 is None:
            f2, u2, v2 = int(row["frame_number"]), float(row["u"]), float(row["v"])
        if f1 is not None and f2 is not None:
            break
    return (f1, u1, v1, f2, u2, v2)


def _angle_to_goal_deg(u: float, v: float, g_u: float, g_v: float) -> float:
    """Angle in degrees from (u,v) toward goal (g_u, g_v). 0 = right, 90 = down."""
    return float(np.degrees(np.arctan2(g_v - v, g_u - u)))


def _run_all_head_direction_analyses(
    trials: list[tuple[Path, str, str | None]],
    early: list,
    mid: list,
    late: list,
    reward_frame_by_trial: dict[str, int],
    params: dict,
    calib_root: Path | None,
    camera: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_dir: Path,
    animal: str,
) -> None:
    """Run all 6 suggested head-direction analyses (start to reward only), save figures to out_dir."""
    trial_to_phase: dict[str, int] = {}
    for t in early:
        trial_to_phase[t[1]] = 0
    for t in mid:
        trial_to_phase[t[1]] = 1
    for t in late:
        trial_to_phase[t[1]] = 2
    phase_names = ["Early", "Mid", "Late"]
    g1_u, g1_v = params.get("goal1_u"), params.get("goal1_v")
    g2_u, g2_v = params.get("goal2_u"), params.get("goal2_v")
    v_mid = params.get("v_mid")

    # Collect per-trial data with head direction (start→reward)
    records: list[dict] = []
    crossing_head_angles: list[tuple[int, float, int]] = []  # phase_id, head_deg, toward_goal
    goal1_entry_heads: list[tuple[int, float]] = []
    goal2_entry_heads: list[tuple[int, float]] = []
    for csv_path, trial_id, _ in trials:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        calib_path = _resolve_calib_path(csv_path.parent, calib_root, camera)
        if calib_path is None or not calib_path.exists():
            continue
        try:
            calib = load_calib(calib_path)
        except Exception:
            continue
        df = _load_trial_start_to_reward_with_head(csv_path, trial_id, reward_frame, csv_path.parent, calib, frame_start)
        if df is None or len(df) < 2:
            continue
        phase_id = trial_to_phase.get(trial_id, 0)
        for _, row in df.iterrows():
            records.append({
                "trial_id": trial_id,
                "phase_id": phase_id,
                "frame": int(row["frame_number"]),
                "u": float(row["u"]),
                "v": float(row["v"]),
                "head_deg": float(row["head_angle_deg"]),
                "movement_deg": float(row["movement_angle_deg"]),
            })
        for (frame, u, v, toward) in _crossing_events_with_frame(df, params):
            head_row = df[df["frame_number"] == frame]
            if len(head_row) > 0:
                crossing_head_angles.append((phase_id, float(head_row["head_angle_deg"].iloc[0]), toward))
        f1, u1, v1, f2, u2, v2 = _first_goal_entries(df, params)
        if f1 is not None:
            r = df[df["frame_number"] == f1]
            if len(r) > 0:
                goal1_entry_heads.append((phase_id, float(r["head_angle_deg"].iloc[0])))
        if f2 is not None:
            r = df[df["frame_number"] == f2]
            if len(r) > 0:
                goal2_entry_heads.append((phase_id, float(r["head_angle_deg"].iloc[0])))

    if not records:
        return
    rec_df = pd.DataFrame(records)

    # 1) Head direction at midline crossing
    if crossing_head_angles:
        fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
        by_phase: dict[int, list[float]] = {0: [], 1: [], 2: []}
        for pid, hd, _ in crossing_head_angles:
            by_phase[pid].append(hd)
        by_phase_list = [by_phase.get(i, []) for i in range(3)]
        axes[0].boxplot(by_phase_list, tick_labels=phase_names)
        _annotate_boxplot_significance(axes[0], by_phase_list)
        axes[0].set_ylabel("Head direction (°)")
        axes[0].set_title("Head direction at midline crossing (by phase)")
        toward1 = [hd for _, hd, t in crossing_head_angles if t == 1]
        toward2 = [hd for _, hd, t in crossing_head_angles if t == 2]
        axes[1].boxplot([toward1, toward2], tick_labels=["Toward goal 1", "Toward goal 2"])
        _annotate_boxplot_significance(axes[1], [np.array(toward1), np.array(toward2)])
        axes[1].set_ylabel("Head direction (°)")
        axes[1].set_title("Head direction at crossing (by direction)")
        fig1.suptitle(f"Head at midline crossing (start→reward) — {animal}", fontsize=12)
        plt.tight_layout()
        fig1.savefig(out_dir / "head_at_midline_crossing.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  head at midline crossing -> {out_dir / 'head_at_midline_crossing.png'}")

    # 2) Head direction when entering goal regions
    if goal1_entry_heads or goal2_entry_heads:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        g1_by_phase = {0: [], 1: [], 2: []}
        g2_by_phase = {0: [], 1: [], 2: []}
        for pid, hd in goal1_entry_heads:
            g1_by_phase[pid].append(hd)
        for pid, hd in goal2_entry_heads:
            g2_by_phase[pid].append(hd)
        g1_list = [g1_by_phase.get(i, []) for i in range(3)]
        g2_list = [g2_by_phase.get(i, []) for i in range(3)]
        axes[0].boxplot(g1_list, tick_labels=phase_names)
        _annotate_boxplot_significance(axes[0], [np.array(x) for x in g1_list])
        axes[0].set_ylabel("Head direction (°)")
        axes[0].set_title("Head at first entry to goal 1")
        axes[1].boxplot(g2_list, tick_labels=phase_names)
        _annotate_boxplot_significance(axes[1], [np.array(x) for x in g2_list])
        axes[1].set_ylabel("Head direction (°)")
        axes[1].set_title("Head at first entry to goal 2")
        fig2.suptitle(f"Head direction when entering goal (start→reward) — {animal}", fontsize=12)
        plt.tight_layout()
        fig2.savefig(out_dir / "head_when_entering_goal.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  head when entering goal -> {out_dir / 'head_when_entering_goal.png'}")

    # 3) Head–goal alignment heatmap (per bin: mean |head - dir_to_goal|, smaller = more aligned)
    n_bins = 30
    u_edges = np.linspace(u_min, u_max, n_bins + 1)
    v_edges = np.linspace(v_min, v_max, n_bins + 1)
    for goal_idx, (g_u, g_v) in enumerate([(g1_u, g1_v), (g2_u, g2_v)]):
        if g_u is None or g_v is None:
            continue
        align_grid = np.full((n_bins, n_bins), np.nan)
        count_grid = np.zeros((n_bins, n_bins))
        for _, row in rec_df.iterrows():
            u, v, head = row["u"], row["v"], row["head_deg"]
            dir_to_goal = _angle_to_goal_deg(u, v, g_u, g_v)
            diff = (head - dir_to_goal + 180) % 360 - 180
            align = abs(diff)
            iu = np.searchsorted(u_edges, u, side="right") - 1
            iv = np.searchsorted(v_edges, v, side="right") - 1
            iu = np.clip(iu, 0, n_bins - 1)
            iv = np.clip(iv, 0, n_bins - 1)
            if np.isnan(align_grid[iv, iu]):
                align_grid[iv, iu] = align
                count_grid[iv, iu] = 1
            else:
                align_grid[iv, iu] += align
                count_grid[iv, iu] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            align_grid = np.where(count_grid >= 3, align_grid / np.maximum(count_grid, 1), np.nan)
        fig3, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.pcolormesh(u_edges, v_edges, align_grid, cmap="viridis", shading="flat", vmin=0, vmax=90)
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        if v_mid is not None:
            ax.axhline(float(v_mid), color="black", linewidth=1)
        plt.colorbar(im, ax=ax, label="|Head − direction to goal| (°)")
        ax.set_title(f"Head–goal {goal_idx + 1} alignment (start→reward) — {animal}")
        plt.tight_layout()
        fig3.savefig(out_dir / f"head_goal_alignment_goal{goal_idx + 1}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  head–goal {goal_idx + 1} alignment -> {out_dir / f'head_goal_alignment_goal{goal_idx + 1}.png'}")

    # 4) Head vs movement direction (scatter / histogram of head - movement by phase)
    rec_df["head_movement_diff"] = (rec_df["head_deg"] - rec_df["movement_deg"] + 180) % 360 - 180
    fig4, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (ax, name) in enumerate(zip(axes, phase_names)):
        sub = rec_df[rec_df["phase_id"] == i]["head_movement_diff"]
        ax.hist(sub.dropna(), bins=36, range=(-180, 180), color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", label="0° (aligned)")
        ax.set_xlabel("Head − movement (°)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}")
    fig4.suptitle(f"Head vs movement direction (start→reward) — {animal}", fontsize=12)
    plt.tight_layout()
    fig4.savefig(out_dir / "head_vs_movement_direction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  head vs movement direction -> {out_dir / 'head_vs_movement_direction.png'}")

    # 5) Polar histograms by region and phase (3 phases × 4 regions)
    if v_mid is not None and g1_u is not None:
        def region(row: pd.Series) -> str:
            u, v = row["u"], row["v"]
            if _point_goal_region_local(u, v, params) == 1:
                return "Goal 1"
            if _point_goal_region_local(u, v, params) == 2:
                return "Goal 2"
            if v < v_mid:
                return "Above midline"
            if v >= v_mid:
                return "Below midline"
            return "Other"
        rec_df["region"] = rec_df.apply(region, axis=1)
        regions_order = ["Above midline", "Below midline", "Goal 1", "Goal 2"]
        phase_names_short = ["Early", "Mid", "Late"]
        fps = getattr(at, "FPS", 180)
        fig5, axes = plt.subplots(3, 4, subplot_kw=dict(projection="polar"), figsize=(16, 12))
        bins = np.linspace(-np.pi, np.pi, 37)
        for phase_id in range(3):
            for reg_idx, reg in enumerate(regions_order):
                ax = axes[phase_id, reg_idx]
                sub = rec_df[(rec_df["region"] == reg) & (rec_df["phase_id"] == phase_id)]["head_deg"]
                if len(sub) == 0:
                    ax.set_title(f"{phase_names_short[phase_id]}\n{reg}", fontsize=10)
                    continue
                rad = np.deg2rad(sub.values)
                weights = np.ones_like(rad) / fps
                ax.hist(rad, bins=bins, weights=weights, color="steelblue", alpha=0.7, edgecolor="white")
                ax.set_ylabel("Time (s)", fontsize=9)
                ax.set_rlabel_position(315)
                for label in ax.get_xticklabels():
                    if label.get_text().replace("°", "").strip() == "180":
                        label.set_text("")
                ax.set_title(f"{phase_names_short[phase_id]}\n{reg}", fontsize=10)
        fig5.suptitle(f"Head direction by region and phase (start→reward) — {animal} [radial = time (s), {fps} fps]", fontsize=12, y=1.02)
        plt.tight_layout()
        fig5.savefig(out_dir / "head_direction_polar_by_region.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  head polar by region (and phase) -> {out_dir / 'head_direction_polar_by_region.png'}")

    # 6) Time facing each goal (±θ°)
    theta_deg = 45
    if g1_u is not None and g2_u is not None:
        rec_df["dir_g1"] = rec_df.apply(lambda r: _angle_to_goal_deg(r["u"], r["v"], g1_u, g1_v), axis=1)
        rec_df["dir_g2"] = rec_df.apply(lambda r: _angle_to_goal_deg(r["u"], r["v"], g2_u, g2_v), axis=1)
        rec_df["diff_g1"] = (rec_df["head_deg"] - rec_df["dir_g1"] + 180) % 360 - 180
        rec_df["diff_g2"] = (rec_df["head_deg"] - rec_df["dir_g2"] + 180) % 360 - 180
        rec_df["facing_g1"] = rec_df["diff_g1"].abs() <= theta_deg
        rec_df["facing_g2"] = rec_df["diff_g2"].abs() <= theta_deg
        frac = rec_df.groupby("phase_id").agg({"facing_g1": "mean", "facing_g2": "mean"})
        frac = frac.reindex([0, 1, 2]).fillna(0)
        fig6, ax = plt.subplots(1, 1, figsize=(8, 5))
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w/2, frac["facing_g1"].values * 100, w, label=f"Facing goal 1 (±{theta_deg}°)", color="#e63946")
        ax.bar(x + w/2, frac["facing_g2"].values * 100, w, label=f"Facing goal 2 (±{theta_deg}°)", color="#1d3557")
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names)
        ax.set_ylabel("% of frames")
        ax.legend()
        ax.set_title(f"Time facing each goal (start→reward) — {animal}")
        plt.tight_layout()
        fig6.savefig(out_dir / "time_facing_each_goal.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  time facing each goal -> {out_dir / 'time_facing_each_goal.png'}")


def _run_all_body_direction_analyses(
    trials: list[tuple[Path, str, str | None]],
    early: list,
    mid: list,
    late: list,
    reward_frame_by_trial: dict[str, int],
    params: dict,
    calib_root: Path | None,
    camera: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_dir: Path,
    animal: str,
) -> None:
    """Run same 6 analyses as head but for body direction (start to reward only)."""
    trial_to_phase: dict[str, int] = {}
    for t in early:
        trial_to_phase[t[1]] = 0
    for t in mid:
        trial_to_phase[t[1]] = 1
    for t in late:
        trial_to_phase[t[1]] = 2
    phase_names = ["Early", "Mid", "Late"]
    g1_u, g1_v = params.get("goal1_u"), params.get("goal1_v")
    g2_u, g2_v = params.get("goal2_u"), params.get("goal2_v")
    v_mid = params.get("v_mid")
    records: list[dict] = []
    crossing_body_angles: list[tuple[int, float, int]] = []
    goal1_entry_body: list[tuple[int, float]] = []
    goal2_entry_body: list[tuple[int, float]] = []
    for csv_path, trial_id, _ in trials:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        calib_path = _resolve_calib_path(csv_path.parent, calib_root, camera)
        if calib_path is None or not calib_path.exists():
            continue
        try:
            calib = load_calib(calib_path)
        except Exception:
            continue
        df = _load_trial_start_to_reward_with_body(csv_path, trial_id, reward_frame, csv_path.parent, calib, frame_start)
        if df is None or len(df) < 2:
            continue
        phase_id = trial_to_phase.get(trial_id, 0)
        for _, row in df.iterrows():
            records.append({
                "trial_id": trial_id,
                "phase_id": phase_id,
                "frame": int(row["frame_number"]),
                "u": float(row["u"]),
                "v": float(row["v"]),
                "body_deg": float(row["body_angle_deg"]),
                "movement_deg": float(row["movement_angle_deg"]),
            })
        for (frame, u, v, toward) in _crossing_events_with_frame(df, params):
            body_row = df[df["frame_number"] == frame]
            if len(body_row) > 0:
                crossing_body_angles.append((phase_id, float(body_row["body_angle_deg"].iloc[0]), toward))
        f1, u1, v1, f2, u2, v2 = _first_goal_entries(df, params)
        if f1 is not None:
            r = df[df["frame_number"] == f1]
            if len(r) > 0:
                goal1_entry_body.append((phase_id, float(r["body_angle_deg"].iloc[0])))
        if f2 is not None:
            r = df[df["frame_number"] == f2]
            if len(r) > 0:
                goal2_entry_body.append((phase_id, float(r["body_angle_deg"].iloc[0])))
    if not records:
        return
    rec_df = pd.DataFrame(records)

    if crossing_body_angles:
        fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
        by_phase: dict[int, list[float]] = {0: [], 1: [], 2: []}
        for pid, bd, _ in crossing_body_angles:
            by_phase[pid].append(bd)
        by_phase_list = [by_phase.get(i, []) for i in range(3)]
        axes[0].boxplot(by_phase_list, tick_labels=phase_names)
        _annotate_boxplot_significance(axes[0], by_phase_list)
        axes[0].set_ylabel("Body direction (°)")
        axes[0].set_title("Body direction at midline crossing (by phase)")
        toward1 = [bd for _, bd, t in crossing_body_angles if t == 1]
        toward2 = [bd for _, bd, t in crossing_body_angles if t == 2]
        axes[1].boxplot([toward1, toward2], tick_labels=["Toward goal 1", "Toward goal 2"])
        _annotate_boxplot_significance(axes[1], [np.array(toward1), np.array(toward2)])
        axes[1].set_ylabel("Body direction (°)")
        axes[1].set_title("Body at crossing (by direction)")
        fig1.suptitle(f"Body at midline crossing (start→reward) — {animal}", fontsize=12)
        plt.tight_layout()
        fig1.savefig(out_dir / "body_at_midline_crossing.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  body at midline crossing -> {out_dir / 'body_at_midline_crossing.png'}")

    if goal1_entry_body or goal2_entry_body:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        g1_by_phase = {0: [], 1: [], 2: []}
        g2_by_phase = {0: [], 1: [], 2: []}
        for pid, bd in goal1_entry_body:
            g1_by_phase[pid].append(bd)
        for pid, bd in goal2_entry_body:
            g2_by_phase[pid].append(bd)
        g1_list = [g1_by_phase.get(i, []) for i in range(3)]
        g2_list = [g2_by_phase.get(i, []) for i in range(3)]
        axes[0].boxplot(g1_list, tick_labels=phase_names)
        _annotate_boxplot_significance(axes[0], [np.array(x) for x in g1_list])
        axes[0].set_ylabel("Body direction (°)")
        axes[0].set_title("Body at first entry to goal 1")
        axes[1].boxplot(g2_list, tick_labels=phase_names)
        _annotate_boxplot_significance(axes[1], [np.array(x) for x in g2_list])
        axes[1].set_ylabel("Body direction (°)")
        axes[1].set_title("Body at first entry to goal 2")
        fig2.suptitle(f"Body direction when entering goal (start→reward) — {animal}", fontsize=12)
        plt.tight_layout()
        fig2.savefig(out_dir / "body_when_entering_goal.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  body when entering goal -> {out_dir / 'body_when_entering_goal.png'}")

    n_bins = 30
    u_edges = np.linspace(u_min, u_max, n_bins + 1)
    v_edges = np.linspace(v_min, v_max, n_bins + 1)
    for goal_idx, (g_u, g_v) in enumerate([(g1_u, g1_v), (g2_u, g2_v)]):
        if g_u is None or g_v is None:
            continue
        align_grid = np.full((n_bins, n_bins), np.nan)
        count_grid = np.zeros((n_bins, n_bins))
        for _, row in rec_df.iterrows():
            u, v, body = row["u"], row["v"], row["body_deg"]
            dir_to_goal = _angle_to_goal_deg(u, v, g_u, g_v)
            diff = (body - dir_to_goal + 180) % 360 - 180
            align = abs(diff)
            iu = np.searchsorted(u_edges, u, side="right") - 1
            iv = np.searchsorted(v_edges, v, side="right") - 1
            iu = np.clip(iu, 0, n_bins - 1)
            iv = np.clip(iv, 0, n_bins - 1)
            if np.isnan(align_grid[iv, iu]):
                align_grid[iv, iu] = align
                count_grid[iv, iu] = 1
            else:
                align_grid[iv, iu] += align
                count_grid[iv, iu] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            align_grid = np.where(count_grid >= 3, align_grid / np.maximum(count_grid, 1), np.nan)
        fig3, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.pcolormesh(u_edges, v_edges, align_grid, cmap="viridis", shading="flat", vmin=0, vmax=90)
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        if v_mid is not None:
            ax.axhline(float(v_mid), color="black", linewidth=1)
        plt.colorbar(im, ax=ax, label="|Body − direction to goal| (°)")
        ax.set_title(f"Body–goal {goal_idx + 1} alignment (start→reward) — {animal}")
        plt.tight_layout()
        fig3.savefig(out_dir / f"body_goal_alignment_goal{goal_idx + 1}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  body–goal {goal_idx + 1} alignment -> {out_dir / f'body_goal_alignment_goal{goal_idx + 1}.png'}")

    rec_df["body_movement_diff"] = (rec_df["body_deg"] - rec_df["movement_deg"] + 180) % 360 - 180
    fig4, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (ax, name) in enumerate(zip(axes, phase_names)):
        sub = rec_df[rec_df["phase_id"] == i]["body_movement_diff"]
        ax.hist(sub.dropna(), bins=36, range=(-180, 180), color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", label="0° (aligned)")
        ax.set_xlabel("Body − movement (°)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}")
    fig4.suptitle(f"Body vs movement direction (start→reward) — {animal}", fontsize=12)
    plt.tight_layout()
    fig4.savefig(out_dir / "body_vs_movement_direction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  body vs movement direction -> {out_dir / 'body_vs_movement_direction.png'}")

    if v_mid is not None and g1_u is not None:
        def region(row: pd.Series) -> str:
            u, v = row["u"], row["v"]
            if _point_goal_region_local(u, v, params) == 1:
                return "Goal 1"
            if _point_goal_region_local(u, v, params) == 2:
                return "Goal 2"
            if v < v_mid:
                return "Above midline"
            if v >= v_mid:
                return "Below midline"
            return "Other"
        rec_df["region"] = rec_df.apply(region, axis=1)
        regions_order = ["Above midline", "Below midline", "Goal 1", "Goal 2"]
        phase_names_short = ["Early", "Mid", "Late"]
        fps = getattr(at, "FPS", 180)
        fig5, axes = plt.subplots(3, 4, subplot_kw=dict(projection="polar"), figsize=(16, 12))
        bins = np.linspace(-np.pi, np.pi, 37)
        for phase_id in range(3):
            for reg_idx, reg in enumerate(regions_order):
                ax = axes[phase_id, reg_idx]
                sub = rec_df[(rec_df["region"] == reg) & (rec_df["phase_id"] == phase_id)]["body_deg"]
                if len(sub) == 0:
                    ax.set_title(f"{phase_names_short[phase_id]}\n{reg}", fontsize=10)
                    continue
                rad = np.deg2rad(sub.values)
                weights = np.ones_like(rad) / fps
                ax.hist(rad, bins=bins, weights=weights, color="steelblue", alpha=0.7, edgecolor="white")
                ax.set_ylabel("Time (s)", fontsize=9)
                ax.set_rlabel_position(315)
                for label in ax.get_xticklabels():
                    if label.get_text().replace("°", "").strip() == "180":
                        label.set_text("")
                ax.set_title(f"{phase_names_short[phase_id]}\n{reg}", fontsize=10)
        fig5.suptitle(f"Body direction by region and phase (start→reward) — {animal} [radial = time (s), {fps} fps]", fontsize=12, y=1.02)
        plt.tight_layout()
        fig5.savefig(out_dir / "body_direction_polar_by_region.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  body polar by region -> {out_dir / 'body_direction_polar_by_region.png'}")

    theta_deg = 45
    if g1_u is not None and g2_u is not None:
        rec_df["dir_g1"] = rec_df.apply(lambda r: _angle_to_goal_deg(r["u"], r["v"], g1_u, g1_v), axis=1)
        rec_df["dir_g2"] = rec_df.apply(lambda r: _angle_to_goal_deg(r["u"], r["v"], g2_u, g2_v), axis=1)
        rec_df["diff_g1"] = (rec_df["body_deg"] - rec_df["dir_g1"] + 180) % 360 - 180
        rec_df["diff_g2"] = (rec_df["body_deg"] - rec_df["dir_g2"] + 180) % 360 - 180
        rec_df["facing_g1"] = rec_df["diff_g1"].abs() <= theta_deg
        rec_df["facing_g2"] = rec_df["diff_g2"].abs() <= theta_deg
        frac = rec_df.groupby("phase_id").agg({"facing_g1": "mean", "facing_g2": "mean"})
        frac = frac.reindex([0, 1, 2]).fillna(0)
        fig6, ax = plt.subplots(1, 1, figsize=(8, 5))
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w/2, frac["facing_g1"].values * 100, w, label=f"Facing goal 1 (±{theta_deg}°)", color="#e63946")
        ax.bar(x + w/2, frac["facing_g2"].values * 100, w, label=f"Facing goal 2 (±{theta_deg}°)", color="#1d3557")
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names)
        ax.set_ylabel("% of frames")
        ax.legend()
        ax.set_title(f"Time facing each goal (body) (start→reward) — {animal}")
        plt.tight_layout()
        fig6.savefig(out_dir / "time_facing_each_goal.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  time facing each goal (body) -> {out_dir / 'time_facing_each_goal.png'}")


def _build_body_vs_head_dataframe(
    trials: list[tuple[Path, str, str | None]],
    early: list,
    mid: list,
    late: list,
    reward_frame_by_trial: dict[str, int],
    params: dict | None,
    calib_root: Path | None,
    camera: str,
    fps: float = 180.0,
) -> pd.DataFrame:
    """Build one DataFrame with all start→reward frames: trial_id, phase_id, frame, u, v, head_deg, body_deg, time_sec, speed_px_s, region, head_body_diff, abs_head_body_diff."""
    trial_to_phase: dict[str, int] = {}
    for t in early:
        trial_to_phase[t[1]] = 0
    for t in mid:
        trial_to_phase[t[1]] = 1
    for t in late:
        trial_to_phase[t[1]] = 2
    v_mid = params.get("v_mid") if params else None
    g1_u = params.get("goal1_u") if params else None

    def region_row(u: float, v: float) -> str:
        if params is None or v_mid is None or g1_u is None:
            return "Other"
        lab = _point_goal_region_local(u, v, params)
        if lab == 1:
            return "Goal 1"
        if lab == 2:
            return "Goal 2"
        if v < v_mid:
            return "Above midline"
        return "Below midline"

    rows: list[dict] = []
    for csv_path, trial_id, _ in trials:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        calib_path = _resolve_calib_path(csv_path.parent, calib_root, camera)
        if calib_path is None or not calib_path.exists():
            continue
        try:
            calib = load_calib(calib_path)
        except Exception:
            continue
        df = _load_trial_start_to_reward_with_head_and_body(csv_path, trial_id, reward_frame, csv_path.parent, calib, frame_start)
        if df is None or len(df) < 2:
            continue
        phase_id = trial_to_phase.get(trial_id, 0)
        u_vals = df["u"].values.astype(float)
        v_vals = df["v"].values.astype(float)
        head_deg = df["head_angle_deg"].values.astype(float)
        body_deg = df["body_angle_deg"].values.astype(float)
        frame_nums = df["frame_number"].values.astype(int)
        speed = np.zeros(len(df))
        for i in range(1, len(df)):
            du = u_vals[i] - u_vals[i - 1]
            dv = v_vals[i] - v_vals[i - 1]
            speed[i] = np.sqrt(du * du + dv * dv) * fps
        for i in range(len(df)):
            hb = (head_deg[i] - body_deg[i] + 180) % 360 - 180
            abs_hb = min(abs(hb), 360 - abs(hb))
            rows.append({
                "trial_id": trial_id,
                "phase_id": phase_id,
                "frame": int(frame_nums[i]),
                "u": float(u_vals[i]),
                "v": float(v_vals[i]),
                "head_deg": float(head_deg[i]),
                "body_deg": float(body_deg[i]),
                "time_sec": (frame_nums[i] - frame_start) / fps,
                "speed_px_s": float(speed[i]),
                "region": region_row(float(u_vals[i]), float(v_vals[i])),
                "head_body_diff": hb,
                "abs_head_body_diff": abs_hb,
            })
    return pd.DataFrame(rows)


def run_body_vs_head_analysis(
    trials: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict | None,
    calib_root: Path | None,
    camera: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_dir: Path,
    animal: str,
    body_still_deg: float = 15.0,
    head_scanning_deg: float = 25.0,
    fps: float = 180.0,
    early: list | None = None,
    mid: list | None = None,
    late: list | None = None,
) -> None:
    """
    Find frames where body is still and head is scanning; save where/when.
    If early, mid, late are provided, also run: head-body yaw dist, heatmap mean |head-body|,
    by phase/region, % aligned vs scanning, scanning bouts, moving vs stationary, opposite condition.
    """
    all_u: list[float] = []
    all_v: list[float] = []
    all_time_sec: list[float] = []
    all_u_opp: list[float] = []
    all_v_opp: list[float] = []
    all_time_sec_opp: list[float] = []
    for csv_path, trial_id, _ in trials:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        calib_path = _resolve_calib_path(csv_path.parent, calib_root, camera)
        if calib_path is None or not calib_path.exists():
            continue
        try:
            calib = load_calib(calib_path)
        except Exception:
            continue
        df = _load_trial_start_to_reward_with_head_and_body(csv_path, trial_id, reward_frame, csv_path.parent, calib, frame_start)
        if df is None or len(df) < 3:
            continue
        head_deg = df["head_angle_deg"].values.astype(float)
        body_deg = df["body_angle_deg"].values.astype(float)
        u_vals = df["u"].values.astype(float)
        v_vals = df["v"].values.astype(float)
        frame_nums = df["frame_number"].values.astype(int)
        def wrap_deg(d: float) -> float:
            d = (d + 180) % 360 - 180
            return min(abs(d), 360 - abs(d))
        d_head = np.zeros(len(df))
        d_body = np.zeros(len(df))
        for i in range(1, len(df)):
            d_head[i] = wrap_deg(head_deg[i] - head_deg[i - 1])
            d_body[i] = wrap_deg(body_deg[i] - body_deg[i - 1])
        for i in range(1, len(df)):
            if d_body[i] < body_still_deg and d_head[i] > head_scanning_deg:
                all_u.append(u_vals[i])
                all_v.append(v_vals[i])
                all_time_sec.append((frame_nums[i] - frame_start) / fps)
            if d_body[i] > head_scanning_deg and d_head[i] < body_still_deg:
                all_u_opp.append(u_vals[i])
                all_v_opp.append(v_vals[i])
                all_time_sec_opp.append((frame_nums[i] - frame_start) / fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not all_u:
        print("  body_vs_head: no frames with head-scanning body-still; skipping where/when plots.")
    else:
        u_arr = np.array(all_u)
        v_arr = np.array(all_v)
        t_arr = np.array(all_time_sec)
        n_bins = 40
        u_edges = np.linspace(u_min, u_max, n_bins + 1)
        v_edges = np.linspace(v_min, v_max, n_bins + 1)
        H, _, _ = np.histogram2d(v_arr, u_arr, bins=[v_edges, u_edges])
        fig1, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.pcolormesh(u_edges, v_edges, H, cmap="hot", shading="flat")
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        if params:
            v_mid = params.get("v_mid")
            if v_mid is not None:
                ax.axhline(float(v_mid), color="black", linewidth=1)
            for key, color in [("goal1", "#e63946"), ("goal2", "#1d3557")]:
                gu, gv = params.get(f"{key}_u"), params.get(f"{key}_v")
                if gu is not None and gv is not None:
                    ax.plot(float(gu), float(gv), "o", color=color, markersize=10, markeredgecolor="white")
        ax.set_title(f"Where: head scanning while body still (n={len(all_u)} frames) — {animal}")
        plt.colorbar(im, ax=ax, label="Count")
        plt.tight_layout()
        fig1.savefig(out_dir / "where_head_scanning_body_still.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  body_vs_head where -> {out_dir / 'where_head_scanning_body_still.png'}")

        try:
            from plot_trajectory_on_frame import get_trial_frame_and_recording, load_frame_from_camera
            example_frame = None
            for csv_path, _trial_id, _ in trials:
                trial_dir = csv_path.parent
                try:
                    frame_start, recording_path = get_trial_frame_and_recording(trial_dir)
                except Exception:
                    continue
                if recording_path is None:
                    continue
                recording_path = Path(recording_path)
                if not (recording_path / f"{camera}.mp4").exists():
                    continue
                try:
                    example_frame = load_frame_from_camera(recording_path, camera, frame_start)
                except Exception:
                    continue
                break
            if example_frame is not None:
                H_img, W_img = example_frame.shape[:2]
                fig1b, ax = plt.subplots(1, 1, figsize=(14, 10))
                ax.imshow(example_frame, extent=(0, W_img, H_img, 0), origin="upper", aspect="auto")
                im_overlay = ax.pcolormesh(u_edges, v_edges, H, cmap="hot", shading="flat", alpha=0.5)
                ax.set_xlim(0, W_img)
                ax.set_ylim(H_img, 0)
                ax.set_xlabel("u (px)")
                ax.set_ylabel("v (px)")
                plt.colorbar(im_overlay, ax=ax, label="Count")
                if params:
                    v_mid = params.get("v_mid")
                    if v_mid is not None:
                        ax.axhline(float(v_mid), color="black", linewidth=1.5)
                    for key, color in [("goal1", "#e63946"), ("goal2", "#1d3557")]:
                        gu, gv = params.get(f"{key}_u"), params.get(f"{key}_v")
                        if gu is not None and gv is not None:
                            ax.plot(float(gu), float(gv), "o", color=color, markersize=12, markeredgecolor="white")
                ax.set_title(f"Where: head scanning while body still on example frame (n={len(all_u)} frames) — {animal}")
                plt.tight_layout()
                fig1b.savefig(out_dir / "where_head_scanning_body_still_on_frame.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head where on frame -> {out_dir / 'where_head_scanning_body_still_on_frame.png'}")
        except Exception as e:
            print(f"  body_vs_head where on frame skipped ({e})")

        fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(t_arr, bins=min(50, max(10, len(t_arr) // 5)), color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Time from trial start (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"When: head scanning while body still (n={len(all_u)} frames) — {animal}")
        plt.tight_layout()
        fig2.savefig(out_dir / "when_head_scanning_body_still.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  body_vs_head when -> {out_dir / 'when_head_scanning_body_still.png'}")

    phase_names = ["Early", "Mid", "Late"]
    trial_to_phase: dict[str, int] = {}
    if early is not None and mid is not None and late is not None:
        for t in early:
            trial_to_phase[t[1]] = 0
        for t in mid:
            trial_to_phase[t[1]] = 1
        for t in late:
            trial_to_phase[t[1]] = 2
    if early is not None and mid is not None and late is not None and params is not None:
        plot_head_body_vector_diagram(out_dir / "head_body_vector_diagram.png")
        print(f"  body_vs_head head/body vector diagram -> {out_dir / 'head_body_vector_diagram.png'}")
        df = _build_body_vs_head_dataframe(trials, early, mid, late, reward_frame_by_trial, params, calib_root, camera, fps)
        if len(df) == 0:
            print("  body_vs_head extended: no dataframe rows; skipping.")
        else:
            u_edges = np.linspace(u_min, u_max, 41)
            v_edges = np.linspace(v_min, v_max, 41)
            n_u, n_v = 40, 40
            v_mid = params.get("v_mid")

            # 1) Head–body yaw distribution: overall and by phase
            fig3, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes[0].hist(df["head_body_diff"], bins=36, range=(-180, 180), color="steelblue", edgecolor="white", alpha=0.8)
            axes[0].axvline(0, color="red", linestyle="--")
            axes[0].set_xlabel("Head − body (°)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Overall")
            for i, name in enumerate(phase_names):
                sub = df[df["phase_id"] == i]["head_body_diff"]
                axes[i + 1].hist(sub, bins=36, range=(-180, 180), color="steelblue", edgecolor="white", alpha=0.8)
                axes[i + 1].axvline(0, color="red", linestyle="--")
                axes[i + 1].set_xlabel("Head − body (°)")
                axes[i + 1].set_ylabel("Count")
                axes[i + 1].set_title(name)
            fig3.suptitle(f"Head–body angle (yaw) distribution — {animal}", fontsize=12)
            plt.tight_layout()
            fig3.savefig(out_dir / "head_body_yaw_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head head-body yaw dist -> {out_dir / 'head_body_yaw_distribution.png'}")

            # 2) Heatmap: mean |head − body| by (u,v)
            sum_abs = np.zeros((n_v, n_u))
            cnt = np.zeros((n_v, n_u))
            for _, row in df.iterrows():
                iu = np.searchsorted(u_edges, row["u"], side="right") - 1
                iv = np.searchsorted(v_edges, row["v"], side="right") - 1
                iu = np.clip(iu, 0, n_u - 1)
                iv = np.clip(iv, 0, n_v - 1)
                sum_abs[iv, iu] += row["abs_head_body_diff"]
                cnt[iv, iu] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_abs = np.where(cnt >= 3, sum_abs / np.maximum(cnt, 1), np.nan)
            fig4, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.pcolormesh(u_edges, v_edges, mean_abs, cmap="viridis", shading="flat", vmin=0, vmax=90)
            ax.set_xlim(u_min, u_max)
            ax.set_ylim(v_max, v_min)
            ax.set_aspect("equal")
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")
            if v_mid is not None:
                ax.axhline(float(v_mid), color="black", linewidth=1)
            for key, color in [("goal1", "#e63946"), ("goal2", "#1d3557")]:
                gu, gv = params.get(f"{key}_u"), params.get(f"{key}_v")
                if gu is not None and gv is not None:
                    ax.plot(float(gu), float(gv), "o", color=color, markersize=10, markeredgecolor="white")
            plt.colorbar(im, ax=ax, label="Mean |head − body| (°)")
            ax.set_title(f"Where head is most decoupled from body — {animal}")
            plt.tight_layout()
            fig4.savefig(out_dir / "where_mean_abs_head_body_diff.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head mean |head-body| heatmap -> {out_dir / 'where_mean_abs_head_body_diff.png'}")

            # 3) Boxplot |head − body| by phase
            fig5, ax = plt.subplots(1, 1, figsize=(8, 5))
            by_phase = [df[df["phase_id"] == i]["abs_head_body_diff"].values for i in range(3)]
            ax.boxplot(by_phase, tick_labels=phase_names)
            _annotate_boxplot_significance(ax, by_phase)
            ax.set_ylabel("|Head − body| (°)")
            ax.set_title(f"Head–body decoupling by phase — {animal}")
            plt.tight_layout()
            fig5.savefig(out_dir / "head_body_by_phase.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head by phase -> {out_dir / 'head_body_by_phase.png'}")

            # 4) Boxplot |head − body| by region
            regions_order = ["Above midline", "Below midline", "Goal 1", "Goal 2"]
            by_region = [df[df["region"] == r]["abs_head_body_diff"].values for r in regions_order]
            by_region = [x for x in by_region if len(x) > 0]
            labels = [r for r in regions_order if len(df[df["region"] == r]) > 0]
            if by_region:
                fig6, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.boxplot(by_region, tick_labels=labels)
                _annotate_boxplot_significance(ax, by_region)
                ax.set_ylabel("|Head − body| (°)")
                ax.set_title(f"Head–body decoupling by region — {animal}")
                plt.tight_layout()
                fig6.savefig(out_dir / "head_body_by_region.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head by region -> {out_dir / 'head_body_by_region.png'}")

            # 4b) Head–body alignment at first entry to each goal
            goal_entry_align: list[tuple[int, int, float]] = []  # phase_id, goal_idx (1 or 2), abs_head_body_deg
            for csv_path, trial_id, _ in trials:
                reward_frame = reward_frame_by_trial.get(trial_id)
                if reward_frame is None:
                    continue
                frames = _parse_trial_frames(trial_id)
                if not frames:
                    continue
                frame_start, _ = frames
                calib_path = _resolve_calib_path(csv_path.parent, calib_root, camera)
                if calib_path is None or not calib_path.exists():
                    continue
                try:
                    calib = load_calib(calib_path)
                except Exception:
                    continue
                tdf = _load_trial_start_to_reward_with_head_and_body(csv_path, trial_id, reward_frame, csv_path.parent, calib, frame_start)
                if tdf is None or len(tdf) < 2:
                    continue
                phase_id = trial_to_phase.get(trial_id, 0)
                f1, u1, v1, f2, u2, v2 = _first_goal_entries(tdf, params)
                for goal_idx, f in enumerate([f1, f2], start=1):
                    if f is None:
                        continue
                    row = tdf[tdf["frame_number"] == f]
                    if len(row) == 0:
                        continue
                    head_deg = float(row["head_angle_deg"].iloc[0])
                    body_deg = float(row["body_angle_deg"].iloc[0])
                    diff = (head_deg - body_deg + 180) % 360 - 180
                    abs_align = min(abs(diff), 360 - abs(diff))
                    goal_entry_align.append((phase_id, goal_idx, abs_align))
            if goal_entry_align:
                align_df = pd.DataFrame(goal_entry_align, columns=["phase_id", "goal_idx", "abs_head_body_deg"])
                fig4b, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, goal_idx in zip(axes, [1, 2]):
                    sub = align_df[align_df["goal_idx"] == goal_idx]
                    by_phase = [sub[sub["phase_id"] == i]["abs_head_body_deg"].values for i in range(3)]
                    ax.boxplot(by_phase, tick_labels=phase_names)
                    _annotate_boxplot_significance(ax, by_phase)
                    ax.set_ylabel("|Head − body| (°)")
                    ax.set_title(f"At first entry to goal {goal_idx}")
                fig4b.suptitle(f"Head–body alignment at first goal entry — {animal}", fontsize=12)
                plt.tight_layout()
                fig4b.savefig(out_dir / "head_body_alignment_at_goal_entry.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head head-body alignment at goal entry -> {out_dir / 'head_body_alignment_at_goal_entry.png'}")

            # 5) % time aligned vs scanning by phase (θ = 15°)
            theta_deg = 15
            df["aligned"] = df["abs_head_body_diff"] <= theta_deg
            frac_aligned = df.groupby("phase_id")["aligned"].mean()
            frac_aligned = frac_aligned.reindex([0, 1, 2]).fillna(0)
            fig7, ax = plt.subplots(1, 1, figsize=(8, 5))
            x = np.arange(3)
            ax.bar(x - 0.2, frac_aligned.values * 100, 0.4, label=f"Aligned (±{theta_deg}°)", color="green", alpha=0.8)
            ax.bar(x + 0.2, (1 - frac_aligned.values) * 100, 0.4, label=f"Scanning (|head−body| > {theta_deg}°)", color="orange", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(phase_names)
            ax.set_ylabel("% of frames")
            ax.legend()
            ax.set_title(f"Time aligned vs scanning (head–body) by phase — {animal}")
            plt.tight_layout()
            fig7.savefig(out_dir / "time_aligned_vs_scanning_by_phase.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head time aligned vs scanning -> {out_dir / 'time_aligned_vs_scanning_by_phase.png'}")

            # 6) Scanning bouts: contiguous frames with body still + head scanning
            bout_durations_sec: list[float] = []
            bout_count_per_trial: list[tuple[str, int, int]] = []  # trial_id, phase_id, count
            bout_count_per_trial_region: list[dict] = []  # trial_id, phase_id, region -> count (one row per trial per region)
            regions_order = ["Above midline", "Below midline", "Goal 1", "Goal 2"]

            def bout_region_name(u: float, v: float) -> str:
                lab = _point_goal_region_local(u, v, params)
                if lab == 1:
                    return "Goal 1"
                if lab == 2:
                    return "Goal 2"
                if v_mid is not None and v < v_mid:
                    return "Above midline"
                return "Below midline"

            for csv_path, trial_id, _ in trials:
                reward_frame = reward_frame_by_trial.get(trial_id)
                if reward_frame is None:
                    continue
                frames = _parse_trial_frames(trial_id)
                if not frames:
                    continue
                frame_start, _ = frames
                calib_path = _resolve_calib_path(csv_path.parent, calib_root, camera)
                if calib_path is None or not calib_path.exists():
                    continue
                try:
                    calib = load_calib(calib_path)
                except Exception:
                    continue
                tdf = _load_trial_start_to_reward_with_head_and_body(csv_path, trial_id, reward_frame, csv_path.parent, calib, frame_start)
                if tdf is None or len(tdf) < 3:
                    continue
                phase_id = trial_to_phase.get(trial_id, 0)
                head_deg = tdf["head_angle_deg"].values.astype(float)
                body_deg = tdf["body_angle_deg"].values.astype(float)
                u_vals = tdf["u"].values.astype(float)
                v_vals = tdf["v"].values.astype(float)
                def wrap_deg(d: float) -> float:
                    d = (d + 180) % 360 - 180
                    return min(abs(d), 360 - abs(d))
                is_scan = np.zeros(len(tdf), dtype=bool)
                for i in range(1, len(tdf)):
                    if wrap_deg(body_deg[i] - body_deg[i - 1]) < body_still_deg and wrap_deg(head_deg[i] - head_deg[i - 1]) > head_scanning_deg:
                        is_scan[i] = True
                i = 0
                count_bouts = 0
                region_counts: dict[str, int] = {r: 0 for r in regions_order}
                while i < len(tdf):
                    if not is_scan[i]:
                        i += 1
                        continue
                    j = i
                    while j < len(tdf) and is_scan[j]:
                        j += 1
                    dur_sec = (j - i) / fps
                    bout_durations_sec.append(dur_sec)
                    count_bouts += 1
                    reg = bout_region_name(float(u_vals[i]), float(v_vals[i]))
                    region_counts[reg] = region_counts.get(reg, 0) + 1
                    i = j
                bout_count_per_trial.append((trial_id, phase_id, count_bouts))
                for r in regions_order:
                    bout_count_per_trial_region.append({"trial_id": trial_id, "phase_id": phase_id, "region": r, "bout_count": region_counts[r]})
            if bout_durations_sec:
                fig8a, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.hist(bout_durations_sec, bins=min(40, max(10, len(bout_durations_sec) // 5)), color="steelblue", edgecolor="white", alpha=0.8)
                ax.set_xlabel("Bout duration (s)")
                ax.set_ylabel("Count")
                ax.set_title(f"Scanning bout duration (head scanning, body still) — {animal}")
                plt.tight_layout()
                fig8a.savefig(out_dir / "scanning_bout_duration_histogram.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head scanning bout duration -> {out_dir / 'scanning_bout_duration_histogram.png'}")
            bc_df = pd.DataFrame(bout_count_per_trial, columns=["trial_id", "phase_id", "bout_count"])
            fig8b, ax = plt.subplots(1, 1, figsize=(8, 5))
            by_phase_bouts = [bc_df[bc_df["phase_id"] == i]["bout_count"].values for i in range(3)]
            ax.boxplot(by_phase_bouts, tick_labels=phase_names)
            _annotate_boxplot_significance(ax, by_phase_bouts)
            ax.set_ylabel("Bouts per trial")
            ax.set_title(f"Scanning bout count per trial by phase — {animal}")
            plt.tight_layout()
            fig8b.savefig(out_dir / "scanning_bout_count_by_phase.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head scanning bout count -> {out_dir / 'scanning_bout_count_by_phase.png'}")

            bc_region_df = pd.DataFrame(bout_count_per_trial_region)
            fig8c, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for ax, reg in zip(axes, regions_order):
                sub = bc_region_df[bc_region_df["region"] == reg]
                by_phase = [sub[sub["phase_id"] == i]["bout_count"].values for i in range(3)]
                ax.boxplot(by_phase, tick_labels=phase_names)
                _annotate_boxplot_significance(ax, by_phase)
                ax.set_ylabel("Bouts per trial")
                ax.set_title(reg)
            fig8c.suptitle(f"Scanning bout count per trial by phase and region — {animal}", fontsize=12, y=1.02)
            plt.tight_layout()
            fig8c.savefig(out_dir / "scanning_bout_count_by_phase_and_region.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head scanning bout count by region -> {out_dir / 'scanning_bout_count_by_phase_and_region.png'}")

            # 6b) Scanning bout count by region only (all phases combined)
            by_region_only = [bc_region_df[bc_region_df["region"] == r]["bout_count"].values for r in regions_order]
            by_region_only = [x for x in by_region_only if len(x) > 0]
            labels_region = [r for r in regions_order if len(bc_region_df[bc_region_df["region"] == r]) > 0]
            if by_region_only:
                fig8d, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.boxplot(by_region_only, tick_labels=labels_region)
                _annotate_boxplot_significance(ax, by_region_only)
                ax.set_ylabel("Bouts per trial")
                ax.set_title(f"Scanning bout count per trial by region (all phases) — {animal}")
                plt.tight_layout()
                fig8d.savefig(out_dir / "scanning_bout_count_by_region_only.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head scanning bout count by region only -> {out_dir / 'scanning_bout_count_by_region_only.png'}")

            # 6c) Scanning bout count by phase, comparing regions (3 panels: Early, Mid, Late; each 4 regions)
            fig8e, axes = plt.subplots(1, 3, figsize=(14, 5))
            for ax, phase_id in zip(axes, range(3)):
                sub = bc_region_df[bc_region_df["phase_id"] == phase_id]
                by_region = [sub[sub["region"] == r]["bout_count"].values for r in regions_order]
                by_region = [x for x in by_region if len(x) > 0]
                labels_r = [r for r in regions_order if len(sub[sub["region"] == r]) > 0]
                if by_region:
                    ax.boxplot(by_region, tick_labels=labels_r)
                    _annotate_boxplot_significance(ax, by_region)
                ax.set_ylabel("Bouts per trial")
                ax.set_title(phase_names[phase_id])
            fig8e.suptitle(f"Scanning bout count per trial: by phase, comparing regions — {animal}", fontsize=12, y=1.02)
            plt.tight_layout()
            fig8e.savefig(out_dir / "scanning_bout_count_by_phase_comparing_regions.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head scanning bout count by phase (comparing regions) -> {out_dir / 'scanning_bout_count_by_phase_comparing_regions.png'}")

            # 7) Mean |head − body| when moving vs stationary (speed threshold)
            speed_thresh_px_s = 20.0
            df["moving"] = df["speed_px_s"] >= speed_thresh_px_s
            mean_moving = df[df["moving"]]["abs_head_body_diff"].mean()
            mean_stationary = df[~df["moving"]]["abs_head_body_diff"].mean()
            fig9, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.bar([0], [mean_stationary], 0.5, label="Stationary", color="gray", alpha=0.8)
            ax.bar([1], [mean_moving], 0.5, label="Moving", color="steelblue", alpha=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Stationary", "Moving"])
            ax.set_ylabel("Mean |Head − body| (°)")
            ax.set_title(f"Head–body decoupling: moving vs stationary (speed ≥ {speed_thresh_px_s} px/s) — {animal}")
            ax.legend()
            plt.tight_layout()
            fig9.savefig(out_dir / "head_body_moving_vs_stationary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  body_vs_head moving vs stationary -> {out_dir / 'head_body_moving_vs_stationary.png'}")

            # 8) Opposite: body turning, head stable — where and when
            if all_u_opp:
                u_opp = np.array(all_u_opp)
                v_opp = np.array(all_v_opp)
                t_opp = np.array(all_time_sec_opp)
                Hopp, _, _ = np.histogram2d(v_opp, u_opp, bins=[v_edges, u_edges])
                fig10a, ax = plt.subplots(1, 1, figsize=(10, 8))
                im = ax.pcolormesh(u_edges, v_edges, Hopp, cmap="coolwarm", shading="flat")
                ax.set_xlim(u_min, u_max)
                ax.set_ylim(v_max, v_min)
                ax.set_aspect("equal")
                ax.set_xlabel("u (px)")
                ax.set_ylabel("v (px)")
                if v_mid is not None:
                    ax.axhline(float(v_mid), color="black", linewidth=1)
                for key, color in [("goal1", "#e63946"), ("goal2", "#1d3557")]:
                    gu, gv = params.get(f"{key}_u"), params.get(f"{key}_v")
                    if gu is not None and gv is not None:
                        ax.plot(float(gu), float(gv), "o", color=color, markersize=10, markeredgecolor="white")
                ax.set_title(f"Where: body turning, head stable (n={len(all_u_opp)} frames) — {animal}")
                plt.colorbar(im, ax=ax, label="Count")
                plt.tight_layout()
                fig10a.savefig(out_dir / "where_body_turning_head_stable.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head where (body turn, head stable) -> {out_dir / 'where_body_turning_head_stable.png'}")
                fig10b, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.hist(t_opp, bins=min(50, max(10, len(t_opp) // 5)), color="coral", edgecolor="white", alpha=0.8)
                ax.set_xlabel("Time from trial start (s)")
                ax.set_ylabel("Count")
                ax.set_title(f"When: body turning, head stable (n={len(all_u_opp)} frames) — {animal}")
                plt.tight_layout()
                fig10b.savefig(out_dir / "when_body_turning_head_stable.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  body_vs_head when (body turn, head stable) -> {out_dir / 'when_body_turning_head_stable.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Head direction analysis (snout + EarL + EarR) by phase → trajectory_analysis/<animal>/head_direction/"
    )
    parser.add_argument("--animal", type=str, default="rory", help="Animal name (e.g. rory, wilfred)")
    parser.add_argument("--predictions-root", type=Path, default=Path("/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("trajectory_analysis"))
    parser.add_argument("--calib-root", type=Path, default=None, help="Root for per-date calibration (e.g. calib_params); each trial uses info.yaml dataset_name under this")
    parser.add_argument("--camera", type=str, default="Cam2005325")
    parser.add_argument("--midline-goals-json", type=Path, default=None, help="Optional midline_and_goals.json to overlay midline and goals on heatmaps")
    parser.add_argument("--reward-times", type=Path, default=None, help="CSV from cbot_climb_log/export_reward_times.py (default: <output-dir>/reward_times.csv)")
    parser.add_argument("--logs-dir", type=Path, default=None, help="cbot_climb_log logs dir; used for start→reward heatmap when --reward-times not provided or file missing")
    args = parser.parse_args()

    animal = args.animal.strip()
    predictions_root = Path(args.predictions_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_dir = out_root / animal / "head_direction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Angle reference diagram (what each degree value means in the image)
    ref_path = out_dir / "head_direction_angle_reference.png"
    plot_head_direction_angle_reference(ref_path)
    print(f"  head direction angle reference -> {ref_path}")

    trials = get_animal_trials(predictions_root, animal)
    if not trials:
        print(f"No {animal} trials found.")
        return
    early, mid, late = _split_phase_trials(trials)
    try:
        from plot_flow_field_rory import uv_limits
        u_min, u_max, v_min, v_max = uv_limits(trials)
    except Exception:
        all_u, all_v = [], []
        for csv_path, _, _ in trials:
            df = at.load_trajectory_csv(csv_path)
            if len(df) and "u" in df.columns and "v" in df.columns:
                all_u.extend(df["u"].tolist())
                all_v.extend(df["v"].tolist())
        u_min = min(all_u) if all_u else 0.0
        u_max = max(all_u) if all_u else 1000.0
        v_min = min(all_v) if all_v else 0.0
        v_max = max(all_v) if all_v else 1000.0
        du, dv = (u_max - u_min) or 1, (v_max - v_min) or 1
        u_min -= 0.05 * du
        u_max += 0.05 * du
        v_min -= 0.05 * dv
        v_max += 0.05 * dv

    params = None
    if getattr(args, "midline_goals_json", None) is not None and args.midline_goals_json is not None:
        jpath = Path(args.midline_goals_json).resolve()
        if jpath.is_file():
            with open(jpath) as f:
                params = json.load(f)
            print(f"Using midline + goals from {jpath}")
    if params is None and (out_root / animal / "midline_and_goals" / "midline_and_goals.json").is_file():
        with open(out_root / animal / "midline_and_goals" / "midline_and_goals.json") as f:
            params = json.load(f)
        print(f"Using midline + goals from {out_root / animal / 'midline_and_goals'}")

    calib_root = Path(args.calib_root).resolve() if args.calib_root else None
    reward_times_path = args.reward_times if args.reward_times is not None else out_root / "reward_times.csv"
    reward_times_path = Path(reward_times_path).resolve()
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else None
    reward_frame_by_trial = _get_reward_frame_by_trial(trials, reward_times_path, logs_dir, animal)
    if reward_frame_by_trial:
        print(f"  reward frame: {len(reward_frame_by_trial)}/{len(trials)} trials (for start→reward heatmap)")

    # Full path (all trajectory in trajectory_filtered)
    out_path_full = out_dir / "head_direction_by_phase.png"
    plot_head_direction_by_phase(
        early, mid, late,
        u_min, u_max, v_min, v_max,
        calib_root=calib_root,
        camera=args.camera,
        out_path=out_path_full,
        animal=animal,
        params=params,
        title_suffix=" (full path)",
    )
    print(f"  head direction by phase (full path) -> {out_path_full}")

    # Start to reward only (separate heatmap when reward times available)
    if reward_frame_by_trial:
        out_path_str = out_dir / "head_direction_by_phase_start_to_reward.png"
        plot_head_direction_by_phase(
            early, mid, late,
            u_min, u_max, v_min, v_max,
            calib_root=calib_root,
            camera=args.camera,
            out_path=out_path_str,
            animal=animal,
            params=params,
            reward_frame_by_trial=reward_frame_by_trial,
            title_suffix=" (start to reward)",
        )
        print(f"  head direction by phase (start to reward) -> {out_path_str}")
        if params is not None and "v_mid" in params and "goal1_u" in params:
            _run_all_head_direction_analyses(
                trials, early, mid, late,
                reward_frame_by_trial, params,
                calib_root=calib_root,
                camera=args.camera,
                u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max,
                out_dir=out_dir,
                animal=animal,
            )


if __name__ == "__main__":
    main()
