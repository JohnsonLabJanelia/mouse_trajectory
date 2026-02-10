#!/usr/bin/env python3
"""
Study filtered trajectory CSVs: compute basic stats and make illustrative plots.

Reads trajectory_filtered.csv from each trial folder (frame_number, x, y, z, u, v, segment_id),
computes per-trial statistics (path length, duration, speed, segments), and produces
summary CSV + plots for exploration.

Outputs include flow_field_low_to_high.png: two panels (elevation + flow direction;
flow speed + flow direction). See docs/FLOW_FIELD_PLOTS.md for how these are computed.
"""

from pathlib import Path
import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def _kmeans2_xyz(pts: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """2-means in 3D so the 2 centroids end up on opposite ends. pts shape (N, 3). Returns (2, 3)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    mid = np.median(pts[:, 0])
    left = pts[pts[:, 0] <= mid]
    right = pts[pts[:, 0] > mid]
    if len(left) == 0:
        left = pts[:1]
    if len(right) == 0:
        right = pts[-1:]
    c0 = left.mean(axis=0)
    c1 = right.mean(axis=0)
    for _ in range(max_iter):
        d0 = np.linalg.norm(pts - c0, axis=1)
        d1 = np.linalg.norm(pts - c1, axis=1)
        assign = (d0 > d1).astype(int)
        n0 = (assign == 0).sum()
        n1 = (assign == 1).sum()
        if n0 == 0 or n1 == 0:
            break
        new_c0 = pts[assign == 0].mean(axis=0)
        new_c1 = pts[assign == 1].mean(axis=0)
        if np.allclose(new_c0, c0) and np.allclose(new_c1, c1):
            break
        c0, c1 = new_c0, new_c1
    return np.array([c0, c1])


_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
from plot_trajectory_on_frame import load_calib, project_3d_to_2d

# Exclude peak-elevation points with z above this (outlier/artifact)
MAX_PEAK_Z = 150.0  # Exclude outlier peak elevations (e.g. ~200) from peak-elevation plots

# In all_trials_xy plot: exclude points where y > Y_FOR_Z_CAP and z > Z_CAP_IN_Y_REGION (physically inconsistent)
Y_FOR_Z_CAP = -100.0
Z_CAP_IN_Y_REGION = 120.0

# Video frame rate for time-in-seconds (path to peak, duration, etc.)
FPS = 180


def find_trajectory_csvs(predictions_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (path_to_csv, trial_id) for each trial with trajectory_filtered.csv."""
    predictions_dir = Path(predictions_dir)
    out = []
    for d in sorted(predictions_dir.iterdir()):
        if not d.is_dir():
            continue
        if not re.match(r"Predictions_3D_trial_\d+_\d+-\d+$", d.name):
            continue
        csv_path = d / "trajectory_filtered.csv"
        if csv_path.exists():
            out.append((csv_path, d.name))
    return out


def load_trajectory_csv(csv_path: Path) -> pd.DataFrame:
    """Load trajectory CSV and drop points with negative elevation (z < 0)."""
    df = pd.read_csv(csv_path)
    if "z" in df.columns and len(df) > 0:
        df = df[df["z"] >= 0].copy()
    return df


def path_length_3d(df: pd.DataFrame) -> float:
    """Total 3D path length (sum of consecutive Euclidean distances in x, y, z)."""
    if len(df) < 2:
        return 0.0
    d = np.diff(df[["x", "y", "z"]].values.astype(float), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def _ordered_xy_path(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 2) array of (x, y) along path: segments in order, points by frame_number."""
    rows = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg = df[df["segment_id"] == seg_id].sort_values("frame_number")
        rows.append(seg[["x", "y"]].values.astype(float))
    if not rows:
        return np.zeros((0, 2))
    return np.vstack(rows)


def _ordered_xyz_path(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 3) array of (x, y, z) along path: segments in order, points by frame_number."""
    xyzf = _ordered_xyz_frame_path(df)
    return xyzf[:, :3] if len(xyzf) > 0 else np.zeros((0, 3))


def _ordered_xyz_frame_path(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 4) array of (x, y, z, frame_number) along path: segments in order, points by frame_number."""
    rows = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg = df[df["segment_id"] == seg_id].sort_values("frame_number")
        rows.append(seg[["x", "y", "z", "frame_number"]].values.astype(float))
    if not rows:
        return np.zeros((0, 4))
    return np.vstack(rows)


def _resample_path_waypoints(xy: np.ndarray, n_waypoints: int) -> np.ndarray:
    """Resample path to n_waypoints (x,y) by cumulative path length. xy shape (N, 2)."""
    if len(xy) < 2 or n_waypoints <= 0:
        return np.zeros((n_waypoints, 2))
    d = np.zeros(len(xy))
    d[1:] = np.cumsum(np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1)))
    total = d[-1]
    if total <= 0:
        return np.tile(xy[0], (n_waypoints, 1))
    out = []
    for i in range(n_waypoints):
        t = (i + 0.5) / n_waypoints * total
        idx = np.searchsorted(d, t, side="right") - 1
        idx = max(0, min(idx, len(d) - 2))
        w = (t - d[idx]) / (d[idx + 1] - d[idx] + 1e-12)
        pt = (1 - w) * xy[idx] + w * xy[idx + 1]
        out.append(pt)
    return np.array(out)


def _resample_path_waypoints_3d(xyz: np.ndarray, n_waypoints: int) -> np.ndarray:
    """Resample 3D path to n_waypoints (x,y,z) by cumulative 3D path length. xyz shape (N, 3)."""
    if len(xyz) < 2 or n_waypoints <= 0:
        return np.zeros((n_waypoints, 3))
    d = np.zeros(len(xyz))
    d[1:] = np.cumsum(np.sqrt(((np.diff(xyz, axis=0)) ** 2).sum(axis=1)))
    total = d[-1]
    if total <= 0:
        return np.tile(xyz[0], (n_waypoints, 1))
    out = []
    for i in range(n_waypoints):
        t = (i + 0.5) / n_waypoints * total
        idx = np.searchsorted(d, t, side="right") - 1
        idx = max(0, min(idx, len(d) - 2))
        w = (t - d[idx]) / (d[idx + 1] - d[idx] + 1e-12)
        pt = (1 - w) * xyz[idx] + w * xyz[idx + 1]
        out.append(pt)
    return np.array(out)


def compute_trial_stats(csv_path: Path, trial_id: str) -> dict:
    """Load one trajectory CSV and return a dict of summary stats (points with z >= 0 only)."""
    df = load_trajectory_csv(csv_path)
    if len(df) == 0:
        return {"trial_id": trial_id, "n_points": 0}

    frame = df["frame_number"].values.astype(int)
    n_points = len(df)
    frame_start = int(frame.min())
    frame_end = int(frame.max())
    duration_frames = frame_end - frame_start + 1  # span; actual points may be sparse

    path_len = path_length_3d(df)
    # Speed = path length per frame (units/frame); if frame rate known, can convert to units/sec
    mean_speed = path_len / max(1, n_points - 1)  # per step (consecutive points)
    n_segments = int(df["segment_id"].max()) + 1
    points_per_seg = df.groupby("segment_id").size()
    seg_lengths = points_per_seg.tolist()

    return {
        "trial_id": trial_id,
        "n_points": n_points,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "duration_frames": duration_frames,
        "path_length_3d": round(path_len, 4),
        "mean_speed_per_step": round(mean_speed, 4),
        "n_segments": n_segments,
        "min_segment_points": int(points_per_seg.min()) if len(points_per_seg) else 0,
        "max_segment_points": int(points_per_seg.max()) if len(points_per_seg) else 0,
    }


def aggregate_all_trials(predictions_dir: Path) -> pd.DataFrame:
    """Build a stats DataFrame for all trials with trajectory_filtered.csv."""
    rows = []
    for csv_path, trial_id in find_trajectory_csvs(predictions_dir):
        rows.append(compute_trial_stats(csv_path, trial_id))
    return pd.DataFrame(rows)


def compute_peak_points_and_path_to_peak(predictions_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each trial, compute peak point and path-to-peak; exclude only the peak point when z > MAX_PEAK_Z (outlier).
    Returns (peak_points_df, path_to_peak_df).
    """
    peak_rows = []
    path_to_peak_rows = []
    for csv_path, trial_id in find_trajectory_csvs(predictions_dir):
        df = load_trajectory_csv(csv_path)
        if len(df) < 1:
            continue
        idx_max = df["z"].idxmax()
        row_peak = df.loc[idx_max]
        if row_peak["z"] > MAX_PEAK_Z:
            continue  # exclude only this 3D point (outlier) from peak/path-to-peak outputs
        peak_rows.append({
            "trial_id": trial_id,
            "x": row_peak["x"],
            "y": row_peak["y"],
            "z": row_peak["z"],
            "frame_number": int(row_peak["frame_number"]),
        })
        pos_max = df.index.get_loc(idx_max)
        segment = df.iloc[0 : pos_max + 1]
        frame_start = int(segment["frame_number"].iloc[0])
        frame_peak = int(segment["frame_number"].iloc[-1])
        frames_to_peak = frame_peak - frame_start
        path_to_peak_rows.append({
            "trial_id": trial_id,
            "frame_start": frame_start,
            "frame_peak": frame_peak,
            "frames_to_peak": frames_to_peak,
            "seconds_to_peak": round(frames_to_peak / FPS, 4),
            "path_length_to_peak": round(path_length_3d(segment), 4),
            "n_points_to_peak": len(segment),
        })
    return pd.DataFrame(peak_rows), pd.DataFrame(path_to_peak_rows)


def _flow_field_two_panels(
    trials_list: list[tuple[Path, str]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Compute elevation + flow direction and flow speed + flow direction for the given trials, save two-panel figure to out_path. title_suffix is appended to panel titles."""
    if len(trials_list) == 0:
        return
    n_bins = 40
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    sum_z_elev = np.zeros((n_bins, n_bins))
    count_z = np.zeros((n_bins, n_bins))
    sum_dx = np.zeros((n_bins, n_bins))
    sum_dy = np.zeros((n_bins, n_bins))
    sum_w = np.zeros((n_bins, n_bins)) + 1e-12
    sum_z = np.zeros((n_bins, n_bins))
    sum_vel = np.zeros((n_bins, n_bins))
    for csv_path, _ in trials_list:
        df = load_trajectory_csv(csv_path)
        if len(df) < 2:
            continue
        mask = (df["z"] <= MAX_PEAK_Z) & ((df["y"] <= Y_FOR_Z_CAP) | (df["z"] <= Z_CAP_IN_Y_REGION))
        df = df.loc[mask]
        if len(df) < 2:
            continue
        xyzf = _ordered_xyz_frame_path(df)
        if len(xyzf) < 2:
            continue
        xyz = xyzf[:, :3]
        frames = xyzf[:, 3]
        for k in range(len(xyz)):
            px, py, pz = xyz[k, 0], xyz[k, 1], xyz[k, 2]
            ix = np.searchsorted(x_edges, px, side="right") - 1
            iy = np.searchsorted(y_edges, py, side="right") - 1
            ix = max(0, min(ix, n_bins - 1))
            iy = max(0, min(iy, n_bins - 1))
            sum_z_elev[iy, ix] += pz
            count_z[iy, ix] += 1
        for i in range(len(xyz) - 1):
            dx = xyz[i + 1, 0] - xyz[i, 0]
            dy = xyz[i + 1, 1] - xyz[i, 1]
            dz = xyz[i + 1, 2] - xyz[i, 2]
            dt = (frames[i + 1] - frames[i]) / FPS
            if dt <= 0:
                continue
            raw_vel = np.sqrt(dx * dx + dy * dy + dz * dz) / dt
            mid_x = (xyz[i, 0] + xyz[i + 1, 0]) / 2
            mid_y = (xyz[i, 1] + xyz[i + 1, 1]) / 2
            mid_z = (xyz[i, 2] + xyz[i + 1, 2]) / 2
            if mid_y > Y_FOR_Z_CAP and mid_z > Z_CAP_IN_Y_REGION:
                continue
            weight = max(0.0, dz)
            if weight <= 0:
                continue
            ix = np.searchsorted(x_edges, mid_x, side="right") - 1
            iy = np.searchsorted(y_edges, mid_y, side="right") - 1
            ix = max(0, min(ix, n_bins - 1))
            iy = max(0, min(iy, n_bins - 1))
            sum_dx[iy, ix] += dx * weight
            sum_dy[iy, ix] += dy * weight
            sum_w[iy, ix] += weight
            sum_z[iy, ix] += mid_z * weight
            sum_vel[iy, ix] += raw_vel * weight
    with np.errstate(divide="ignore", invalid="ignore"):
        u = sum_dx / sum_w
        v = sum_dy / sum_w
    mean_raw_vel = np.where(sum_w > 1e-10, sum_vel / sum_w, 0.0)
    elev_grid = np.where(count_z > 0, sum_z_elev / count_z, np.nan)
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    XX, YY = np.meshgrid(cx, cy)
    speed = np.sqrt(u ** 2 + v ** 2)
    scale = np.percentile(speed[speed > 0], 95) / (0.05 * (x_max - x_min)) if (speed > 0).any() else 1.0
    scale = max(scale, 1e-6) * 2.5
    x_tips = XX + u / scale
    y_tips = YY + v / scale
    plot_x_min = min(x_min, np.nanmin(x_tips))
    plot_x_max = max(x_max, np.nanmax(x_tips))
    plot_y_min = min(y_min, np.nanmin(y_tips))
    plot_y_max = max(y_max, np.nanmax(y_tips))
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_range = max(
        plot_x_max - center_x, center_x - plot_x_min,
        plot_y_max - center_y, center_y - plot_y_min,
    )
    pad_frac = 0.02
    half_range *= 1 + pad_frac
    plot_x_min = center_x - half_range
    plot_x_max = center_x + half_range
    plot_y_min = center_y - half_range
    plot_y_max = center_y + half_range
    vel_finite = mean_raw_vel[(mean_raw_vel > 0) & np.isfinite(mean_raw_vel)]
    vmin_vel = np.percentile(vel_finite, 10) if len(vel_finite) > 0 else 0.0
    vmax_vel = np.percentile(vel_finite, 95) if len(vel_finite) > 0 else 1.0
    elev_vmin = np.nanmin(elev_grid) if np.any(np.isfinite(elev_grid)) else 0.0
    elev_vmax = np.nanmax(elev_grid) if np.any(np.isfinite(elev_grid)) else 1.0

    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 8))
    for ax in (ax_left, ax_right):
        ax.set_xlim(plot_x_min, plot_x_max)
        ax.set_ylim(plot_y_min, plot_y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if np.any(np.isfinite(elev_grid)):
        im = ax_left.pcolormesh(x_edges, y_edges, elev_grid, cmap="turbo", shading="flat", vmin=elev_vmin, vmax=elev_vmax, alpha=0.9, zorder=0)
        plt.colorbar(im, ax=ax_left, label="elevation z", shrink=0.7)

    q_left = ax_left.quiver(XX, YY, u, v, color="black", alpha=0.9, scale_units="xy", scale=scale, width=0.006, headwidth=3, headlength=4, edgecolor="white", linewidth=0.2, zorder=10)
    ax_left.quiverkey(q_left, 0.88, 0.96, np.percentile(speed[speed > 0], 50) if (speed > 0).any() else 1.0, "flow", coordinates="axes")
    ax_left.set_title("Elevation + flow direction" + title_suffix)

    speed_display = np.where(mean_raw_vel > 0, mean_raw_vel, np.nan)
    cmap_turbo = plt.cm.turbo.copy()
    cmap_turbo.set_bad("white")
    im_speed = ax_right.pcolormesh(x_edges, y_edges, speed_display, cmap=cmap_turbo, shading="flat", vmin=vmin_vel, vmax=vmax_vel, alpha=0.9, zorder=0)
    plt.colorbar(im_speed, ax=ax_right, label="raw velocity (3D, dist/s)", shrink=0.7)
    q_right = ax_right.quiver(XX, YY, u, v, color="black", alpha=0.9, scale_units="xy", scale=scale, width=0.006, headwidth=3, headlength=4, edgecolor="white", linewidth=0.2, zorder=10)
    ax_right.quiverkey(q_right, 0.88, 0.96, np.percentile(speed[speed > 0], 50) if (speed > 0).any() else 1.0, "flow", coordinates="axes")
    ax_right.set_title("Flow speed + flow direction" + title_suffix)

    plt.tight_layout()
    fig2.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _time_spent_heatmap(
    trials_list: list[tuple[Path, str]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Compute time spent in each spatial bin (same grid as flow field) for the given trials; save single-panel heatmap to out_path."""
    if len(trials_list) == 0:
        return
    n_bins = 40
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    time_in_cell = np.zeros((n_bins, n_bins))
    for csv_path, _ in trials_list:
        df = load_trajectory_csv(csv_path)
        if len(df) < 2:
            continue
        mask = (df["z"] <= MAX_PEAK_Z) & ((df["y"] <= Y_FOR_Z_CAP) | (df["z"] <= Z_CAP_IN_Y_REGION))
        df = df.loc[mask]
        if len(df) < 2:
            continue
        xyzf = _ordered_xyz_frame_path(df)
        if len(xyzf) < 2:
            continue
        xyz = xyzf[:, :3]
        frames = xyzf[:, 3]
        for i in range(len(xyz) - 1):
            dt = (frames[i + 1] - frames[i]) / FPS
            if dt <= 0:
                continue
            mid_x = (xyz[i, 0] + xyz[i + 1, 0]) / 2
            mid_y = (xyz[i, 1] + xyz[i + 1, 1]) / 2
            mid_z = (xyz[i, 2] + xyz[i + 1, 2]) / 2
            if mid_y > Y_FOR_Z_CAP and mid_z > Z_CAP_IN_Y_REGION:
                continue
            ix = np.searchsorted(x_edges, mid_x, side="right") - 1
            iy = np.searchsorted(y_edges, mid_y, side="right") - 1
            ix = max(0, min(ix, n_bins - 1))
            iy = max(0, min(iy, n_bins - 1))
            time_in_cell[iy, ix] += dt
    fig, ax = plt.subplots(figsize=(10, 9))
    time_display = np.where(time_in_cell > 0, time_in_cell, np.nan)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white", alpha=0.3)
    im = ax.pcolormesh(x_edges, y_edges, time_display, cmap=cmap, shading="flat", alpha=0.9)
    plt.colorbar(im, ax=ax, label="Time spent (s)", shrink=0.7)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Time spent per region" + title_suffix)
    for v in [x_min + (x_max - x_min) * k / 4 for k in range(5)]:
        ax.axvline(v, color="k", linewidth=0.4, alpha=0.4)
    for v in [y_min + (y_max - y_min) * k / 4 for k in range(5)]:
        ax.axhline(v, color="k", linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_plots(stats_df: pd.DataFrame, predictions_dir: Path, out_dir: Path) -> None:
    """Generate illustrative plots and save to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = stats_df["trial_id"].values
    n_trials = len(stats_df)

    # 1) Path length (3D) per trial
    fig, ax = plt.subplots(figsize=(max(8, n_trials * 0.25), 5))
    ax.bar(range(n_trials), stats_df["path_length_3d"], color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(range(n_trials))
    ax.set_xticklabels([t.replace("Predictions_3D_", "") for t in trials], rotation=45, ha="right")
    ax.set_ylabel("Path length (3D, a.u.)")
    ax.set_title("Trajectory path length per trial")
    plt.tight_layout()
    fig.savefig(out_dir / "path_length_per_trial.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Duration (frames) per trial
    fig, ax = plt.subplots(figsize=(max(8, n_trials * 0.25), 5))
    ax.bar(range(n_trials), stats_df["duration_frames"], color="coral", edgecolor="darkred", alpha=0.8)
    ax.set_xticks(range(n_trials))
    ax.set_xticklabels([t.replace("Predictions_3D_", "") for t in trials], rotation=45, ha="right")
    ax.set_ylabel("Duration (frames)")
    ax.set_title("Trial duration (frame span) per trial")
    plt.tight_layout()
    fig.savefig(out_dir / "duration_per_trial.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Path length vs duration (scatter)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(stats_df["duration_frames"], stats_df["path_length_3d"], alpha=0.7, s=40)
    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Path length (3D)")
    ax.set_title("Path length vs duration")
    plt.tight_layout()
    fig.savefig(out_dir / "path_length_vs_duration.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4) Number of segments per trial
    fig, ax = plt.subplots(figsize=(max(8, n_trials * 0.25), 5))
    ax.bar(range(n_trials), stats_df["n_segments"], color="seagreen", edgecolor="darkgreen", alpha=0.8)
    ax.set_xticks(range(n_trials))
    ax.set_xticklabels([t.replace("Predictions_3D_", "") for t in trials], rotation=45, ha="right")
    ax.set_ylabel("Number of segments")
    ax.set_title("Trajectory segments per trial")
    plt.tight_layout()
    fig.savefig(out_dir / "n_segments_per_trial.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5) Histogram: path length distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(stats_df["path_length_3d"], bins=min(20, max(5, n_trials // 3)), color="steelblue", edgecolor="white")
    ax.set_xlabel("Path length (3D)")
    ax.set_ylabel("Number of trials")
    ax.set_title("Distribution of path length across trials")
    plt.tight_layout()
    fig.savefig(out_dir / "path_length_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6) Example top-down trajectories (xy) for first 6 trials — shared axis scale
    csv_list = find_trajectory_csvs(predictions_dir)[:6]
    n_ex = len(csv_list)
    if n_ex > 0:
        # Compute global x, y limits across all example trials so every panel has the same scale
        all_x, all_y = [], []
        for csv_path, _ in csv_list:
            df = load_trajectory_csv(csv_path)
            if len(df) >= 1:
                all_x.extend(df["x"].tolist())
                all_y.extend(df["y"].tolist())
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            # Small margin and ensure equal aspect range so scale is comparable
            dx = x_max - x_min or 1
            dy = y_max - y_min or 1
            margin = 0.05
            x_min = x_min - margin * dx
            x_max = x_max + margin * dx
            y_min = y_min - margin * dy
            y_max = y_max + margin * dy
        else:
            x_min = y_min = -100
            x_max = y_max = 100

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i, (csv_path, trial_id) in enumerate(csv_list):
            if i >= len(axes):
                break
            df = load_trajectory_csv(csv_path)
            ax = axes[i]
            if len(df) >= 2:
                for seg_id in df["segment_id"].unique():
                    seg = df[df["segment_id"] == seg_id]
                    ax.plot(seg["x"], seg["y"], alpha=0.8, linewidth=1)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.set_title(trial_id.replace("Predictions_3D_", ""), fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Example trajectories (top-down xy, 3D world, same scale)")
        plt.tight_layout()
        fig.savefig(out_dir / "example_trajectories_xy.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6b) All trials x-y in one plot (top-down, colored by elevation z)
    all_trials_xy = find_trajectory_csvs(predictions_dir)
    if len(all_trials_xy) > 0:
        all_x, all_y, all_z = [], [], []
        all_x_lim, all_y_lim = [], []
        for csv_path, _ in all_trials_xy:
            df = load_trajectory_csv(csv_path)
            if len(df) >= 1:
                all_x_lim.extend(df["x"].tolist())
                all_y_lim.extend(df["y"].tolist())
                # z <= 150 and exclude (y > -100 and z > 120)
                mask = (df["z"] <= MAX_PEAK_Z) & ((df["y"] <= Y_FOR_Z_CAP) | (df["z"] <= Z_CAP_IN_Y_REGION))
                all_x.extend(df.loc[mask, "x"].tolist())
                all_y.extend(df.loc[mask, "y"].tolist())
                all_z.extend(df.loc[mask, "z"].tolist())
        if all_x_lim and all_y_lim:
            x_min, x_max = min(all_x_lim), max(all_x_lim)
            y_min, y_max = min(all_y_lim), max(all_y_lim)
            dx = x_max - x_min or 1
            dy = y_max - y_min or 1
            margin = 0.05
            x_min = x_min - margin * dx
            x_max = x_max + margin * dx
            y_min = y_min - margin * dy
            y_max = y_max + margin * dy
        else:
            x_min = y_min = -100
            x_max = y_max = 100
        z_min = min(all_z) if all_z else 0
        z_max = max(all_z) if all_z else 100
        norm = plt.Normalize(vmin=z_min, vmax=z_max)
        cmap = plt.colormaps.get_cmap("viridis")
        fig, ax = plt.subplots(figsize=(10, 10))
        for csv_path, trial_id in all_trials_xy:
            df = load_trajectory_csv(csv_path)
            if len(df) < 2:
                continue
            for seg_id in df["segment_id"].unique():
                seg = df[df["segment_id"] == seg_id]
                seg = seg[(seg["z"] <= MAX_PEAK_Z) & ((seg["y"] <= Y_FOR_Z_CAP) | (seg["z"] <= Z_CAP_IN_Y_REGION))]
                if len(seg) < 2:
                    continue
                x = seg["x"].values.astype(float)
                y = seg["y"].values.astype(float)
                z = seg["z"].values.astype(float)
                n = len(x)
                segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
                z_seg = (z[:-1] + z[1:]) / 2
                lc = LineCollection(segments, array=z_seg, cmap=cmap, norm=norm, linewidth=1.2, alpha=0.9)
                ax.add_collection(lc)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("All trials — x-y (top-down), colored by elevation z")
        cbar = plt.colorbar(lc, ax=ax, label="z (elevation)")

        # Find 2 peak regions (opposite ends): high-z points, cluster into 2, use centroids
        high_z_pct = 92.0
        if all_z and len(all_z) >= 10:
            z_thresh = np.percentile(all_z, high_z_pct)
            pts_x, pts_y, pts_z = [], [], []
            for csv_path, _ in all_trials_xy:
                df = load_trajectory_csv(csv_path)
                if len(df) < 1:
                    continue
                mask = (df["z"] <= MAX_PEAK_Z) & ((df["y"] <= Y_FOR_Z_CAP) | (df["z"] <= Z_CAP_IN_Y_REGION))
                mask = mask & (df["z"].values >= z_thresh)
                pts_x.extend(df.loc[mask, "x"].tolist())
                pts_y.extend(df.loc[mask, "y"].tolist())
                pts_z.extend(df.loc[mask, "z"].tolist())
            pts = np.column_stack([pts_x, pts_y, pts_z]).astype(float)
            if len(pts) >= 2:
                centroids = _kmeans2_xyz(pts)
                ax.scatter(centroids[:, 0], centroids[:, 1], s=40, c="red", marker="o", zorder=5)
                peak_df = pd.DataFrame({"peak_id": [1, 2], "x": centroids[:, 0], "y": centroids[:, 1], "z": centroids[:, 2]})
                peak_path = out_dir / "high_peak_centroids.csv"
                peak_df.to_csv(peak_path, index=False)
                print(f"Saved 2 peak centroids (x,y,z) to {peak_path}")

        plt.tight_layout()
        fig.savefig(out_dir / "all_trials_xy.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 6b2) Trajectories for trials where vertical is on the left (left_angle_deg == 360)
        trial_types_path = out_dir / "trial_types.csv"
        if trial_types_path.is_file():
            tt = pd.read_csv(trial_types_path)
            vertical_left_ids = set(tt.loc[(tt["left_angle_deg"] == 360.0) & (tt["right_angle_deg"] != 360.0), "trial_id"].astype(str))
            vertical_left_trials = [(p, tid) for p, tid in all_trials_xy if tid in vertical_left_ids]
            if len(vertical_left_trials) > 0:
                all_x_vl, all_y_vl, all_z_vl = [], [], []
                all_x_lim_vl, all_y_lim_vl = [], []
                for csv_path, _ in vertical_left_trials:
                    df = load_trajectory_csv(csv_path)
                    if len(df) >= 1:
                        all_x_lim_vl.extend(df["x"].tolist())
                        all_y_lim_vl.extend(df["y"].tolist())
                        mask = (df["z"] <= MAX_PEAK_Z) & ((df["y"] <= Y_FOR_Z_CAP) | (df["z"] <= Z_CAP_IN_Y_REGION))
                        all_x_vl.extend(df.loc[mask, "x"].tolist())
                        all_y_vl.extend(df.loc[mask, "y"].tolist())
                        all_z_vl.extend(df.loc[mask, "z"].tolist())
                if all_x_lim_vl and all_y_lim_vl:
                    x_min_vl = min(all_x_lim_vl) - 0.05 * (max(all_x_lim_vl) - min(all_x_lim_vl) or 1)
                    x_max_vl = max(all_x_lim_vl) + 0.05 * (max(all_x_lim_vl) - min(all_x_lim_vl) or 1)
                    y_min_vl = min(all_y_lim_vl) - 0.05 * (max(all_y_lim_vl) - min(all_y_lim_vl) or 1)
                    y_max_vl = max(all_y_lim_vl) + 0.05 * (max(all_y_lim_vl) - min(all_y_lim_vl) or 1)
                else:
                    x_min_vl, x_max_vl, y_min_vl, y_max_vl = x_min, x_max, y_min, y_max
                z_min_vl = min(all_z_vl) if all_z_vl else 0
                z_max_vl = max(all_z_vl) if all_z_vl else 100
                norm_vl = plt.Normalize(vmin=z_min_vl, vmax=z_max_vl)
                fig_vl, ax_vl = plt.subplots(figsize=(10, 10))
                for csv_path, trial_id in vertical_left_trials:
                    df = load_trajectory_csv(csv_path)
                    if len(df) < 2:
                        continue
                    for seg_id in df["segment_id"].unique():
                        seg = df[df["segment_id"] == seg_id]
                        seg = seg[(seg["z"] <= MAX_PEAK_Z) & ((seg["y"] <= Y_FOR_Z_CAP) | (seg["z"] <= Z_CAP_IN_Y_REGION))]
                        if len(seg) < 2:
                            continue
                        x = seg["x"].values.astype(float)
                        y = seg["y"].values.astype(float)
                        z = seg["z"].values.astype(float)
                        segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
                        z_seg = (z[:-1] + z[1:]) / 2
                        lc_vl = LineCollection(segments, array=z_seg, cmap=cmap, norm=norm_vl, linewidth=1.2, alpha=0.9)
                        ax_vl.add_collection(lc_vl)
                ax_vl.set_xlim(x_min_vl, x_max_vl)
                ax_vl.set_ylim(y_min_vl, y_max_vl)
                ax_vl.set_aspect("equal")
                ax_vl.set_xlabel("x")
                ax_vl.set_ylabel("y")
                ax_vl.set_title(f"Trials with vertical on left — x-y (top-down), n = {len(vertical_left_trials)}")
                plt.colorbar(lc_vl, ax=ax_vl, label="z (elevation)")
                plt.tight_layout()
                fig_vl.savefig(out_dir / "trajectories_xy_vertical_on_left.png", dpi=150, bbox_inches="tight")
                plt.close()

            # 6b3) Trajectories for trials where vertical is on the right (right_angle_deg == 360)
            vertical_right_ids = set(tt.loc[(tt["right_angle_deg"] == 360.0) & (tt["left_angle_deg"] != 360.0), "trial_id"].astype(str))
            vertical_right_trials = [(p, tid) for p, tid in all_trials_xy if tid in vertical_right_ids]
            if len(vertical_right_trials) > 0:
                all_x_vr, all_y_vr, all_z_vr = [], [], []
                all_x_lim_vr, all_y_lim_vr = [], []
                for csv_path, _ in vertical_right_trials:
                    df = load_trajectory_csv(csv_path)
                    if len(df) >= 1:
                        all_x_lim_vr.extend(df["x"].tolist())
                        all_y_lim_vr.extend(df["y"].tolist())
                        mask = (df["z"] <= MAX_PEAK_Z) & ((df["y"] <= Y_FOR_Z_CAP) | (df["z"] <= Z_CAP_IN_Y_REGION))
                        all_x_vr.extend(df.loc[mask, "x"].tolist())
                        all_y_vr.extend(df.loc[mask, "y"].tolist())
                        all_z_vr.extend(df.loc[mask, "z"].tolist())
                if all_x_lim_vr and all_y_lim_vr:
                    x_min_vr = min(all_x_lim_vr) - 0.05 * (max(all_x_lim_vr) - min(all_x_lim_vr) or 1)
                    x_max_vr = max(all_x_lim_vr) + 0.05 * (max(all_x_lim_vr) - min(all_x_lim_vr) or 1)
                    y_min_vr = min(all_y_lim_vr) - 0.05 * (max(all_y_lim_vr) - min(all_y_lim_vr) or 1)
                    y_max_vr = max(all_y_lim_vr) + 0.05 * (max(all_y_lim_vr) - min(all_y_lim_vr) or 1)
                else:
                    x_min_vr, x_max_vr, y_min_vr, y_max_vr = x_min, x_max, y_min, y_max
                z_min_vr = min(all_z_vr) if all_z_vr else 0
                z_max_vr = max(all_z_vr) if all_z_vr else 100
                norm_vr = plt.Normalize(vmin=z_min_vr, vmax=z_max_vr)
                fig_vr, ax_vr = plt.subplots(figsize=(10, 10))
                for csv_path, trial_id in vertical_right_trials:
                    df = load_trajectory_csv(csv_path)
                    if len(df) < 2:
                        continue
                    for seg_id in df["segment_id"].unique():
                        seg = df[df["segment_id"] == seg_id]
                        seg = seg[(seg["z"] <= MAX_PEAK_Z) & ((seg["y"] <= Y_FOR_Z_CAP) | (seg["z"] <= Z_CAP_IN_Y_REGION))]
                        if len(seg) < 2:
                            continue
                        x = seg["x"].values.astype(float)
                        y = seg["y"].values.astype(float)
                        z = seg["z"].values.astype(float)
                        segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
                        z_seg = (z[:-1] + z[1:]) / 2
                        lc_vr = LineCollection(segments, array=z_seg, cmap=cmap, norm=norm_vr, linewidth=1.2, alpha=0.9)
                        ax_vr.add_collection(lc_vr)
                ax_vr.set_xlim(x_min_vr, x_max_vr)
                ax_vr.set_ylim(y_min_vr, y_max_vr)
                ax_vr.set_aspect("equal")
                ax_vr.set_xlabel("x")
                ax_vr.set_ylabel("y")
                ax_vr.set_title(f"Trials with vertical on right — x-y (top-down), n = {len(vertical_right_trials)}")
                plt.colorbar(lc_vr, ax=ax_vr, label="z (elevation)")
                plt.tight_layout()
                fig_vr.savefig(out_dir / "trajectories_xy_vertical_on_right.png", dpi=150, bbox_inches="tight")
                plt.close()

            # 6c2) Flow field for trials with vertical on left
            if len(vertical_left_trials) > 0:
                _flow_field_two_panels(
                    vertical_left_trials,
                    x_min_vl, x_max_vl, y_min_vl, y_max_vl,
                    out_dir / "flow_field_low_to_high_vertical_on_left.png",
                    " (vertical on left)",
                )
                # Time spent per region (same grid as flow map) for vertical on left
                _time_spent_heatmap(
                    vertical_left_trials,
                    x_min_vl, x_max_vl, y_min_vl, y_max_vl,
                    out_dir / "time_spent_vertical_on_left.png",
                    " (vertical on left)",
                )
            # 6c3) Flow field for trials with vertical on right
            if len(vertical_right_trials) > 0:
                _flow_field_two_panels(
                    vertical_right_trials,
                    x_min_vr, x_max_vr, y_min_vr, y_max_vr,
                    out_dir / "flow_field_low_to_high_vertical_on_right.png",
                    " (vertical on right)",
                )
                # Time spent per region (same grid as flow map) for vertical on right
                _time_spent_heatmap(
                    vertical_right_trials,
                    x_min_vr, x_max_vr, y_min_vr, y_max_vr,
                    out_dir / "time_spent_vertical_on_right.png",
                    " (vertical on right)",
                )

        # 6c) Flow field (low -> high elevation): two panels. All trials + (if trial_types) vertical-on-left and vertical-on-right.
        _flow_field_two_panels(all_trials_xy, x_min, x_max, y_min, y_max, out_dir / "flow_field_low_to_high.png", "")

    # 7) Elevation (z) vs x and z vs y for first 6 trials — shared scales
    if n_ex > 0:
        all_x, all_y, all_z = [], [], []
        for csv_path, _ in csv_list:
            df = load_trajectory_csv(csv_path)
            if len(df) >= 1:
                all_x.extend(df["x"].tolist())
                all_y.extend(df["y"].tolist())
                all_z.extend(df["z"].tolist())
        if all_x and all_y and all_z:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            z_min, z_max = min(all_z), max(all_z)
            x_min, x_max = x_min - 0.05 * (x_max - x_min or 1), x_max + 0.05 * (x_max - x_min or 1)
            y_min, y_max = y_min - 0.05 * (y_max - y_min or 1), y_max + 0.05 * (y_max - y_min or 1)
            z_min, z_max = z_min - 0.05 * (z_max - z_min or 1), z_max + 0.05 * (z_max - z_min or 1)
        else:
            x_min = y_min = z_min = -100
            x_max = y_max = z_max = 100

        # Row 1: z vs x; Row 2: z vs y
        fig, axes = plt.subplots(2, 6, figsize=(14, 5))
        for i, (csv_path, trial_id) in enumerate(csv_list):
            if i >= 6:
                break
            df = load_trajectory_csv(csv_path)
            # z vs x
            ax_x = axes[0, i]
            if len(df) >= 2:
                for seg_id in df["segment_id"].unique():
                    seg = df[df["segment_id"] == seg_id]
                    ax_x.plot(seg["x"], seg["z"], alpha=0.8, linewidth=1)
            ax_x.set_xlim(x_min, x_max)
            ax_x.set_ylim(z_min, z_max)
            ax_x.set_xlabel("x")
            ax_x.set_ylabel("z (elevation)")
            ax_x.set_title(trial_id.replace("Predictions_3D_", ""), fontsize=9)
            # z vs y
            ax_y = axes[1, i]
            if len(df) >= 2:
                for seg_id in df["segment_id"].unique():
                    seg = df[df["segment_id"] == seg_id]
                    ax_y.plot(seg["y"], seg["z"], alpha=0.8, linewidth=1)
            ax_y.set_xlim(y_min, y_max)
            ax_y.set_ylim(z_min, z_max)
            ax_y.set_xlabel("y")
            ax_y.set_ylabel("z (elevation)")
        fig.suptitle("Elevation (z) vs x (top) and vs y (bottom), same scale across trials")
        plt.tight_layout()
        fig.savefig(out_dir / "elevation_vs_xy.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 7b) More example trajectories: z vs y (x-axis: positive y → negative y; y-axis: negative z → positive z)
    csv_list_zvy = find_trajectory_csvs(predictions_dir)
    n_zvy = len(csv_list_zvy)
    if n_zvy > 0:
        all_y, all_z = [], []
        for csv_path, _ in csv_list_zvy:
            df = load_trajectory_csv(csv_path)
            if len(df) >= 1:
                all_y.extend(df["y"].tolist())
                all_z.extend(df["z"].tolist())
        if all_y and all_z:
            y_min, y_max = min(all_y), max(all_y)
            z_min, z_max = min(all_z), max(all_z)
            y_min = y_min - 0.05 * (y_max - y_min or 1)
            y_max = y_max + 0.05 * (y_max - y_min or 1)
            z_min = z_min - 0.05 * (z_max - z_min or 1)
            z_max = z_max + 0.05 * (z_max - z_min or 1)
        else:
            y_min, y_max, z_min, z_max = -100, 100, -100, 100
        n_cols = 5
        n_rows = (n_zvy + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes)
        for i, (csv_path, trial_id) in enumerate(csv_list_zvy):
            if i >= n_zvy:
                break
            ax = axes.flat[i]
            df = load_trajectory_csv(csv_path)
            if len(df) >= 2:
                for seg_id in df["segment_id"].unique():
                    seg = df[df["segment_id"] == seg_id]
                    ax.plot(seg["y"], seg["z"], alpha=0.8, linewidth=1)
            ax.set_xlim(y_max, y_min)  # x-axis: positive y → negative y (left to right)
            ax.set_ylim(z_min, z_max)  # y-axis: negative z → positive z (bottom to top)
            ax.set_xlabel("y")
            ax.set_ylabel("z (elevation)")
            ax.set_title(trial_id.replace("Predictions_3D_", ""), fontsize=8)
        for j in range(n_zvy, axes.size):
            axes.flat[j].set_visible(False)
        fig.suptitle("Elevation (z) vs y — x-axis: +y → −y, y-axis: −z → +z (more examples)")
        plt.tight_layout()
        fig.savefig(out_dir / "elevation_vs_y_examples.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 8) Elevation (z) vs time (frame_number) for first 6 trials — shared z scale
    if n_ex > 0:
        all_z = []
        for csv_path, _ in csv_list:
            df = load_trajectory_csv(csv_path)
            if len(df) >= 1:
                all_z.extend(df["z"].tolist())
        z_min, z_max = (min(all_z), max(all_z)) if all_z else (-100, 100)
        z_min = z_min - 0.05 * (z_max - z_min or 1)
        z_max = z_max + 0.05 * (z_max - z_min or 1)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i, (csv_path, trial_id) in enumerate(csv_list):
            if i >= len(axes):
                break
            df = load_trajectory_csv(csv_path)
            ax = axes[i]
            if len(df) >= 2:
                for seg_id in df["segment_id"].unique():
                    seg = df[df["segment_id"] == seg_id]
                    ax.plot(seg["frame_number"], seg["z"], alpha=0.8, linewidth=1)
            ax.set_ylim(z_min, z_max)
            ax.set_xlabel("Frame (time)")
            ax.set_ylabel("z (elevation)")
            ax.set_title(trial_id.replace("Predictions_3D_", ""), fontsize=9)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Elevation (z) vs time (frame number), same z scale")
        plt.tight_layout()
        fig.savefig(out_dir / "elevation_vs_frame.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 9) Elevation (z) vs normalized time (frame from 0) — all trials, one line per trial, color-coded
    all_trials_list = find_trajectory_csvs(predictions_dir)
    if len(all_trials_list) > 0:
        try:
            cmap = plt.colormaps.get_cmap("turbo").resampled(len(all_trials_list))
        except AttributeError:
            cmap = plt.cm.get_cmap("turbo")
        fig, ax = plt.subplots(figsize=(10, 6))
        for trial_idx, (csv_path, trial_id) in enumerate(all_trials_list):
            df = load_trajectory_csv(csv_path)
            if len(df) < 2:
                continue
            frame_start = df["frame_number"].min()
            frame_norm = df["frame_number"] - frame_start
            color = cmap(trial_idx / max(1, len(all_trials_list) - 1))
            for seg_id in df["segment_id"].unique():
                seg = df[df["segment_id"] == seg_id]
                ax.plot(
                    seg["frame_number"] - frame_start,
                    seg["z"],
                    color=color,
                    alpha=0.85,
                    linewidth=0.8,
                )
        ax.set_xlabel("Frame (from trial start, 0)")
        ax.set_ylabel("z (elevation)")
        ax.set_title("Elevation vs time — all trials (frame normalized to start at 0)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(all_trials_list) - 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Trial index")
        plt.tight_layout()
        fig.savefig(out_dir / "elevation_vs_frame_all_trials.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 10) Point cloud: (x, y) location where z is highest for each trial (exclude only the point when z > MAX_PEAK_Z)
    all_trials_list = find_trajectory_csvs(predictions_dir)
    if len(all_trials_list) > 0:
        x_peak, y_peak, z_peak = [], [], []
        for csv_path, trial_id in all_trials_list:
            df = load_trajectory_csv(csv_path)
            if len(df) < 1:
                continue
            idx_max = df["z"].idxmax()
            row = df.loc[idx_max]
            if row["z"] > MAX_PEAK_Z:
                continue
            x_peak.append(row["x"])
            y_peak.append(row["y"])
            z_peak.append(row["z"])
        if x_peak:
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(x_peak, y_peak, c=z_peak, cmap="viridis", s=50, alpha=0.8, edgecolors="k", linewidths=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_title("Location of highest elevation (z) per trial — 2D point cloud")
            plt.colorbar(sc, ax=ax, label="z (max elevation)")
            plt.tight_layout()
            fig.savefig(out_dir / "peak_elevation_xy.png", dpi=150, bbox_inches="tight")
            plt.close()

    # 11) Summary boxplots: path length, duration, n_points
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, col, title in zip(
        axes,
        ["path_length_3d", "duration_frames", "n_points"],
        ["Path length (3D)", "Duration (frames)", "Number of points"],
    ):
        ax.boxplot(stats_df[col], vert=True)
        ax.set_ylabel(title)
        ax.set_title(title)
    plt.suptitle("Summary statistics across trials")
    plt.tight_layout()
    fig.savefig(out_dir / "summary_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Side pick (angle vs angle): heatmap of proportion "left" per (left_angle, right_angle) bin + scatter; 360 labelled as "vertical"
    mouse_choice_path = out_dir / "mouse_choice.csv"
    if mouse_choice_path.is_file():
        mc = pd.read_csv(mouse_choice_path)
        if "left_angle_deg" in mc.columns and "right_angle_deg" in mc.columns and "side_pick" in mc.columns:
            mc = mc.dropna(subset=["left_angle_deg", "right_angle_deg", "side_pick"])
            if len(mc) > 0:
                fig, ax = plt.subplots(figsize=(9, 8))
                # Heatmap: bin (left_angle, right_angle), value = proportion of trials that picked "left" in that bin
                bin_edges = np.arange(0, 361, 45)  # 0, 45, ..., 360 → 8 bins per axis
                left_bin = np.digitize(mc["left_angle_deg"].values, bin_edges, right=False) - 1
                right_bin = np.digitize(mc["right_angle_deg"].values, bin_edges, right=False) - 1
                left_bin = np.clip(left_bin, 0, len(bin_edges) - 2)
                right_bin = np.clip(right_bin, 0, len(bin_edges) - 2)
                n_bins = len(bin_edges) - 1
                H_left = np.zeros((n_bins, n_bins))
                H_count = np.zeros((n_bins, n_bins))
                for i in range(len(mc)):
                    li, ri = int(left_bin[i]), int(right_bin[i])
                    H_count[li, ri] += 1
                    if mc["side_pick"].iloc[i] == "left":
                        H_left[li, ri] += 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    prop_left_pct = np.where(H_count > 0, 100 * H_left / H_count, np.nan)
                cmap = plt.cm.RdBu_r
                cmap = cmap.copy()
                cmap.set_bad("white", alpha=0.3)
                im = ax.pcolormesh(
                    bin_edges, bin_edges, prop_left_pct.T,
                    cmap=cmap, vmin=0, vmax=100, shading="flat", alpha=0.7,
                )
                cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="% choose left")
                cbar.ax.axhline(0.5, color="k", linewidth=0.5)  # 50% line
                # Axis labels: show 360 as "vertical"
                def format_angle(tick_val, _):
                    if tick_val == 360:
                        return "vertical"
                    return f"{int(tick_val)}"
                tick_vals = [0, 90, 180, 270, 360]
                ax.set_xticks(tick_vals)
                ax.set_yticks(tick_vals)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_angle))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_angle))
                ax.set_xlabel("Left angle")
                ax.set_ylabel("Right angle")
                ax.set_title("Side pick at peak z (x at highest elevation): angle vs angle")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 360)
                ax.set_aspect("equal")
                # Draw lines at major angles so bins are visible
                for v in [0, 90, 180, 270, 360]:
                    ax.axvline(v, color="k", linewidth=0.6, alpha=0.6)
                    ax.axhline(v, color="k", linewidth=0.6, alpha=0.6)
                plt.tight_layout()
                fig.savefig(out_dir / "side_pick_angle_vs_angle.png", dpi=150, bbox_inches="tight")
                plt.close()


def overlay_peak_elevation_on_frame(
    frame_path: Path,
    predictions_dir: Path,
    out_path: Path,
    calib_path: Path,
) -> None:
    """Load peak (x,y,z) points per trial, project to 2D with calib, draw on frame, save. Exclude only the point when z > MAX_PEAK_Z."""
    from PIL import Image

    frame_path = Path(frame_path)
    calib_path = Path(calib_path)
    frame = np.array(Image.open(frame_path))
    img_h, img_w = frame.shape[:2]

    calib = load_calib(calib_path)
    all_trials_list = find_trajectory_csvs(Path(predictions_dir))
    x_peak, y_peak, z_peak = [], [], []
    for csv_path, _ in all_trials_list:
        df = load_trajectory_csv(csv_path)
        if len(df) < 1:
            continue
        idx_max = df["z"].idxmax()
        row = df.loc[idx_max]
        if row["z"] > MAX_PEAK_Z:
            continue
        x_peak.append(row["x"])
        y_peak.append(row["y"])
        z_peak.append(row["z"])
    if not x_peak:
        raise SystemExit("No peak elevation points to plot (all excluded by z > MAX_PEAK_Z?).")
    points_3d = np.column_stack([x_peak, y_peak, z_peak])
    points_2d = project_3d_to_2d(points_3d, calib)
    u, v = points_2d[:, 0], points_2d[:, 1]
    z_peak = np.array(z_peak)

    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100))
    ax.imshow(frame, extent=[0, img_w, img_h, 0])
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    sc = ax.scatter(u[in_bounds], v[in_bounds], c=np.array(z_peak)[in_bounds], cmap="viridis", s=80, alpha=0.9, edgecolors="white", linewidths=1)
    plt.colorbar(sc, ax=ax, label="z (max elevation)")
    ax.set_title("Peak elevation (x,y) per trial — projected on frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved peak elevation overlay: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute stats and plots from trajectory_filtered.csv across trials."
    )
    parser.add_argument(
        "predictions_dir",
        type=Path,
        nargs="?",
        default=Path("predictions3D"),
        help="Directory containing Predictions_3D_trial_* folders (default: predictions3D)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("trajectory_analysis"),
        help="Output directory for summary CSV and plots (default: trajectory_analysis)",
    )
    parser.add_argument(
        "--overlay-peak-on-frame",
        type=Path,
        default=None,
        metavar="FRAME.png",
        help="Overlay 50 peak-elevation points on this frame image; requires --camera for calibration",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera name (e.g. Cam2005325); required when using --overlay-peak-on-frame",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir).resolve()
    if not predictions_dir.is_dir():
        raise SystemExit(f"Not a directory: {predictions_dir}")

    csv_list = find_trajectory_csvs(predictions_dir)
    if not csv_list:
        raise SystemExit(f"No trajectory_filtered.csv found under {predictions_dir}")

    print(f"Found {len(csv_list)} trials with trajectory_filtered.csv")

    stats_df = aggregate_all_trials(predictions_dir)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir / "trajectory_stats_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved summary: {stats_path}")

    peak_points_df, path_to_peak_df = compute_peak_points_and_path_to_peak(predictions_dir)
    peak_path = out_dir / "peak_elevation_points.csv"
    peak_points_df.to_csv(peak_path, index=False)
    print(f"Saved peak elevation points: {peak_path}")

    # Mouse choice: at the point of peak z (highest elevation), use the corresponding x: x > -10 = left, x <= -10 = right
    peak_points_df = peak_points_df.copy()
    peak_points_df["side_pick"] = np.where(peak_points_df["x"] > -10, "left", "right")
    mouse_choice_df = peak_points_df[["trial_id", "x", "y", "z", "side_pick"]].rename(columns={"x": "peak_x", "y": "peak_y", "z": "peak_z"})
    trial_types_path = out_dir / "trial_types.csv"
    if trial_types_path.is_file():
        trial_types = pd.read_csv(trial_types_path)
        mouse_choice_df = mouse_choice_df.merge(trial_types[["trial_id", "left_angle_deg", "right_angle_deg", "trial_type"]], on="trial_id", how="left")
    mouse_choice_path = out_dir / "mouse_choice.csv"
    mouse_choice_df.to_csv(mouse_choice_path, index=False)
    print(f"Saved mouse choice (side pick at peak): {mouse_choice_path}")
    path_to_peak_path = out_dir / "path_to_peak_summary.csv"
    path_to_peak_df.to_csv(path_to_peak_path, index=False)
    print(f"Saved path-to-peak summary: {path_to_peak_path}")
    if len(path_to_peak_df) > 0:
        ftp = path_to_peak_df["frames_to_peak"]
        stp = path_to_peak_df["seconds_to_peak"]
        print(f"\nPath to peak elevation (trajectory start → peak):")
        print(f"  Frames: mean = {ftp.mean():.1f}, std = {ftp.std():.1f}")
        print(f"  Time (@ {FPS} fps): mean = {stp.mean():.2f} s, std = {stp.std():.2f} s, n = {len(path_to_peak_df)}")

    make_plots(stats_df, predictions_dir, out_dir)
    print(f"Saved plots to: {out_dir}")

    if args.overlay_peak_on_frame is not None:
        frame_path = Path(args.overlay_peak_on_frame).resolve()
        if not frame_path.exists():
            raise SystemExit(f"Frame not found: {frame_path}")
        if not args.camera:
            raise SystemExit("--camera is required when using --overlay-peak-on-frame")
        trial_dir = frame_path.parent
        import yaml
        calib_path = None
        info_path = trial_dir / "info.yaml"
        if info_path.exists():
            with open(info_path) as f:
                info = yaml.safe_load(f)
                if "dataset_name" in info:
                    calib_path = Path(info["dataset_name"]) / f"{args.camera}.yaml"
        if calib_path is None or not calib_path.exists():
            raise SystemExit(f"Calibration not found for {args.camera}; check {info_path} and dataset_name")
        overlay_peak_elevation_on_frame(
            frame_path,
            predictions_dir,
            out_dir / "peak_elevation_on_frame.png",
            calib_path,
        )

    print("\nSummary stats:")
    print(stats_df[["path_length_3d", "duration_frames", "n_points", "n_segments"]].describe().round(2))


if __name__ == "__main__":
    main()
