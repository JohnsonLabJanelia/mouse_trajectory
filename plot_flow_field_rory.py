#!/usr/bin/env python3
"""
Midline-and-goals analysis per animal (u-v camera space).

Generates: elevation/flow panels, midline + goals JSON, path plots (trial start → reward; by crossing count; phase and vertical left/right), crossing-location histograms, goal-region heatmaps, etc.

Outputs: trajectory_analysis/<animal>/midline_and_goals/*.png and midline_and_goals.json.
Use --animal rory or --animal wilfred (default: rory). See docs/RORY_MIDLINE_AND_GOALS.md for full documentation.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Optional: resolve reward time from cbot_climb_log
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
import numpy as np
import pandas as pd
from scipy import ndimage

import analyze_trajectories as at

FPS = getattr(at, "FPS", 180)
# Video FPS for converting log time_to_target (sec) to reward frame
VIDEO_FPS = 180

# Trial folder name: Predictions_3D_trial_0000_11572-20491 -> frame_start=11572, frame_end=20491
TRIAL_PATTERN = re.compile(r"^Predictions_3D_trial_(\d+)_(\d+)-(\d+)$")

# Geometry of rectangular goal regions (set by _flow_field_uv_highlight_high_z in the current run)
GOAL_RECT_GEOM: dict | None = None


def get_animal_trials(predictions_dir: Path, animal: str) -> list[tuple[Path, str, str | None]]:
    """Return list of (csv_path, trial_id, session_folder) for the given animal only.
    Session folder is expected to start with '<animal>_' (e.g. rory_..., wilfred_...)."""
    animal_lower = animal.strip().lower()
    all_trials = at.find_trajectory_csvs(predictions_dir)
    return [t for t in all_trials if t[2] and t[2].lower().startswith(f"{animal_lower}_")]


def _parse_trial_frames(trial_id: str) -> tuple[int, int] | None:
    """Parse trial_id (e.g. Predictions_3D_trial_0000_11572-20491) -> (frame_start, frame_end)."""
    m = TRIAL_PATTERN.match(trial_id)
    if not m:
        return None
    return int(m.group(2)), int(m.group(3))


def _video_folder_to_date_minutes(vf: str) -> tuple[str, int] | None:
    """Parse video_folder YYYY_MM_DD_HH_MM_SS -> (YYYY_MM_DD, minutes_since_midnight)."""
    parts = vf.strip().split("_")
    if len(parts) != 6:
        return None
    try:
        y, mo, d, h, mi, s = (int(x) for x in parts)
        date_str = f"{y:04d}_{mo:02d}_{d:02d}"
        minutes = h * 60 + mi + s / 60.0
        return (date_str, int(minutes))
    except (ValueError, TypeError):
        return None


def _best_matching_video_folder(pred_vf: str, csv_video_folders: list[str], max_minutes_diff: int = 15) -> str | None:
    """Return CSV video_folder with same date and closest time to pred_vf, or None."""
    pred_parsed = _video_folder_to_date_minutes(pred_vf)
    if pred_parsed is None:
        return None
    pred_date, pred_m = pred_parsed
    best_vf: str | None = None
    best_diff: int = max_minutes_diff + 1
    for vf in csv_video_folders:
        p = _video_folder_to_date_minutes(vf)
        if p is None or p[0] != pred_date:
            continue
        diff = abs(p[1] - pred_m)
        if diff < best_diff:
            best_diff = diff
            best_vf = vf
    return best_vf


def _get_reward_frame_from_csv(
    trials_list: list[tuple[Path, str, str | None]],
    reward_times_path: Path,
) -> dict[str, int]:
    """Load reward_times.csv (animal, video_folder, trial_index, time_to_target_sec) and return dict trial_id -> reward_frame.
    Matches prediction session to CSV by same animal and same date + closest time (log vs video folder can differ slightly)."""
    reward_times_path = Path(reward_times_path)
    if not reward_times_path.is_file():
        return {}
    try:
        df = pd.read_csv(reward_times_path)
    except Exception:
        return {}
    if "animal" not in df.columns or "video_folder" not in df.columns or "trial_index" not in df.columns or "time_to_target_sec" not in df.columns:
        return {}
    df = df.dropna(subset=["time_to_target_sec"])
    lookup: dict[tuple[str, str, int], float] = {}
    for _, row in df.iterrows():
        key = (str(row["animal"]).strip().lower(), str(row["video_folder"]).strip(), int(row["trial_index"]))
        lookup[key] = float(row["time_to_target_sec"])
    csv_by_animal: dict[str, list[str]] = {}
    for (a, vf, _) in lookup:
        csv_by_animal.setdefault(a, []).append(vf)
    for a in csv_by_animal:
        csv_by_animal[a] = list(dict.fromkeys(csv_by_animal[a]))

    out: dict[str, int] = {}
    for _csv_path, trial_id, session_folder in trials_list:
        if not session_folder:
            continue
        parts = session_folder.split("_", 1)
        animal = parts[0].lower() if len(parts) > 1 else ""
        video_folder = parts[1] if len(parts) > 1 else ""
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        trial_index = int(TRIAL_PATTERN.match(trial_id).group(1))
        key = (animal, video_folder, trial_index)
        if key not in lookup:
            matched_vf = _best_matching_video_folder(video_folder, csv_by_animal.get(animal, [])) if animal else None
            if matched_vf is not None:
                key = (animal, matched_vf, trial_index)
        if key not in lookup:
            continue
        time_to_target = lookup[key]
        reward_frame = int(frame_start + time_to_target * VIDEO_FPS)
        out[trial_id] = reward_frame
    return out


def _get_reward_frame_per_trial(
    trials_list: list[tuple[Path, str, str | None]],
    logs_dir: Path,
    animal: str,
) -> dict[str, int]:
    """Resolve reward frame for each trial_id from robot_manager.log. Returns dict trial_id -> reward_frame (video frame index).
    Uses time_to_target = door open -> reward (sec); reward_frame = frame_start + time_to_target * VIDEO_FPS."""
    try:
        from cbot_climb_log.analyze_logs import parse_robot_manager_log
        from cbot_climb_log.extract_trial_frames import find_matching_log_session
    except ImportError:
        return {}
    logs_dir = Path(logs_dir)
    if not logs_dir.is_dir():
        return {}
    animal_lower = animal.strip().lower()
    out: dict[str, int] = {}
    for csv_path, trial_id, session_folder in trials_list:
        if not session_folder or not session_folder.lower().startswith(f"{animal_lower}_"):
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        parts = session_folder.split("_", 1)
        animal_name = parts[0].lower() if len(parts) > 1 else animal_lower
        video_folder = parts[1] if len(parts) > 1 else ""
        log_path = find_matching_log_session(video_folder, animal_name, logs_dir)
        if log_path is None:
            continue
        _, training = parse_robot_manager_log(log_path)
        trial_num = int(TRIAL_PATTERN.match(trial_id).group(1))
        if trial_num >= len(training):
            continue
        time_to_target = training[trial_num][4]
        if time_to_target is None:
            continue
        reward_frame = int(frame_start + time_to_target * VIDEO_FPS)
        out[trial_id] = reward_frame
    return out


def _ordered_uvz_frame_path(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 4) array of (u, v, z, frame_number) along path."""
    if "u" not in df.columns or "v" not in df.columns:
        return np.zeros((0, 4))
    rows = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg = df[df["segment_id"] == seg_id].sort_values("frame_number")
        rows.append(seg[["u", "v", "z", "frame_number"]].values.astype(float))
    if not rows:
        return np.zeros((0, 4))
    return np.vstack(rows)


def uv_limits(trials_list: list[tuple[Path, str, str | None]], margin: float = 0.05) -> tuple[float, float, float, float]:
    """Compute u_min, u_max, v_min, v_max from trajectory CSVs (after load_trajectory_csv filters)."""
    all_u, all_v = [], []
    for csv_path, _, _ in trials_list:
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
            continue
        all_u.extend(df["u"].tolist())
        all_v.extend(df["v"].tolist())
    if not all_u or not all_v:
        return 0.0, 1000.0, 0.0, 1000.0
    u_min, u_max = min(all_u), max(all_u)
    v_min, v_max = min(all_v), max(all_v)
    du = (u_max - u_min) or 1
    dv = (v_max - v_min) or 1
    u_min -= margin * du
    u_max += margin * du
    v_min -= margin * dv
    v_max += margin * dv
    return u_min, u_max, v_min, v_max


def _compute_flow_field_grids(
    trials_list: list[tuple[Path, str, str | None]],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    n_bins: int = 40,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, float, float, float, float,
]:
    """Build elevation grid, flow, and mesh. Returns (elev_grid, u_edges, v_edges, flow_u, flow_v, UU, VV, mean_vel, scale, elev_vmin, elev_vmax)."""
    u_edges = np.linspace(u_min, u_max, n_bins + 1)
    v_edges = np.linspace(v_min, v_max, n_bins + 1)
    sum_z_elev = np.zeros((n_bins, n_bins))
    count_z = np.zeros((n_bins, n_bins))
    sum_du = np.zeros((n_bins, n_bins))
    sum_dv = np.zeros((n_bins, n_bins))
    sum_w = np.zeros((n_bins, n_bins)) + 1e-12
    sum_vel = np.zeros((n_bins, n_bins))
    for csv_path, _, _ in trials_list:
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
            continue
        uvzf = _ordered_uvz_frame_path(df)
        if len(uvzf) < 2:
            continue
        u_vals, v_vals = uvzf[:, 0], uvzf[:, 1]
        z_vals, frames = uvzf[:, 2], uvzf[:, 3]
        for k in range(len(uvzf)):
            iu = np.searchsorted(u_edges, u_vals[k], side="right") - 1
            iv = np.searchsorted(v_edges, v_vals[k], side="right") - 1
            iu = max(0, min(iu, n_bins - 1))
            iv = max(0, min(iv, n_bins - 1))
            sum_z_elev[iv, iu] += z_vals[k]
            count_z[iv, iu] += 1
        for i in range(len(uvzf) - 1):
            du = u_vals[i + 1] - u_vals[i]
            dv = v_vals[i + 1] - v_vals[i]
            dz = z_vals[i + 1] - z_vals[i]
            dt = (frames[i + 1] - frames[i]) / FPS
            if dt <= 0 or max(0.0, dz) <= 0:
                continue
            weight = max(0.0, dz)
            mid_u = (u_vals[i] + u_vals[i + 1]) / 2
            mid_v = (v_vals[i] + v_vals[i + 1]) / 2
            iu = np.searchsorted(u_edges, mid_u, side="right") - 1
            iv = np.searchsorted(v_edges, mid_v, side="right") - 1
            iu = max(0, min(iu, n_bins - 1))
            iv = max(0, min(iv, n_bins - 1))
            raw_vel = np.sqrt(du * du + dv * dv) / dt
            sum_du[iv, iu] += du * weight
            sum_dv[iv, iu] += dv * weight
            sum_w[iv, iu] += weight
            sum_vel[iv, iu] += raw_vel * weight
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_u = sum_du / sum_w
        flow_v = sum_dv / sum_w
    elev_grid = np.where(count_z > 0, sum_z_elev / count_z, np.nan)
    cu = (u_edges[:-1] + u_edges[1:]) / 2
    cv = (v_edges[:-1] + v_edges[1:]) / 2
    UU, VV = np.meshgrid(cu, cv)
    speed = np.sqrt(flow_u ** 2 + flow_v ** 2)
    scale = np.percentile(speed[speed > 0], 95) / (0.05 * (u_max - u_min)) if (speed > 0).any() else 1.0
    scale = max(scale, 1e-6) * 2.5
    mean_vel = np.where(sum_w > 1e-10, sum_vel / sum_w, 0.0)
    elev_vmin = np.nanmin(elev_grid) if np.any(np.isfinite(elev_grid)) else 0.0
    elev_vmax = np.nanmax(elev_grid) if np.any(np.isfinite(elev_grid)) else 1.0
    return elev_grid, u_edges, v_edges, flow_u, flow_v, UU, VV, mean_vel, scale, elev_vmin, elev_vmax


def _find_two_high_z_peaks(
    elev_grid: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
) -> list[tuple[float, float]]:
    """Find the two highest local maxima; return list of (u_center, v_center) for the two peaks."""
    filled = np.nan_to_num(elev_grid, nan=-np.inf, posinf=0, neginf=-np.inf)
    if not np.any(np.isfinite(elev_grid)):
        return []
    smoothed = ndimage.gaussian_filter(filled, sigma=1.0, mode="constant", cval=-np.inf)
    nv, nu = elev_grid.shape
    peaks = []
    for iv in range(1, nv - 1):
        for iu in range(1, nu - 1):
            v = smoothed[iv, iu]
            if v <= -np.inf or not np.isfinite(v):
                continue
            if v >= smoothed[iv - 1 : iv + 2, iu - 1 : iu + 2].max() and np.isfinite(elev_grid[iv, iu]):
                cu = (u_edges[iu] + u_edges[iu + 1]) / 2
                cv = (v_edges[iv] + v_edges[iv + 1]) / 2
                peaks.append((elev_grid[iv, iu], cu, cv))
    peaks.sort(key=lambda x: -x[0])
    return [(u, v) for (_, u, v) in peaks[:2]]


def _flow_field_uv_highlight_high_z(
    trials_list: list[tuple[Path, str, str | None]],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_path: Path,
    title_suffix: str = "",
    params_override: dict | None = None,
) -> tuple[float, float, float, float, float] | None:
    """Single-panel: elevation + flow with two regions split by v. Returns (v_mid, goal1_u, goal1_v, goal2_u, goal2_v).
    If params_override is provided (e.g. from another animal's midline_and_goals.json), use it for midline and goals
    instead of auto-detecting peaks."""
    global GOAL_RECT_GEOM
    if len(trials_list) == 0:
        return None
    elev_grid, u_edges, v_edges, flow_u, flow_v, UU, VV, _mean_vel, scale, elev_vmin, elev_vmax = _compute_flow_field_grids(
        trials_list, u_min, u_max, v_min, v_max
    )
    if params_override is not None:
        v_mid = float(params_override["v_mid"])
        g1_u = float(params_override["goal1_u"])
        g1_v = float(params_override["goal1_v"])
        g2_u = float(params_override["goal2_u"])
        g2_v = float(params_override["goal2_v"])
        if "half_u" in params_override:
            GOAL_RECT_GEOM = {
                "half_u": float(params_override["half_u"]),
                "top_bottom": float(params_override["top_bottom"]),
                "top_top": float(params_override["top_top"]),
                "bottom_bottom": float(params_override["bottom_bottom"]),
                "bottom_top": float(params_override["bottom_top"]),
            }
        else:
            GOAL_RECT_GEOM = _goal_rect_geom_from_params(v_mid, g1_u, g1_v, g2_u, g2_v, u_min, u_max, v_min, v_max)
        half_u = GOAL_RECT_GEOM["half_u"]
        rect_top_bottom = GOAL_RECT_GEOM["top_bottom"]
        rect_top_top = GOAL_RECT_GEOM["top_top"]
        rect_bottom_bottom = GOAL_RECT_GEOM["bottom_bottom"]
        rect_bottom_top = GOAL_RECT_GEOM["bottom_top"]
        v1, v2 = (g1_v, g2_v) if g1_v < g2_v else (g2_v, g1_v)
    else:
        peaks = _find_two_high_z_peaks(elev_grid, u_edges, v_edges)
        if len(peaks) < 2:
            return None
        (u1, v1), (u2, v2) = peaks[0], peaks[1]
        v_mid = (v1 + v2) / 2.0

        # --- Derive extended goal regions around the two high-z peaks ---
        cu = (u_edges[:-1] + u_edges[1:]) / 2.0
        cv = (v_edges[:-1] + v_edges[1:]) / 2.0
        CU, CV = np.meshgrid(cu, cv)
        elev_vals = elev_grid[np.isfinite(elev_grid)]
        if elev_vals.size == 0:
            return None
        max_elev = float(elev_vals.max())
        z_thresh = 50.0 if max_elev > 50.0 else float(np.percentile(elev_vals, 80.0))
        high_mask = np.isfinite(elev_grid) & (elev_grid >= z_thresh)
        top_mask = high_mask & (CV < v_mid)
        bottom_mask = high_mask & (CV >= v_mid)

        def _half_extents_for_peak(u_peak: float, v_peak: float) -> tuple[float, float]:
            du = (u_max - u_min) * 0.05
            dv_raw = abs(v_mid - v_peak) * 0.45
            max_dv = max(5.0, abs(v_mid - v_peak) - 10.0)
            dv = min(dv_raw, max_dv)
            return float(max(8.0, du)), float(max(10.0, dv))

        half_u1, _ = _half_extents_for_peak(u1, v1)
        half_u2, _ = _half_extents_for_peak(u2, v2)
        half_u = min(half_u1, half_u2)
        top_y = CV[top_mask]
        bottom_y = CV[bottom_mask]
        if top_y.size > 0:
            band_top_bottom = float(np.min(top_y))
            band_top_top = float(min(v_mid - 5.0, np.max(top_y) + 5.0))
        else:
            band_top_bottom = float(max(v_min, v1 - 20.0))
            band_top_top = float(min(v_mid - 5.0, v1 + 20.0))
        if bottom_y.size > 0:
            band_bottom_bottom = float(max(v_mid + 5.0, np.min(bottom_y) - 5.0))
            band_bottom_top = float(min(v_max, np.max(bottom_y) + 5.0))
        else:
            band_bottom_bottom = float(max(v_mid + 5.0, v2 - 20.0))
            band_bottom_top = float(min(v_max, v2 + 20.0))
        base_height = min(
            max(0.0, band_top_top - band_top_bottom),
            max(0.0, band_bottom_top - band_bottom_bottom),
        )
        rect_height = max(15.0, 0.7 * base_height)
        margin = 5.0
        rect_top_top = v_mid - margin
        rect_top_bottom = max(v_min, rect_top_top - rect_height)
        rect_bottom_bottom = v_mid + margin
        rect_bottom_top = min(v_max, rect_bottom_bottom + rect_height)
        GOAL_RECT_GEOM = {
            "half_u": half_u,
            "top_bottom": rect_top_bottom,
            "top_top": rect_top_top,
            "bottom_bottom": rect_bottom_bottom,
            "bottom_top": rect_bottom_top,
        }
        g1_u, g1_v = u1, v1
        g2_u, g2_v = u2, v2

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_max, v_min)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    if np.any(np.isfinite(elev_grid)):
        im = ax.pcolormesh(u_edges, v_edges, elev_grid, cmap="turbo", shading="flat", vmin=elev_vmin, vmax=elev_vmax, alpha=0.9, zorder=0)
        plt.colorbar(im, ax=ax, label="elevation z", shrink=0.7)
    # Two regions: split along v = v_mid. Shade so red region contains (u1,v1), blue contains (u2,v2).
    color_red, color_blue = "#e63946", "#1d3557"
    if v1 < v_mid:  # red dot in top half (smaller v)
        ax.axhspan(v_min, v_mid, alpha=0.25, color=color_red, zorder=1)
        ax.axhspan(v_mid, v_max, alpha=0.25, color=color_blue, zorder=1)
    else:
        ax.axhspan(v_mid, v_max, alpha=0.25, color=color_red, zorder=1)
        ax.axhspan(v_min, v_mid, alpha=0.25, color=color_blue, zorder=1)
    ax.axhline(v_mid, color="black", linewidth=2, zorder=9)
    ax.quiver(UU, VV, flow_u, flow_v, color="black", alpha=0.9, scale_units="xy", scale=scale, width=0.006, headwidth=3, headlength=4, edgecolor="white", linewidth=0.2, zorder=5)
    # Draw high-z goal regions as symmetric rectangles (one per side of the midline).
    def _add_goal_rect(u_peak: float, v_peak: float, color: str) -> None:
        # Horizontal extent is symmetric around u_peak
        left = u_peak - half_u
        width = 2 * half_u
        # Vertical extent: same height on both sides, anchored near the midline,
        # slightly narrower than the full z>=60 band.
        if v_peak < v_mid:
            bottom = rect_top_bottom
            top = rect_top_top
        else:
            bottom = rect_bottom_bottom
            top = rect_bottom_top
        height = max(0.0, top - bottom)
        if height <= 0.0:
            return
        rect = Rectangle((left, bottom), width, height, linewidth=2.0, edgecolor=color, facecolor="none", linestyle="--", zorder=10)
        ax.add_patch(rect)

    _add_goal_rect(g1_u, g1_v, color_red)
    _add_goal_rect(g2_u, g2_v, color_blue)
    # Mark goal points
    ax.plot(g1_u, g1_v, "o", color=color_red, markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.plot(g2_u, g2_v, "o", color=color_blue, markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.annotate("Goal region 1", (g1_u, g1_v), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color=color_red)
    ax.annotate("Goal region 2", (g2_u, g2_v), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color=color_blue)
    ax.set_title("Elevation + flow (u-v) — two regions split by v" + title_suffix)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return (v_mid, g1_u, g1_v, g2_u, g2_v)


def _flow_field_uv_two_panels(
    trials_list: list[tuple[Path, str, str | None]],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Flow field in u-v space: elevation + flow direction, speed + flow direction. v-axis inverted for camera view."""
    if len(trials_list) == 0:
        return
    elev_grid, u_edges, v_edges, flow_u, flow_v, UU, VV, mean_vel, scale, elev_vmin, elev_vmax = _compute_flow_field_grids(
        trials_list, u_min, u_max, v_min, v_max
    )
    speed = np.sqrt(flow_u ** 2 + flow_v ** 2)
    vel_finite = mean_vel[(mean_vel > 0) & np.isfinite(mean_vel)]
    vmin_vel = np.percentile(vel_finite, 10) if len(vel_finite) > 0 else 0.0
    vmax_vel = np.percentile(vel_finite, 95) if len(vel_finite) > 0 else 1.0

    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 8))
    for ax in (ax_left, ax_right):
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)  # invert v for camera view
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")

    if np.any(np.isfinite(elev_grid)):
        im = ax_left.pcolormesh(u_edges, v_edges, elev_grid, cmap="turbo", shading="flat", vmin=elev_vmin, vmax=elev_vmax, alpha=0.9, zorder=0)
        plt.colorbar(im, ax=ax_left, label="elevation z", shrink=0.7)
    q_left = ax_left.quiver(UU, VV, flow_u, flow_v, color="black", alpha=0.9, scale_units="xy", scale=scale, width=0.006, headwidth=3, headlength=4, edgecolor="white", linewidth=0.2, zorder=10)
    ax_left.quiverkey(q_left, 0.88, 0.96, np.percentile(speed[speed > 0], 50) if (speed > 0).any() else 1.0, "flow", coordinates="axes")
    ax_left.set_title("Elevation + flow direction (u-v)" + title_suffix)

    speed_display = np.where(mean_vel > 0, mean_vel, np.nan)
    cmap_turbo = plt.cm.turbo.copy()
    cmap_turbo.set_bad("white")
    im_speed = ax_right.pcolormesh(u_edges, v_edges, speed_display, cmap=cmap_turbo, shading="flat", vmin=vmin_vel, vmax=vmax_vel, alpha=0.9, zorder=0)
    plt.colorbar(im_speed, ax=ax_right, label="speed (px/s)", shrink=0.7)
    q_right = ax_right.quiver(UU, VV, flow_u, flow_v, color="black", alpha=0.9, scale_units="xy", scale=scale, width=0.006, headwidth=3, headlength=4, edgecolor="white", linewidth=0.2, zorder=10)
    ax_right.quiverkey(q_right, 0.88, 0.96, np.percentile(speed[speed > 0], 50) if (speed > 0).any() else 1.0, "flow", coordinates="axes")
    ax_right.set_title("Flow speed + flow direction (u-v)" + title_suffix)

    plt.tight_layout()
    fig2.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _goal_rect_geom_from_params(
    v_mid: float,
    g1_u: float,
    g1_v: float,
    g2_u: float,
    g2_v: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> dict:
    """Compute GOAL_RECT_GEOM from midline and goal points (e.g. loaded from another animal's JSON)."""
    half_u = max(8.0, (u_max - u_min) * 0.05)
    margin = 5.0
    v_above = min(g1_v, g2_v)
    v_below = max(g1_v, g2_v)
    dist_top = max(0.0, v_mid - margin - v_above)
    dist_bottom = max(0.0, v_below - (v_mid + margin))
    rect_height = 0.7 * min(dist_top, dist_bottom) if (dist_top > 0 and dist_bottom > 0) else 80.0
    rect_height = max(rect_height, 15.0)
    rect_top_top = v_mid - margin
    rect_top_bottom = max(v_min, rect_top_top - rect_height)
    rect_bottom_bottom = v_mid + margin
    rect_bottom_top = min(v_max, rect_bottom_bottom + rect_height)
    return {
        "half_u": half_u,
        "top_bottom": rect_top_bottom,
        "top_top": rect_top_top,
        "bottom_bottom": rect_bottom_bottom,
        "bottom_top": rect_bottom_top,
    }


# Midline crossing: when goal positions are given, "crossing" = clearly heading toward the *other* dot
# (past the midpoint between v_mid and that goal). Otherwise fallback to fixed tolerance (px).
MIDLINE_CROSSING_TOLERANCE_PX = 25.0


def _path_crossing_count(
    v_vals: np.ndarray,
    v_mid: float,
    tolerance: float | None = None,
    goal1_v: float | None = None,
    goal2_v: float | None = None,
) -> int:
    """Number of times the path crosses the midline toward the other goal.
    When goal1_v and goal2_v are provided: count only when path goes clearly toward the other dot,
    i.e. past the midpoint between v_mid and that goal (so it's clearly committed to the other side).
    Otherwise use tolerance (px) each side of v_mid. Smaller v = top of image (one goal), larger v = bottom."""
    if len(v_vals) < 2:
        return 0
    if goal1_v is not None and goal2_v is not None:
        v_above_goal = min(goal1_v, goal2_v)
        v_below_goal = max(goal1_v, goal2_v)
        # Past midpoint toward each goal = "clearly on that side"
        v_thresh_toward_top = (v_mid + v_above_goal) / 2.0
        v_thresh_toward_bottom = (v_mid + v_below_goal) / 2.0
        # above = toward top goal (smaller v); below = toward bottom goal (larger v)
        use_toward_top = -1
        use_toward_bottom = 1
    else:
        tol = tolerance if tolerance is not None and tolerance >= 0 else MIDLINE_CROSSING_TOLERANCE_PX
        v_thresh_toward_top = v_mid - tol
        v_thresh_toward_bottom = v_mid + tol
        use_toward_top = -1
        use_toward_bottom = 1

    count = 0
    last_side = None
    for v in v_vals:
        if v <= v_thresh_toward_top:
            side = use_toward_top
        elif v >= v_thresh_toward_bottom:
            side = use_toward_bottom
        else:
            side = 0
        if side != 0:
            if last_side is not None and side != last_side:
                count += 1
            last_side = side
    return count


def _point_goal_region(
    u: float,
    v: float,
    params: dict,
) -> int:
    """Return 1 if (u,v) is inside goal1 rectangle, 2 if inside goal2 rectangle, 0 otherwise.
    Uses rectangular geometry derived in _flow_field_uv_highlight_high_z (GOAL_RECT_GEOM)."""
    geom = GOAL_RECT_GEOM
    if geom is None:
        return 0
    v_mid = params.get("v_mid")
    g1_u, g1_v = params.get("goal1_u"), params.get("goal1_v")
    g2_u, g2_v = params.get("goal2_u"), params.get("goal2_v")
    if v_mid is None or g1_u is None or g1_v is None or g2_u is None or g2_v is None:
        return 0
    half_u = float(geom.get("half_u", 0.0))
    top_bottom = float(geom.get("top_bottom", v_mid))
    top_top = float(geom.get("top_top", v_mid))
    bottom_bottom = float(geom.get("bottom_bottom", v_mid))
    bottom_top = float(geom.get("bottom_top", v_mid))
    # Goal 1 is the one on the same side of v_mid as (goal1_v)
    if g1_v < v_mid:
        # goal1 in top rectangle, goal2 in bottom
        if top_bottom <= v <= top_top and abs(u - g1_u) <= half_u:
            return 1
        if bottom_bottom <= v <= bottom_top and abs(u - g2_u) <= half_u:
            return 2
    else:
        # goal1 in bottom rectangle, goal2 in top
        if bottom_bottom <= v <= bottom_top and abs(u - g1_u) <= half_u:
            return 1
        if top_bottom <= v <= top_top and abs(u - g2_u) <= half_u:
            return 2
    return 0


def _path_crosses_midline(
    v_vals: np.ndarray,
    v_mid: float,
    tolerance: float | None = None,
    goal1_v: float | None = None,
    goal2_v: float | None = None,
) -> bool:
    """True if path clearly crosses the midline toward the other goal (see _path_crossing_count)."""
    return _path_crossing_count(v_vals, v_mid, tolerance, goal1_v, goal2_v) > 0


def _trial_goal_hits(
    df: pd.DataFrame,
    params: dict,
) -> tuple[bool, bool]:
    """Return (hits_goal1, hits_goal2) based on whether the path ever enters each goal rectangle."""
    if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
        return (False, False)
    hit1 = False
    hit2 = False
    for u, v in zip(df["u"].values, df["v"].values):
        lab = _point_goal_region(float(u), float(v), params)
        if lab == 1:
            hit1 = True
        elif lab == 2:
            hit2 = True
        if hit1 and hit2:
            break
    return (hit1, hit2)


def _classify_trials_by_goal_region(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> tuple[list[str], list[str]]:
    """Classify trials that visit only goal1 region or only goal2 region (start → reward)."""
    goal1_only: list[str] = []
    goal2_only: list[str] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df_tr = df.loc[mask].sort_values("frame_number")
        if len(df_tr) < 1:
            continue
        hit1, hit2 = _trial_goal_hits(df_tr, params)
        if hit1 and not hit2:
            goal1_only.append(trial_id)
        elif hit2 and not hit1:
            goal2_only.append(trial_id)
    return (goal1_only, goal2_only)


def _first_goal_visited(
    df: pd.DataFrame,
    params: dict,
) -> int:
    """Return 1 or 2 for the first goal rectangle the path enters, 0 if it never enters either."""
    if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
        return 0
    for u, v in zip(df["u"].values, df["v"].values):
        lab = _point_goal_region(float(u), float(v), params)
        if lab in (1, 2):
            return lab
    return 0


def _get_first_goal_visit_locations(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> list[tuple[int, float, float, int]]:
    """Return (goal_label, u, v, phase_id) for each trial's first entry into either goal rectangle.
    phase_id: 0=early, 1=mid, 2=late."""
    early, mid, late = _split_phase_trials(trials_list)
    trial_to_phase: dict[str, int] = {}
    for t in early:
        trial_to_phase[t[1]] = 0
    for t in mid:
        trial_to_phase[t[1]] = 1
    for t in late:
        trial_to_phase[t[1]] = 2
    out: list[tuple[int, float, float, int]] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df_tr = df.loc[mask].sort_values("frame_number")
        for _, row in df_tr.iterrows():
            u, v = float(row["u"]), float(row["v"])
            lab = _point_goal_region(u, v, params)
            if lab in (1, 2):
                phase_id = trial_to_phase.get(trial_id, 0)
                out.append((lab, u, v, phase_id))
                break
    return out


def _get_first_visit_to_goal_when_other_first_locations(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]], dict[str, tuple[int, int, int]]]:
    """First visit to goal 1 when goal 2 was visited first, and first visit to goal 2 when goal 1 was visited first.
    Only trials that visit BOTH goals are included; trials that visit only one goal are excluded.
    Returns (list of (u, v, phase_id) for first entry to goal 1 in trials that visited goal 2 first,
             list of (u, v, phase_id) for first entry to goal 2 in trials that visited goal 1 first,
             per_phase_counts: phase_name -> (n_both_goals, n_goal2_first_then_goal1, n_goal1_first_then_goal2))."""
    early, mid, late = _split_phase_trials(trials_list)
    trial_to_phase: dict[str, int] = {}
    for t in early:
        trial_to_phase[str(t[1])] = 0
    for t in mid:
        trial_to_phase[str(t[1])] = 1
    for t in late:
        trial_to_phase[str(t[1])] = 2
    phase_names = ["early", "mid", "late"]
    per_phase: dict[str, tuple[int, int, int]] = {p: (0, 0, 0) for p in phase_names}
    first_visit_to_goal1_when_goal2_first: list[tuple[float, float, int]] = []
    first_visit_to_goal2_when_goal1_first: list[tuple[float, float, int]] = []
    for csv_path, trial_id, _ in trials_list:
        trial_id_str = str(trial_id)
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df_tr = df.loc[mask].sort_values("frame_number")
        # Build ordered list of (goal, u, v) at each entry into a goal (transition from 0 or from other goal)
        entries: list[tuple[int, float, float]] = []
        prev_lab = 0
        for _, row in df_tr.iterrows():
            u, v = float(row["u"]), float(row["v"])
            lab = _point_goal_region(u, v, params)
            if lab in (1, 2) and lab != prev_lab:
                entries.append((lab, u, v))
            prev_lab = lab if lab != 0 else prev_lab
        if not entries:
            continue
        # Only count trials that visit BOTH goals (have at least one entry to each)
        has_goal1 = any(g == 1 for g, _, _ in entries)
        has_goal2 = any(g == 2 for g, _, _ in entries)
        if not (has_goal1 and has_goal2):
            continue
        phase_id = trial_to_phase.get(trial_id_str, 0)
        pname = phase_names[phase_id]
        n_both, n_g2_first, n_g1_first = per_phase[pname]
        per_phase[pname] = (n_both + 1, n_g2_first, n_g1_first)
        first_goal = entries[0][0]
        if first_goal == 1:
            for g, u, v in entries:
                if g == 2:
                    first_visit_to_goal2_when_goal1_first.append((u, v, phase_id))
                    per_phase[pname] = (n_both + 1, n_g2_first, n_g1_first + 1)
                    break
        else:
            for g, u, v in entries:
                if g == 1:
                    first_visit_to_goal1_when_goal2_first.append((u, v, phase_id))
                    per_phase[pname] = (n_both + 1, n_g2_first + 1, n_g1_first)
                    break
    return (first_visit_to_goal1_when_goal2_first, first_visit_to_goal2_when_goal1_first, per_phase)


def _goal_rect_bounds(params: dict, goal_label: int) -> tuple[float, float, float, float] | None:
    """Return (u_lo, u_hi, v_lo, v_hi) for the given goal rectangle (1 or 2)."""
    geom = GOAL_RECT_GEOM
    if geom is None:
        return None
    v_mid = params.get("v_mid")
    g1_u, g1_v = params.get("goal1_u"), params.get("goal1_v")
    g2_u, g2_v = params.get("goal2_u"), params.get("goal2_v")
    if v_mid is None or g1_u is None or g2_u is None:
        return None
    half_u = float(geom.get("half_u", 0.0))
    top_bottom = float(geom.get("top_bottom", v_mid))
    top_top = float(geom.get("top_top", v_mid))
    bottom_bottom = float(geom.get("bottom_bottom", v_mid))
    bottom_top = float(geom.get("bottom_top", v_mid))
    if goal_label == 1:
        u_center = g1_u
        if g1_v < v_mid:
            v_lo, v_hi = top_bottom, top_top
        else:
            v_lo, v_hi = bottom_bottom, bottom_top
    else:
        u_center = g2_u
        if g2_v < v_mid:
            v_lo, v_hi = top_bottom, top_top
        else:
            v_lo, v_hi = bottom_bottom, bottom_top
    u_lo = u_center - half_u
    u_hi = u_center + half_u
    return (u_lo, u_hi, v_lo, v_hi)


def _classify_trials_by_first_goal(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> tuple[list[str], list[str]]:
    """Return (first_goal_left_ids, first_goal_right_ids). Left = smaller u; right = larger u."""
    g1_u = params.get("goal1_u")
    g2_u = params.get("goal2_u")
    if g1_u is None or g2_u is None:
        return ([], [])
    left_goal = 1 if g1_u < g2_u else 2
    right_goal = 2 if g1_u < g2_u else 1
    first_left: list[str] = []
    first_right: list[str] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df_tr = df.loc[mask].sort_values("frame_number")
        if len(df_tr) < 1:
            continue
        fg = _first_goal_visited(df_tr, params)
        if fg == left_goal:
            first_left.append(trial_id)
        elif fg == right_goal:
            first_right.append(trial_id)
    return (first_left, first_right)


def _count_goal_regions_visited(
    df: pd.DataFrame,
    params: dict,
) -> int:
    """Return 0, 1, or 2: number of distinct goal rectangles the path enters."""
    hit1, hit2 = _trial_goal_hits(df, params)
    return (1 if hit1 else 0) + (1 if hit2 else 0)


def _get_both_goals_crossing_events(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> list[tuple[str, int, float, float, int, int, int]]:
    """For trials that visit both goals, return (trial_id, frame, u, v, phase_id, frame_start, reward_frame) at first transition.
    phase_id: 0=early, 1=mid, 2=late. Uses chronological order of trials_list for phase assignment."""
    early, mid, late = _split_phase_trials(trials_list)
    trial_to_phase: dict[str, int] = {}
    for t in early:
        trial_to_phase[t[1]] = 0
    for t in mid:
        trial_to_phase[t[1]] = 1
    for t in late:
        trial_to_phase[t[1]] = 2
    events: list[tuple[str, int, float, float, int, int, int]] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df_tr = df.loc[mask].sort_values("frame_number").reset_index(drop=True)
        if len(df_tr) < 2:
            continue
        hit1, hit2 = _trial_goal_hits(df_tr, params)
        if not (hit1 and hit2):
            continue
        # Find first transition from one goal to the other (first point in the second goal)
        prev_lab = 0
        for i in range(len(df_tr)):
            row = df_tr.iloc[i]
            u, v = float(row["u"]), float(row["v"])
            lab = _point_goal_region(u, v, params)
            if lab == 0:
                continue
            if prev_lab != 0 and lab != prev_lab:
                frame = int(row["frame_number"])
                phase_id = trial_to_phase.get(trial_id, 0)
                events.append((trial_id, frame, u, v, phase_id, frame_start, reward_frame))
                break
            prev_lab = lab
    return events


def _path_crossing_u_locations(
    u_vals: np.ndarray,
    v_vals: np.ndarray,
    v_mid: float,
    goal1_v: float | None = None,
    goal2_v: float | None = None,
    tolerance: float | None = None,
) -> list[float]:
    """Return u-coordinates where the path crosses the midline (same goal-based logic as _path_crossing_count)."""
    if len(u_vals) < 2 or len(v_vals) < 2 or len(u_vals) != len(v_vals):
        return []
    if goal1_v is not None and goal2_v is not None:
        v_above_goal = min(goal1_v, goal2_v)
        v_below_goal = max(goal1_v, goal2_v)
        v_thresh_toward_top = (v_mid + v_above_goal) / 2.0
        v_thresh_toward_bottom = (v_mid + v_below_goal) / 2.0
    else:
        tol = tolerance if tolerance is not None and tolerance >= 0 else MIDLINE_CROSSING_TOLERANCE_PX
        v_thresh_toward_top = v_mid - tol
        v_thresh_toward_bottom = v_mid + tol

    def side(v: float) -> int:
        if v <= v_thresh_toward_top:
            return -1
        if v >= v_thresh_toward_bottom:
            return 1
        return 0

    out: list[float] = []
    last_side = side(float(v_vals[0]))
    for i in range(1, len(v_vals)):
        s = side(float(v_vals[i]))
        if s != 0 and last_side != 0 and s != last_side:
            v0, v1 = float(v_vals[i - 1]), float(v_vals[i])
            if v1 != v0:
                t = (v_mid - v0) / (v1 - v0)
                u_cross = float(u_vals[i - 1]) + t * (float(u_vals[i]) - float(u_vals[i - 1]))
                out.append(u_cross)
        if s != 0:
            last_side = s
    return out


def _classify_trials_by_crossing_count(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> dict[int, list[str]]:
    """Return dict: crossing_count -> list of trial_id. Uses goal-based crossing (toward the other dot)."""
    v_mid = params["v_mid"]
    g1_v = params.get("goal1_v")
    g2_v = params.get("goal2_v")
    by_count: dict[int, list[str]] = {}
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "v" not in df.columns:
            by_count.setdefault(0, []).append(trial_id)
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df = df.loc[mask]
        total = 0
        for seg_id in sorted(df["segment_id"].unique()):
            seg = df[df["segment_id"] == seg_id].sort_values("frame_number")
            v = seg["v"].values
            total += _path_crossing_count(v, v_mid, goal1_v=g1_v, goal2_v=g2_v)
        by_count.setdefault(total, []).append(trial_id)
    return by_count


def _classify_trials_by_midline_crossing(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> tuple[list[str], list[str]]:
    """Return (trial_ids_that_cross, trial_ids_that_dont_cross). Uses goal-based crossing."""
    v_mid = params["v_mid"]
    g1_v = params.get("goal1_v")
    g2_v = params.get("goal2_v")
    crosses: list[str] = []
    does_not: list[str] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 1 or "v" not in df.columns:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df = df.loc[mask]
        if len(df) < 2:
            does_not.append(trial_id)
            continue
        any_cross = False
        for seg_id in sorted(df["segment_id"].unique()):
            seg = df[df["segment_id"] == seg_id].sort_values("frame_number")
            v = seg["v"].values
            if len(v) >= 2 and _path_crosses_midline(v, v_mid, goal1_v=g1_v, goal2_v=g2_v):
                any_cross = True
                break
        if any_cross:
            crosses.append(trial_id)
        else:
            does_not.append(trial_id)
    return (crosses, does_not)


def _late_phase_trials(
    trials_list: list[tuple[Path, str, str | None]],
) -> list[tuple[Path, str, str | None]]:
    """Return the last third of trials when ordered chronologically (session_folder, trial_id). Same logic as plot_phase_vs_side early/mid/late."""
    early, mid, late = _split_phase_trials(trials_list)
    return late


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


def _load_trial_type_sets(trial_types_path: Path) -> tuple[set[str], set[str]] | None:
    """Load trial_types.csv; return (vertical_left_trial_ids, vertical_right_trial_ids) or None.
    vertical_left = left_angle_deg==360 and right!=360; vertical_right = right==360 and left!=360."""
    trial_types_path = Path(trial_types_path)
    if not trial_types_path.is_file():
        return None
    try:
        tt = pd.read_csv(trial_types_path)
    except Exception:
        return None
    if "trial_id" not in tt.columns or "left_angle_deg" not in tt.columns or "right_angle_deg" not in tt.columns:
        return None
    vl, vr = set(), set()
    for _, row in tt.iterrows():
        lid, rid = row.get("left_angle_deg"), row.get("right_angle_deg")
        tid = str(row["trial_id"]).strip()
        if pd.isna(lid) or pd.isna(rid):
            continue
        try:
            lf, rf = float(lid), float(rid)
        except (TypeError, ValueError):
            continue
        if lf == 360.0 and rf != 360.0:
            vl.add(tid)
        elif rf == 360.0 and lf != 360.0:
            vr.add(tid)
    return (vl, vr)


def _collect_crossing_u_locations(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
) -> list[float]:
    """Collect u-coordinates of all midline crossings (trial start → reward) for trials that cross at least once."""
    v_mid = params["v_mid"]
    g1_v = params.get("goal1_v")
    g2_v = params.get("goal2_v")
    all_u: list[float] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df = df.loc[mask].sort_values("frame_number")
        if len(df) < 2:
            continue
        u_vals = df["u"].values
        v_vals = df["v"].values
        u_crossings = _path_crossing_u_locations(u_vals, v_vals, v_mid, goal1_v=g1_v, goal2_v=g2_v)
        all_u.extend(u_crossings)
    return all_u


def _trial_goal_switch_direction(
    df: pd.DataFrame,
    params: dict,
) -> int:
    """Return 1 if path clearly goes goal1→goal2 and ends in goal2, 2 if goal2→goal1 and ends in goal1, 0 otherwise.

    Additional constraints:
    - Only consider trials where the whole path (start→reward) stays at low elevation (z < 20).
    - Require exactly one clear switch: segments of non-zero goal labels must be [1, 2] or [2, 1],
      with each segment lasting at least a few frames.
    """
    if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
        return 0
    if "z" not in df.columns:
        return 0
    # Path must stay low in elevation
    if (df["z"].values > 20.0).any():
        return 0

    seg_labels: list[int] = []
    seg_lengths: list[int] = []
    for u, v in zip(df["u"].values, df["v"].values):
        lab = _point_goal_region(float(u), float(v), params)
        if lab == 0:
            continue
        if not seg_labels or lab != seg_labels[-1]:
            seg_labels.append(lab)
            seg_lengths.append(1)
        else:
            seg_lengths[-1] += 1

    if len(seg_labels) != 2:
        return 0
    # Require each segment to be reasonably long
    MIN_RUN = 5
    if seg_lengths[0] < MIN_RUN or seg_lengths[1] < MIN_RUN:
        return 0

    if seg_labels == [1, 2]:
        return 1  # goal1 -> goal2, end near goal2
    if seg_labels == [2, 1]:
        return 2  # goal2 -> goal1, end near goal1
    return 0


def _plot_paths_goal_switches(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> None:
    """Plot all paths (start → reward) that visit one goal rectangle then the other.
    Red paths: goal1 → goal2; blue paths: goal2 → goal1.
    Only segments inside goal rectangles (labels 1 or 2) are drawn; '0' segments are omitted."""
    ids_1_to_2: list[str] = []
    ids_2_to_1: list[str] = []
    for csv_path, trial_id, _ in trials_list:
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        df = at.load_trajectory_csv(csv_path)
        if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
            continue
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df = df.loc[mask].sort_values("frame_number")
        direction = _trial_goal_switch_direction(df, params)
        if direction == 1:
            ids_1_to_2.append(trial_id)
        elif direction == 2:
            ids_2_to_1.append(trial_id)
    if not ids_1_to_2 and not ids_2_to_1:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_max, v_min)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    # Draw only segments inside goal rectangles (labels 1 or 2); skip label 0 segments entirely.
    def _plot_runs_for_ids(trial_ids: list[str], color: str) -> None:
        id_set = set(trial_ids)
        if not id_set:
            return
        for csv_path, trial_id, _ in trials_list:
            if trial_id not in id_set:
                continue
            reward_frame = reward_frame_by_trial.get(trial_id)
            if reward_frame is None:
                continue
            frames = _parse_trial_frames(trial_id)
            if not frames:
                continue
            frame_start, _ = frames
            df = at.load_trajectory_csv(csv_path)
            if len(df) < 2 or "u" not in df.columns or "v" not in df.columns:
                continue
            mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
            df_trial = df.loc[mask].sort_values("frame_number")
            if len(df_trial) < 2:
                continue
            u_vals = df_trial["u"].values
            v_vals = df_trial["v"].values
            current_u: list[float] = []
            current_v: list[float] = []
            for u, v in zip(u_vals, v_vals):
                lab = _point_goal_region(float(u), float(v), params)
                if lab == 0:
                    if len(current_u) >= 2:
                        ax.plot(current_u, current_v, color=color, alpha=0.85, linewidth=linewidth, zorder=2)
                    current_u, current_v = [], []
                else:
                    current_u.append(float(u))
                    current_v.append(float(v))
            if len(current_u) >= 2:
                ax.plot(current_u, current_v, color=color, alpha=0.85, linewidth=linewidth, zorder=2)

    if ids_1_to_2:
        _plot_runs_for_ids(ids_1_to_2, "#e63946")
    if ids_2_to_1:
        _plot_runs_for_ids(ids_2_to_1, "#1d3557")
    v_mid = params["v_mid"]
    g1_u, g1_v = params["goal1_u"], params["goal1_v"]
    g2_u, g2_v = params["goal2_u"], params["goal2_v"]
    ax.axhline(v_mid, color="black", linewidth=2, zorder=9)
    ax.plot(g1_u, g1_v, "o", color="#e63946", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.plot(g2_u, g2_v, "o", color="#1d3557", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.set_title(f"Paths that switch goal regions (goal1→goal2: n={len(ids_1_to_2)}, goal2→goal1: n={len(ids_2_to_1)})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_paths_by_goal_region_exclusive(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> None:
    """Two-panel figure: paths that visit only goal1 region vs only goal2 region (start → reward)."""
    goal1_only, goal2_only = _classify_trials_by_goal_region(trials_list, reward_frame_by_trial, params)
    if not goal1_only and not goal2_only:
        return
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, gridspec_kw={"wspace": 0.04})
    for ax in (ax_left, ax_right):
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
    v_mid = params["v_mid"]
    g1_u, g1_v = params["goal1_u"], params["goal1_v"]
    g2_u, g2_v = params["goal2_u"], params["goal2_v"]
    for ax in (ax_left, ax_right):
        ax.axhline(v_mid, color="black", linewidth=2, zorder=9)
        ax.plot(g1_u, g1_v, "o", color="#e63946", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
        ax.plot(g2_u, g2_v, "o", color="#1d3557", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)

    if goal1_only:
        _plot_paths_subset(trials_list, goal1_only, reward_frame_by_trial, params, ax_left, u_min, u_max, v_min, v_max, fixed_color="#e63946")
    if goal2_only:
        _plot_paths_subset(trials_list, goal2_only, reward_frame_by_trial, params, ax_right, u_min, u_max, v_min, v_max, fixed_color="#1d3557")

    ax_left.set_title(f"Paths that visit only goal1 region (n={len(goal1_only)})")
    ax_right.set_title(f"Paths that visit only goal2 region (n={len(goal2_only)})")
    ax_right.label_outer()
    fig.suptitle("Paths: trial start → reward — exclusive goal regions", y=0.98)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_paths_by_first_goal(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    *,
    title_suffix: str = "",
) -> None:
    """Two-panel figure: full path (start→reward) for trials whose first goal visited is left vs right (by u)."""
    first_left, first_right = _classify_trials_by_first_goal(trials_list, reward_frame_by_trial, params)
    if not first_left and not first_right:
        return
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, gridspec_kw={"wspace": 0.04})
    for ax in (ax_left, ax_right):
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
    v_mid = params["v_mid"]
    g1_u, g1_v = params["goal1_u"], params["goal1_v"]
    g2_u, g2_v = params["goal2_u"], params["goal2_v"]
    for ax in (ax_left, ax_right):
        ax.axhline(v_mid, color="black", linewidth=2, zorder=9)
        ax.plot(g1_u, g1_v, "o", color="#e63946", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
        ax.plot(g2_u, g2_v, "o", color="#1d3557", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    if first_left:
        _plot_paths_subset(trials_list, first_left, reward_frame_by_trial, params, ax_left, u_min, u_max, v_min, v_max, fixed_color="#2a9d8f")
    if first_right:
        _plot_paths_subset(trials_list, first_right, reward_frame_by_trial, params, ax_right, u_min, u_max, v_min, v_max, fixed_color="#e76f51")
    ax_left.set_title(f"First goal visited: left (n={len(first_left)})")
    ax_right.set_title(f"First goal visited: right (n={len(first_right)})")
    ax_right.label_outer()
    suptitle = "Paths: trial start → reward — first goal left vs right"
    if title_suffix:
        suptitle += " — " + title_suffix
    fig.suptitle(suptitle, y=0.98)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_goal_regions_visited_histogram(
    early_trials: list[tuple[Path, str, str | None]],
    mid_trials: list[tuple[Path, str, str | None]],
    late_trials: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    *,
    animal: str = "rory",
) -> None:
    """Bar chart: number of goal regions visited (0, 1, 2) for early, mid, late phase (3 subplots)."""
    def _counts_for_phase(trials: list) -> tuple[int, int, int]:
        n0, n1, n2 = 0, 0, 0
        for csv_path, trial_id, _ in trials:
            reward_frame = reward_frame_by_trial.get(trial_id)
            if reward_frame is None:
                continue
            frames = _parse_trial_frames(trial_id)
            if not frames:
                continue
            frame_start, _ = frames
            df = at.load_trajectory_csv(csv_path)
            if len(df) < 1 or "u" not in df.columns or "v" not in df.columns:
                n0 += 1
                continue
            mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
            df_tr = df.loc[mask].sort_values("frame_number")
            n_reg = _count_goal_regions_visited(df_tr, params)
            if n_reg == 0:
                n0 += 1
            elif n_reg == 1:
                n1 += 1
            else:
                n2 += 1
        return (n0, n1, n2)

    c_early = _counts_for_phase(early_trials)
    c_mid = _counts_for_phase(mid_trials)
    c_late = _counts_for_phase(late_trials)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, counts, label in zip(axes, [c_early, c_mid, c_late], ["Early", "Mid", "Late"]):
        n0, n1, n2 = counts
        ax.bar([0, 1, 2], [n0, n1, n2], color=["#94a3b8", "#3b82f6", "#22c55e"], edgecolor="black")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0 regions", "1 region", "2 regions"])
        ax.set_ylabel("Number of trials")
        ax.set_title(f"{label} (n={n0 + n1 + n2})")
    fig.suptitle(f"Goal regions visited per trial ({animal})", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_both_goals_when_and_where(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> None:
    """Two-panel: (1) when do trials that visit both goals first cross (normalized time 0–1); (2) where in u–v do they cross."""
    events = _get_both_goals_crossing_events(trials_list, reward_frame_by_trial, params)
    if not events:
        return
    # Unpack: trial_id, frame, u, v, phase_id, frame_start, reward_frame
    time_norm = []
    phase_ids = []
    u_vals = []
    v_vals = []
    for ev in events:
        _, frame, u, v, phase_id, frame_start, reward_frame = ev
        dur = max(1, reward_frame - frame_start)
        time_norm.append((frame - frame_start) / dur)
        phase_ids.append(phase_id)
        u_vals.append(u)
        v_vals.append(v)
    phase_names = ["Early", "Mid", "Late"]
    phase_colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    fig, (ax_when, ax_where) = plt.subplots(1, 2, figsize=(14, 6))
    # When: histogram of normalized time at first cross, by phase
    for phase_id in [0, 1, 2]:
        t = [time_norm[i] for i in range(len(time_norm)) if phase_ids[i] == phase_id]
        if not t:
            continue
        ax_when.hist(t, bins=15, range=(0, 1), alpha=0.6, label=f"{phase_names[phase_id]} (n={len(t)})", color=phase_colors[phase_id], edgecolor="black")
    ax_when.set_xlabel("Normalized time in trial (0 = start, 1 = reward)")
    ax_when.set_ylabel("Number of trials")
    ax_when.set_title("When do mice first cross from one goal to the other?")
    ax_when.legend()
    ax_when.set_xlim(0, 1)
    # Where: scatter (u, v) of crossing point, colored by phase
    v_mid = params["v_mid"]
    g1_u, g1_v = params["goal1_u"], params["goal1_v"]
    g2_u, g2_v = params["goal2_u"], params["goal2_v"]
    for phase_id in [0, 1, 2]:
        us = [u_vals[i] for i in range(len(u_vals)) if phase_ids[i] == phase_id]
        vs = [v_vals[i] for i in range(len(v_vals)) if phase_ids[i] == phase_id]
        if not us:
            continue
        ax_where.scatter(us, vs, alpha=0.7, s=40, label=f"{phase_names[phase_id]} (n={len(us)})", color=phase_colors[phase_id], edgecolors="black")
    ax_where.axhline(v_mid, color="black", linewidth=2, zorder=0)
    ax_where.plot(g1_u, g1_v, "o", color="#e63946", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=2)
    ax_where.plot(g2_u, g2_v, "o", color="#1d3557", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=2)
    ax_where.set_xlim(u_min, u_max)
    ax_where.set_ylim(v_max, v_min)
    ax_where.set_aspect("equal")
    ax_where.set_xlabel("u (px)")
    ax_where.set_ylabel("v (px)")
    ax_where.set_title("Where in u–v do they cross? (first transition)")
    ax_where.legend()
    fig.suptitle(f"Trials that visit both goals (total n={len(events)})", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_first_goal_visit_heatmaps(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    *,
    n_bins: int = 15,
) -> None:
    """U–v heatmap of where the mouse first visits each goal rectangle, by phase.
    Layout: 3 rows (early, mid, late) × 2 columns (goal1, goal2). Each cell shows the rectangle
    and a density heatmap of first-visit (u,v) points for that goal and phase."""
    events = _get_first_goal_visit_locations(trials_list, reward_frame_by_trial, params)
    if not events:
        return
    phase_names = ["Early", "Mid", "Late"]
    goal_names = ["Goal 1", "Goal 2"]
    all_vals: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for goal in (1, 2):
        for phase_id in (0, 1, 2):
            pts = [(u, v) for (g, u, v, p) in events if g == goal and p == phase_id]
            all_vals[(goal, phase_id)] = pts
    bounds1 = _goal_rect_bounds(params, 1)
    bounds2 = _goal_rect_bounds(params, 2)
    if bounds1 is None or bounds2 is None:
        return
    u_lo1, u_hi1, v_lo1, v_hi1 = bounds1
    u_lo2, u_hi2, v_lo2, v_hi2 = bounds2
    margin_u = max(5.0, (u_hi1 - u_lo1) * 0.05)
    margin_v = max(5.0, (v_hi1 - v_lo1) * 0.05)
    # Compute all histograms; each panel gets its own vmax so spatial structure is visible
    hist_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for phase_id in range(3):
        for goal in (1, 2):
            pts = all_vals.get((goal, phase_id), [])
            if goal == 1:
                u_lo, u_hi, v_lo, v_hi = u_lo1, u_hi1, v_lo1, v_hi1
            else:
                u_lo, u_hi, v_lo, v_hi = u_lo2, u_hi2, v_lo2, v_hi2
            if pts:
                us = [p[0] for p in pts]
                vs = [p[1] for p in pts]
                u_edges = np.linspace(u_lo, u_hi, n_bins + 1)
                v_edges = np.linspace(v_lo, v_hi, n_bins + 1)
                H, _, _ = np.histogram2d(us, vs, bins=(u_edges, v_edges))
                H = H.T
                vmax_cell = max(1.0, float(np.nanmax(H)) if np.any(H > 0) else 1.0)
                hist_data[(phase_id, goal)] = (H, u_edges, v_edges, vmax_cell)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
    im = None
    for phase_id in range(3):
        for goal in (1, 2):
            ax = axes[phase_id, goal - 1]
            if goal == 1:
                u_lo, u_hi, v_lo, v_hi = u_lo1, u_hi1, v_lo1, v_hi1
            else:
                u_lo, u_hi, v_lo, v_hi = u_lo2, u_hi2, v_lo2, v_hi2
            ax.set_xlim(u_lo - margin_u, u_hi + margin_u)
            ax.set_ylim(v_hi + margin_v, v_lo - margin_v)
            ax.set_aspect("equal")
            rect = Rectangle((u_lo, v_lo), u_hi - u_lo, v_hi - v_lo, linewidth=2, edgecolor="black", facecolor="none", zorder=2)
            ax.add_patch(rect)
            pts = all_vals.get((goal, phase_id), [])
            if (phase_id, goal) in hist_data:
                H, u_edges, v_edges, vmax_cell = hist_data[(phase_id, goal)]
                norm = plt.Normalize(vmin=0, vmax=vmax_cell)
                im = ax.pcolormesh(u_edges, v_edges, H, cmap="YlOrRd", norm=norm, shading="flat", alpha=0.85, zorder=1)
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")
            ax.set_title(f"{phase_names[phase_id]} — {goal_names[goal - 1]} (n={len(pts)})")
    if im is not None:
        fig.colorbar(im, ax=axes[0, 0], label="Count (scale per panel)", shrink=0.6)
    fig.suptitle("First visit to each goal: u–v heatmap by phase (each panel scaled to its own max)", y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_first_visit_to_goal_when_other_first_heatmaps(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    *,
    n_bins: int = 15,
) -> None:
    """U–v heatmap: first visit to goal 1 when goal 2 was visited first, and first visit to goal 2 when goal 1 was visited first.
    Only trials that visit both goals are included (single-goal trials excluded).
    Layout: 3 rows (early, mid, late) × 2 columns (Goal 1, Goal 2). Col 1 = where they first enter goal 1 in trials that
    visited goal 2 first; col 2 = where they first enter goal 2 in trials that visited goal 1 first."""
    goal1_pts, goal2_pts, per_phase = _get_first_visit_to_goal_when_other_first_locations(trials_list, reward_frame_by_trial, params)
    if not goal1_pts and not goal2_pts:
        return
    phase_names = ["Early", "Mid", "Late"]
    # (phase_id, goal) -> list of (u, v); goal 1 = first col, goal 2 = second col
    all_vals: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for phase_id in (0, 1, 2):
        all_vals[(phase_id, 1)] = [(u, v) for (u, v, p) in goal1_pts if p == phase_id]
        all_vals[(phase_id, 2)] = [(u, v) for (u, v, p) in goal2_pts if p == phase_id]
    bounds1 = _goal_rect_bounds(params, 1)
    bounds2 = _goal_rect_bounds(params, 2)
    if bounds1 is None or bounds2 is None:
        return
    u_lo1, u_hi1, v_lo1, v_hi1 = bounds1
    u_lo2, u_hi2, v_lo2, v_hi2 = bounds2
    margin_u = max(5.0, (u_hi1 - u_lo1) * 0.05)
    margin_v = max(5.0, (v_hi1 - v_lo1) * 0.05)
    hist_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for phase_id in range(3):
        for goal in (1, 2):
            pts = all_vals.get((goal, phase_id), [])
            if goal == 1:
                u_lo, u_hi, v_lo, v_hi = u_lo1, u_hi1, v_lo1, v_hi1
            else:
                u_lo, u_hi, v_lo, v_hi = u_lo2, u_hi2, v_lo2, v_hi2
            if pts:
                us = [p[0] for p in pts]
                vs = [p[1] for p in pts]
                u_edges = np.linspace(u_lo, u_hi, n_bins + 1)
                v_edges = np.linspace(v_lo, v_hi, n_bins + 1)
                H, _, _ = np.histogram2d(us, vs, bins=(u_edges, v_edges))
                H = H.T
                vmax_cell = max(1.0, float(np.nanmax(H)) if np.any(H > 0) else 1.0)
                hist_data[(phase_id, goal)] = (H, u_edges, v_edges, vmax_cell)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
    im = None
    for phase_id in range(3):
        for goal in (1, 2):
            ax = axes[phase_id, goal - 1]
            if goal == 1:
                u_lo, u_hi, v_lo, v_hi = u_lo1, u_hi1, v_lo1, v_hi1
            else:
                u_lo, u_hi, v_lo, v_hi = u_lo2, u_hi2, v_lo2, v_hi2
            ax.set_xlim(u_lo - margin_u, u_hi + margin_u)
            ax.set_ylim(v_hi + margin_v, v_lo - margin_v)
            ax.set_aspect("equal")
            rect = Rectangle((u_lo, v_lo), u_hi - u_lo, v_hi - v_lo, linewidth=2, edgecolor="black", facecolor="none", zorder=2)
            ax.add_patch(rect)
            pts = all_vals.get((goal, phase_id), [])
            if (phase_id, goal) in hist_data:
                H, u_edges, v_edges, vmax_cell = hist_data[(phase_id, goal)]
                norm = plt.Normalize(vmin=0, vmax=float(vmax_cell))
                im = ax.pcolormesh(u_edges, v_edges, H, cmap="YlOrRd", norm=norm, shading="flat", alpha=0.85, zorder=1)
                ax.text(0.02, 0.98, f"max={int(round(vmax_cell))}", transform=ax.transAxes, fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")
            if goal == 1:
                ax.set_title(f"{phase_names[phase_id]} — first visit to Goal 1 (after visiting G2 first) (n={len(pts)})")
            else:
                ax.set_title(f"{phase_names[phase_id]} — first visit to Goal 2 (after visiting G1 first) (n={len(pts)})")
    if im is not None:
        fig.colorbar(im, ax=axes[0, 0], label="Count (scale per panel)", shrink=0.6)
    # Per-phase summary: trials that visit both goals (only those are included; single-goal trials excluded)
    summary_parts = []
    for pname in ["early", "mid", "late"]:
        n_both, n_g2_first, n_g1_first = per_phase.get(pname, (0, 0, 0))
        summary_parts.append(f"{pname}: {n_both} both-goal trials (G2→G1: {n_g2_first}, G1→G2: {n_g1_first})")
    summary_str = "; ".join(summary_parts)
    fig.suptitle(
        "First visit to each goal when the other was visited first (only trials that visit both goals; each panel scaled to its own max)\n"
        + summary_str,
        y=1.00,
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_crossing_locations_on_midline(
    crossing_u: list[float],
    params: dict,
    u_min: float,
    u_max: float,
    out_path: Path,
    *,
    title_suffix: str = "",
    n_bins: int = 40,
    animal: str = "rory",
) -> None:
    """Plot histogram of crossing locations along the midline (u-axis). Goals marked as vertical lines."""
    if not crossing_u:
        return
    g1_u = params.get("goal1_u")
    g2_u = params.get("goal2_u")
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.hist(crossing_u, bins=n_bins, range=(u_min, u_max), color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xlim(u_min, u_max)
    ax.set_xlabel("u (px) — position along midline")
    ax.set_ylabel("Count of crossings")
    if g1_u is not None:
        ax.axvline(g1_u, color="#e63946", linewidth=2, label="goal 1")
    if g2_u is not None:
        ax.axvline(g2_u, color="#1d3557", linewidth=2, label="goal 2")
    if g1_u is not None or g2_u is not None:
        ax.legend(loc="upper right")
    title = f"Midline crossing locations ({animal})"
    if title_suffix:
        title += " — " + title_suffix
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_crossing_locations_combined(
    crossing_u_early_all: list[float],
    crossing_u_early_vl: list[float],
    crossing_u_early_vr: list[float],
    crossing_u_mid_all: list[float],
    crossing_u_mid_vl: list[float],
    crossing_u_mid_vr: list[float],
    crossing_u_late_all: list[float],
    crossing_u_late_vl: list[float],
    crossing_u_late_vr: list[float],
    params: dict,
    u_min: float,
    u_max: float,
    out_path: Path,
    n_bins: int = 40,
    *,
    n_early: int = 0,
    n_mid: int = 0,
    n_late: int = 0,
    animal: str = "rory",
) -> None:
    """Plot 3 rows (early / mid / late). Each row = one axes with 2 overlaid histograms: vertical left, vertical right. Shared y-axis scale."""
    g1_u = params.get("goal1_u")
    g2_u = params.get("goal2_u")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
    row_configs = [
        ("early", n_early, crossing_u_early_vl, crossing_u_early_vr),
        ("mid", n_mid, crossing_u_mid_vl, crossing_u_mid_vr),
        ("late", n_late, crossing_u_late_vl, crossing_u_late_vr),
    ]
    for row, (phase_name, n_trials, u_vl, u_vr) in enumerate(row_configs):
        ax = axes[row]
        series = [
            (u_vl, "vertical left", "green", 0.5),
            (u_vr, "vertical right", "purple", 0.5),
        ]
        for data, label, color, alpha in series:
            if data:
                ax.hist(data, bins=n_bins, range=(u_min, u_max), color=color, alpha=alpha, label=label, edgecolor="white", linewidth=0.5)
        ax.set_xlim(u_min, u_max)
        if g1_u is not None:
            ax.axvline(g1_u, color="#e63946", linewidth=1.2, linestyle="--")
        if g2_u is not None:
            ax.axvline(g2_u, color="#1d3557", linewidth=1.2, linestyle="--")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{phase_name} (n={n_trials})", fontsize=11)
    axes[2].set_xlabel("u (px) — position along midline")
    fig.suptitle(f"Midline crossing locations ({animal}) — vertical left vs vertical right by phase", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _draw_elevation_background(
    ax: plt.Axes,
    elev_grid: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    elev_vmin: float,
    elev_vmax: float,
    alpha: float = 0.6,
) -> None:
    """Draw average elevation (all trials) as pcolormesh background; high-z regions show as warm colors."""
    if np.any(np.isfinite(elev_grid)):
        ax.pcolormesh(u_edges, v_edges, elev_grid, cmap="turbo", shading="flat", vmin=elev_vmin, vmax=elev_vmax, alpha=alpha, zorder=0)


def _plot_paths_subset(
    trials_list: list[tuple[Path, str, str | None]],
    trial_ids_subset: list[str],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    ax: plt.Axes,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    *,
    linewidth: float = 1.0,
    alpha: float = 0.75,
    fixed_color: str | None = None,
) -> None:
    """Draw paths for trials in trial_ids_subset as thin lines; each trial gets a distinct color."""
    subset_set = set(trial_ids_subset)
    if not subset_set:
        return
    sorted_ids = sorted(subset_set)
    if fixed_color is None:
        cmap = plt.cm.get_cmap("turbo", max(len(sorted_ids), 1))
        if len(sorted_ids) == 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(i / (len(sorted_ids) - 1)) for i in range(len(sorted_ids))]
        id_to_color = dict(zip(sorted_ids, colors))
    for csv_path, trial_id, _ in trials_list:
        if trial_id not in subset_set:
            continue
        reward_frame = reward_frame_by_trial.get(trial_id)
        if reward_frame is None:
            continue
        color = fixed_color if fixed_color is not None else id_to_color[trial_id]
        df = at.load_trajectory_csv(csv_path)
        frames = _parse_trial_frames(trial_id)
        if not frames:
            continue
        frame_start, _ = frames
        mask = (df["frame_number"] >= frame_start) & (df["frame_number"] <= reward_frame)
        df = df.loc[mask].sort_values("frame_number")
        if len(df) < 2:
            continue
        u, v = df["u"].values, df["v"].values
        ax.plot(u, v, color=color, alpha=alpha, linewidth=linewidth, zorder=2)


def _plot_paths_by_crossing_count(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    *,
    title_suffix: str = "",
    animal: str = "rory",
) -> None:
    """Grid of panels by number of midline crossings: 0, 1, 2, 3, 4, 5+. Trial start → reward."""
    v_mid = params["v_mid"]
    g1_u, g1_v = params["goal1_u"], params["goal1_v"]
    g2_u, g2_v = params["goal2_u"], params["goal2_v"]

    by_count = _classify_trials_by_crossing_count(trials_list, reward_frame_by_trial, params)
    # Panels: 0, 1, 2, …, 10, 11+ (12 panels = 3x4)
    panel_labels = [
        "0 crossings", "1 crossing", "2 crossings", "3 crossings", "4 crossings", "5 crossings",
        "6 crossings", "7 crossings", "8 crossings", "9 crossings", "10 crossings", "11+ crossings",
    ]
    n_panels = 12

    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True, sharey=True, gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    axes_flat = axes.flatten()
    for k in range(n_panels):
        ax = axes_flat[k]
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_max, v_min)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.axhline(v_mid, color="black", linewidth=2, zorder=9)
        ax.plot(g1_u, g1_v, "o", color="#e63946", markersize=6, markeredgecolor="white", markeredgewidth=1, zorder=11)
        ax.plot(g2_u, g2_v, "o", color="#1d3557", markersize=6, markeredgecolor="white", markeredgewidth=1, zorder=11)
        if k < 11:
            trial_ids = by_count.get(k, [])
        else:
            trial_ids = []
            for c in by_count:
                if c >= 11:
                    trial_ids.extend(by_count[c])
        _plot_paths_subset(trials_list, trial_ids, reward_frame_by_trial, params, ax, u_min, u_max, v_min, v_max)
        ax.set_title(f"{panel_labels[k]} (n={len(trial_ids)})")
    for ax in axes_flat:
        ax.label_outer()
    suptitle = f"Trial start → reward by number of midline crossings ({animal})"
    if title_suffix:
        suptitle += " — " + title_suffix
    fig.suptitle(suptitle, y=1.00)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.04)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_paths_trial_start_to_reward(
    trials_list: list[tuple[Path, str, str | None]],
    reward_frame_by_trial: dict[str, int],
    params: dict,
    out_path: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    *,
    animal: str = "rory",
) -> None:
    """Plot trajectory paths from trial start to reward: each trial a different color; midline and goals; no background, no colorbar."""
    v_mid = params["v_mid"]
    g1_u, g1_v = params["goal1_u"], params["goal1_v"]
    g2_u, g2_v = params["goal2_u"], params["goal2_v"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_max, v_min)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    all_trial_ids = [t[1] for t in trials_list]
    _plot_paths_subset(trials_list, all_trial_ids, reward_frame_by_trial, params, ax, u_min, u_max, v_min, v_max)
    ax.axhline(v_mid, color="black", linewidth=2, zorder=9)
    ax.plot(g1_u, g1_v, "o", color="#e63946", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.plot(g2_u, g2_v, "o", color="#1d3557", markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.set_title(f"Paths: trial start → reward ({animal})")
    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Midline-and-goals analysis per animal → trajectory_analysis/<animal>/midline_and_goals/")
    parser.add_argument("--animal", type=str, default="rory", help="Animal name, e.g. rory or wilfred (session folder must start with <animal>_)")
    parser.add_argument("--predictions-root", type=Path, default=Path("/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("trajectory_analysis"))
    parser.add_argument("--reward-times", type=Path, default=None, help="CSV from cbot_climb_log/export_reward_times.py (default: <output-dir>/reward_times.csv)")
    parser.add_argument("--logs-dir", type=Path, default=None, help="cbot_climb_log logs dir; used only if --reward-times not provided or file missing")
    parser.add_argument("--trial-types", type=Path, default=None, help="trial_types.csv for vertical left/right (default: <output-dir>/trial_types.csv)")
    parser.add_argument("--midline-goals-json", type=Path, default=None, help="Use midline and goal positions from this JSON (e.g. rory's midline_and_goals.json) instead of auto-detecting")
    args = parser.parse_args()

    animal = args.animal.strip()
    predictions_root = Path(args.predictions_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_dir = out_root / animal / "midline_and_goals"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = get_animal_trials(predictions_root, animal)
    if not trials:
        print(f"No {animal} trials found (session folder must start with '{animal}_')")
        return
    u_min, u_max, v_min, v_max = uv_limits(trials)
    out_path = out_dir / "flow_field_low_to_high.png"
    _flow_field_uv_two_panels(trials, u_min, u_max, v_min, v_max, out_path, f" ({animal})")
    print(f"  elevation + flow (u-v) -> {out_path}")
    highlight_path = out_dir / "flow_field_high_z_regions.png"
    params_override: dict | None = None
    if getattr(args, "midline_goals_json", None) is not None:
        override_path = Path(args.midline_goals_json).resolve()
        if override_path.is_file():
            with open(override_path) as f:
                params_override = json.load(f)
            print(f"  using midline + goals from {override_path}")
    params_tuple = _flow_field_uv_highlight_high_z(trials, u_min, u_max, v_min, v_max, highlight_path, f" ({animal})", params_override=params_override)
    print(f"  high-z regions (midline + goals) -> {highlight_path}")

    if params_tuple is not None:
        v_mid, g1_u, g1_v, g2_u, g2_v = params_tuple
        params = {
            "v_mid": v_mid,
            "goal1_u": g1_u,
            "goal1_v": g1_v,
            "goal2_u": g2_u,
            "goal2_v": g2_v,
        }
        if GOAL_RECT_GEOM is not None:
            params["half_u"] = GOAL_RECT_GEOM["half_u"]
            params["top_bottom"] = GOAL_RECT_GEOM["top_bottom"]
            params["top_top"] = GOAL_RECT_GEOM["top_top"]
            params["bottom_bottom"] = GOAL_RECT_GEOM["bottom_bottom"]
            params["bottom_top"] = GOAL_RECT_GEOM["bottom_top"]
        params_path = out_dir / "midline_and_goals.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  midline + goals -> {params_path}")

    reward_frame_by_trial: dict[str, int] = {}
    reward_times_path = args.reward_times if args.reward_times is not None else out_root / "reward_times.csv"
    reward_times_path = Path(reward_times_path).resolve()
    if reward_times_path.is_file():
        reward_frame_by_trial = _get_reward_frame_from_csv(trials, reward_times_path)
        if reward_frame_by_trial:
            print(f"  reward frame from CSV: {len(reward_frame_by_trial)}/{len(trials)} trials")
    if not reward_frame_by_trial and args.logs_dir is not None:
        logs_dir = Path(args.logs_dir).resolve()
        reward_frame_by_trial = _get_reward_frame_per_trial(trials, logs_dir, animal)
        if reward_frame_by_trial:
            print(f"  reward frame from logs: {len(reward_frame_by_trial)}/{len(trials)} trials")

    if reward_frame_by_trial:
        params_path = out_dir / "midline_and_goals.json"
        if params_path.is_file():
            with open(params_path) as f:
                params = json.load(f)
        elif params_tuple is not None:
            params = {
                "v_mid": params_tuple[0],
                "goal1_u": params_tuple[1],
                "goal1_v": params_tuple[2],
                "goal2_u": params_tuple[3],
                "goal2_v": params_tuple[4],
            }
        else:
            params = None
        if params is not None:
            paths_path = out_dir / "paths_trial_start_to_reward.png"
            _plot_paths_trial_start_to_reward(trials, reward_frame_by_trial, params, paths_path, u_min, u_max, v_min, v_max, animal=animal)
            print(f"  paths (trial start → reward) -> {paths_path}")
            count_path = out_dir / "paths_by_crossing_count.png"
            _plot_paths_by_crossing_count(trials, reward_frame_by_trial, params, count_path, u_min, u_max, v_min, v_max, animal=animal)
            print(f"  paths by crossing count (0, 1, 2, …) -> {count_path}")
            crossing_u_all = _collect_crossing_u_locations(trials, reward_frame_by_trial, params)
            if crossing_u_all:
                crossing_loc_path = out_dir / "crossing_locations_on_midline.png"
                _plot_crossing_locations_on_midline(crossing_u_all, params, u_min, u_max, crossing_loc_path, title_suffix="all trials", animal=animal)
                print(f"  crossing locations on midline -> {crossing_loc_path}")
            early_trials, mid_trials, late_trials = _split_phase_trials(trials)
            trial_types_path = args.trial_types if args.trial_types is not None else out_root / "trial_types.csv"
            trial_types_path = Path(trial_types_path).resolve()
            type_sets = _load_trial_type_sets(trial_types_path)
            early_vl, early_vr = [], []
            mid_vl, mid_vr = [], []
            late_vl, late_vr = [], []
            if type_sets is not None:
                vertical_left_ids, vertical_right_ids = type_sets
                early_vl = [(p, tid, sess) for p, tid, sess in early_trials if tid in vertical_left_ids]
                early_vr = [(p, tid, sess) for p, tid, sess in early_trials if tid in vertical_right_ids]
                mid_vl = [(p, tid, sess) for p, tid, sess in mid_trials if tid in vertical_left_ids]
                mid_vr = [(p, tid, sess) for p, tid, sess in mid_trials if tid in vertical_right_ids]
                late_vl = [(p, tid, sess) for p, tid, sess in late_trials if tid in vertical_left_ids]
                late_vr = [(p, tid, sess) for p, tid, sess in late_trials if tid in vertical_right_ids]
            crossing_u_early_all = _collect_crossing_u_locations(early_trials, reward_frame_by_trial, params)
            crossing_u_mid_all = _collect_crossing_u_locations(mid_trials, reward_frame_by_trial, params)
            crossing_u_late_all = _collect_crossing_u_locations(late_trials, reward_frame_by_trial, params)
            crossing_u_early_vl = _collect_crossing_u_locations(early_vl, reward_frame_by_trial, params) if early_vl else []
            crossing_u_early_vr = _collect_crossing_u_locations(early_vr, reward_frame_by_trial, params) if early_vr else []
            crossing_u_mid_vl = _collect_crossing_u_locations(mid_vl, reward_frame_by_trial, params) if mid_vl else []
            crossing_u_mid_vr = _collect_crossing_u_locations(mid_vr, reward_frame_by_trial, params) if mid_vr else []
            crossing_u_late_vl = _collect_crossing_u_locations(late_vl, reward_frame_by_trial, params) if late_vl else []
            crossing_u_late_vr = _collect_crossing_u_locations(late_vr, reward_frame_by_trial, params) if late_vr else []
            for phase_name, trials, vl_trials, vr_trials in [
                ("early", early_trials, early_vl, early_vr),
                ("mid", mid_trials, mid_vl, mid_vr),
                ("late", late_trials, late_vl, late_vr),
            ]:
                phase_dir = out_dir / phase_name
                phase_dir.mkdir(parents=True, exist_ok=True)
                if trials:
                    p_all = phase_dir / "paths_by_crossing_count.png"
                    _plot_paths_by_crossing_count(trials, reward_frame_by_trial, params, p_all, u_min, u_max, v_min, v_max, title_suffix=f"{phase_name} phase", animal=animal)
                    print(f"  {phase_name}/paths_by_crossing_count.png -> {p_all}")
                if vl_trials:
                    p_vl = phase_dir / "paths_by_crossing_count_vertical_left.png"
                    _plot_paths_by_crossing_count(vl_trials, reward_frame_by_trial, params, p_vl, u_min, u_max, v_min, v_max, title_suffix=f"{phase_name} phase, vertical left", animal=animal)
                    print(f"  {phase_name}/paths_by_crossing_count_vertical_left.png -> {p_vl}")
                if vr_trials:
                    p_vr = phase_dir / "paths_by_crossing_count_vertical_right.png"
                    _plot_paths_by_crossing_count(vr_trials, reward_frame_by_trial, params, p_vr, u_min, u_max, v_min, v_max, title_suffix=f"{phase_name} phase, vertical right", animal=animal)
                    print(f"  {phase_name}/paths_by_crossing_count_vertical_right.png -> {p_vr}")
            combined_crossing_path = out_dir / "crossing_locations_on_midline_combined.png"
            _plot_crossing_locations_combined(
                crossing_u_early_all, crossing_u_early_vl, crossing_u_early_vr,
                crossing_u_mid_all, crossing_u_mid_vl, crossing_u_mid_vr,
                crossing_u_late_all, crossing_u_late_vl, crossing_u_late_vr,
                params, u_min, u_max, combined_crossing_path,
                n_early=len(early_trials), n_mid=len(mid_trials), n_late=len(late_trials),
                animal=animal,
            )
            print(f"  crossing locations on midline (combined) -> {combined_crossing_path}")
            # Paths that approach one goal rectangle and then the other
            switches_path = out_dir / "paths_goal_region_switches.png"
            # Only consider late-phase trials for goal-region switches (clearer visualization).
            _plot_paths_goal_switches(late_trials, reward_frame_by_trial, params, switches_path, u_min, u_max, v_min, v_max)
            if switches_path.is_file():
                print(f"  paths that switch goal regions -> {switches_path}")
            # Exclusive goal-region paths (visit only goal1 region vs only goal2 region), full paths start→reward.
            for phase_name, phase_trials in [("early", early_trials), ("mid", mid_trials), ("late", late_trials)]:
                if not phase_trials:
                    continue
                goal_exclusive_path = out_dir / f"paths_goal_region_exclusive_{phase_name}.png"
                _plot_paths_by_goal_region_exclusive(phase_trials, reward_frame_by_trial, params, goal_exclusive_path, u_min, u_max, v_min, v_max)
                if goal_exclusive_path.is_file():
                    print(f"  paths by exclusive goal region ({phase_name} phase) -> {goal_exclusive_path}")
            # First goal visited: left vs right (by u), full path, for each phase.
            for phase_name, phase_trials in [("early", early_trials), ("mid", mid_trials), ("late", late_trials)]:
                if not phase_trials:
                    continue
                first_goal_path = out_dir / f"paths_first_goal_left_vs_right_{phase_name}.png"
                _plot_paths_by_first_goal(phase_trials, reward_frame_by_trial, params, first_goal_path, u_min, u_max, v_min, v_max, title_suffix=f"{phase_name} phase")
                if first_goal_path.is_file():
                    print(f"  paths first goal left vs right ({phase_name} phase) -> {first_goal_path}")
            # Histogram: number of goal regions visited (0, 1, 2) for early, mid, late.
            hist_path = out_dir / "goal_regions_visited_histogram.png"
            _plot_goal_regions_visited_histogram(early_trials, mid_trials, late_trials, reward_frame_by_trial, params, hist_path, animal=animal)
            print(f"  goal regions visited (0/1/2) histogram -> {hist_path}")
            # When and where do trials that visit both goals first cross?
            when_where_path = out_dir / "both_goals_when_and_where.png"
            _plot_both_goals_when_and_where(trials, reward_frame_by_trial, params, when_where_path, u_min, u_max, v_min, v_max)
            if when_where_path.is_file():
                print(f"  both goals: when and where -> {when_where_path}")
            # U–v heatmap of first visit to each goal rectangle, by phase (3×2: early/mid/late × goal1/goal2).
            first_visit_heatmap_path = out_dir / "first_goal_visit_heatmap_by_phase.png"
            _plot_first_goal_visit_heatmaps(trials, reward_frame_by_trial, params, first_visit_heatmap_path)
            if first_visit_heatmap_path.is_file():
                print(f"  first goal visit u–v heatmap (by phase) -> {first_visit_heatmap_path}")
            # First visit to each goal when the other was visited first (e.g. visit G1 first, then where in G2 is first entry).
            first_visit_when_other_first_path = out_dir / "first_visit_to_goal_when_other_first_heatmap.png"
            _plot_first_visit_to_goal_when_other_first_heatmaps(trials, reward_frame_by_trial, params, first_visit_when_other_first_path)
            if first_visit_when_other_first_path.is_file():
                print(f"  first visit to goal when other first (heatmap) -> {first_visit_when_other_first_path}")
        else:
            print("  skip paths plot: no midline/goals")
    elif not reward_frame_by_trial:
        if reward_times_path.is_file():
            print(f"  skip paths plot: no matching sessions in reward_times.csv for {animal} (check session dates match predictions)")
        else:
            print("  skip paths plot: no reward_times.csv and no --logs-dir (run cbot_climb_log/export_reward_times.py first)")


if __name__ == "__main__":
    main()
