#!/usr/bin/env python3
"""
Plot trajectory_filtered.csv per session, one output folder per animal and session.

Uses (u, v) — the same 2D camera/image coordinates used to draw the trajectory on the
video frame in plot_trajectory_on_frame.py — so the orientation matches the arena view.
Z is shown as color.

Output layout:
  trajectory_analysis/
    rory/
      sessions/
        rory_2025_12_23_16_57_09/
          trajectory_xy_z.png      # all trials
          vertical_left/
            trajectory_xy_z.png    # trials with vertical on left (from trial_types.csv)
          vertical_right/
            trajectory_xy_z.png    # trials with vertical on right
        ...
    wilfred/
      sessions/
        ...

Trial type (vertical left vs right) comes from cbot_climb_log/export_trial_types_for_trajectories.py
which writes trajectory_analysis/trial_types.csv from robot_manager logs. Use --trial-types to override path.

Use --use-world-xy to plot 3D world (x, y) instead of camera (u, v).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

SESSION_PATTERN = re.compile(r"^([a-z]+)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")
TRIAL_PATTERN = re.compile(r"^Predictions_3D_trial_\d+_\d+-\d+$")

# Trajectory filters (see docs/TRAJECTORY_FILTERS.md)
MAX_Z = 150.0  # drop points with z > 150
U_LOW_THRESHOLD = 1250.0  # px; in low-u region cap z
Z_CAP_WHEN_U_LOW = 50.0  # when u < U_LOW_THRESHOLD, drop points with z > this


def load_trajectory_csv(csv_path: Path) -> pd.DataFrame:
    """Load trajectory CSV; apply elevation and region filters (see docs/TRAJECTORY_FILTERS.md)."""
    df = pd.read_csv(csv_path)
    if "z" not in df.columns or len(df) == 0:
        return df
    keep = (df["z"] >= 0) & (df["z"] <= MAX_Z)
    if "u" in df.columns:
        keep = keep & ((df["u"] >= U_LOW_THRESHOLD) | (df["z"] <= Z_CAP_WHEN_U_LOW))
    df = df.loc[keep].copy()
    return df


def load_trial_type_sets(trial_types_path: Path) -> tuple[set[str], set[str]] | None:
    """
    Load trial_types.csv (from export_trial_types_for_trajectories.py). Return (vertical_left_ids, vertical_right_ids)
    or None if file missing/invalid. vertical_left = left_angle_deg==360 and right!=360; vertical_right = right==360 and left!=360.
    """
    trial_types_path = Path(trial_types_path)
    if not trial_types_path.is_file():
        return None
    try:
        tt = pd.read_csv(trial_types_path)
    except Exception:
        return None
    if "trial_id" not in tt.columns or "left_angle_deg" not in tt.columns or "right_angle_deg" not in tt.columns:
        return None
    vl = set()
    vr = set()
    for _, row in tt.iterrows():
        lid = row.get("left_angle_deg")
        rid = row.get("right_angle_deg")
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


def _write_trajectory_plot(
    trials: list[tuple[Path, str]],
    out_path: Path,
    title: str,
    use_uv: bool,
    a1: str,
    a2: str,
    l1: str,
    l2: str,
) -> None:
    """Write one trajectory_xy_z.png from the given list of (csv_path, trial_id)."""
    if not trials:
        return
    all_a1, all_a2, all_z = [], [], []
    for csv_path, _ in trials:
        df = load_trajectory_csv(csv_path)
        if len(df) < 1 or a1 not in df.columns or a2 not in df.columns:
            continue
        all_a1.extend(df[a1].tolist())
        all_a2.extend(df[a2].tolist())
        all_z.extend(df["z"].tolist())
    if not all_a1 or not all_z:
        return
    a1_arr = np.array(all_a1, dtype=float)
    a2_arr = np.array(all_a2, dtype=float)
    z_arr = np.array(all_z, dtype=float)
    a1_min, a1_max = a1_arr.min(), a1_arr.max()
    a2_min, a2_max = a2_arr.min(), a2_arr.max()
    margin = 0.05
    d1, d2 = (a1_max - a1_min) or 1, (a2_max - a2_min) or 1
    a1_min -= margin * d1
    a1_max += margin * d1
    a2_min -= margin * d2
    a2_max += margin * d2
    z_min, z_max = z_arr.min(), z_arr.max()
    cmap = plt.colormaps.get_cmap("viridis")
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    fig, ax = plt.subplots(figsize=(10, 10))
    for csv_path, _ in trials:
        df = load_trajectory_csv(csv_path)
        if len(df) < 2 or a1 not in df.columns or a2 not in df.columns:
            continue
        for seg_id in sorted(df["segment_id"].unique()):
            seg = df[df["segment_id"] == seg_id].sort_values("frame_number")
            if len(seg) < 2:
                continue
            p1 = seg[a1].values.astype(float)
            p2 = seg[a2].values.astype(float)
            z = seg["z"].values.astype(float)
            segments = np.stack([
                np.column_stack([p1[:-1], p2[:-1]]),
                np.column_stack([p1[1:], p2[1:]])
            ], axis=1)
            z_seg = (z[:-1] + z[1:]) / 2
            lc = LineCollection(segments, array=z_seg, cmap=cmap, norm=norm, linewidth=1.2, alpha=0.9)
            ax.add_collection(lc)
    ax.set_xlim(a1_min, a1_max)
    ax.set_ylim(a2_min, a2_max)
    ax.set_xlabel(l1)
    ax.set_ylabel(l2)
    if use_uv:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="z (elevation)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def iter_sessions_and_trials(predictions_root: Path):
    """Yield (animal, session_folder, list of (csv_path, trial_id)) per session."""
    predictions_root = Path(predictions_root)
    for session_dir in sorted(predictions_root.iterdir()):
        if not session_dir.is_dir():
            continue
        m = SESSION_PATTERN.match(session_dir.name)
        if not m:
            continue
        animal, _ = m.group(1), m.group(2)
        trials = []
        for trial_dir in sorted(session_dir.iterdir()):
            if not trial_dir.is_dir() or not TRIAL_PATTERN.match(trial_dir.name):
                continue
            csv_path = trial_dir / "trajectory_filtered.csv"
            if not csv_path.exists():
                continue
            trials.append((csv_path, trial_dir.name))
        if trials:
            yield (animal, session_dir.name, trials)


def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectory x-y (z as color) per session under trajectory_analysis/animal/session/"
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D"),
        help="Predictions root with session folders (animal_YYYY_MM_DD_HH_MM_SS)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("trajectory_analysis"),
        help="Output root (default: trajectory_analysis)",
    )
    parser.add_argument(
        "--use-world-xy",
        action="store_true",
        help="Plot 3D world (x, y) instead of camera (u, v). Default is (u, v) to match frame/arena view.",
    )
    parser.add_argument(
        "--animals",
        nargs="*",
        default=None,
        help="Only these animals (default: all)",
    )
    parser.add_argument(
        "--trial-types",
        type=Path,
        default=None,
        help="Path to trial_types.csv (default: <output-dir>/trial_types.csv). From export_trial_types_for_trajectories.py.",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    predictions_root = Path(args.predictions_root).resolve()
    if not predictions_root.is_dir():
        raise SystemExit(f"Not a directory: {predictions_root}")

    trial_types_path = Path(args.trial_types) if args.trial_types is not None else out_root / "trial_types.csv"
    type_sets = load_trial_type_sets(trial_types_path)
    if type_sets is None:
        print("No trial_types.csv found; skipping vertical_left / vertical_right breakdown. Run cbot_climb_log/export_trial_types_for_trajectories.py to generate.")
    else:
        vertical_left_ids, vertical_right_ids = type_sets
        print(f"Trial types loaded: {len(vertical_left_ids)} vertical-left, {len(vertical_right_ids)} vertical-right trials")

    animals_filter = set(a.lower() for a in args.animals) if args.animals else None
    n_plots = 0
    for animal, session_folder, trials in iter_sessions_and_trials(predictions_root):
        if animals_filter and animal.lower() not in animals_filter:
            continue
        session_out = out_root / animal / "sessions" / session_folder
        session_out.mkdir(parents=True, exist_ok=True)

        use_uv = not args.use_world_xy and "u" in load_trajectory_csv(trials[0][0]).columns and "v" in load_trajectory_csv(trials[0][0]).columns
        if use_uv:
            a1, a2, l1, l2 = "u", "v", "u (px)", "v (px)"
        else:
            a1, a2, l1, l2 = "x", "y", "x", "y"

        _write_trajectory_plot(
            trials,
            session_out / "trajectory_xy_z.png",
            f"{session_folder} — {len(trials)} trials, trajectory (z = color)",
            use_uv, a1, a2, l1, l2,
        )
        n_plots += 1
        print(f"  {animal}/sessions/{session_folder}: {len(trials)} trials -> {session_out / 'trajectory_xy_z.png'}")

        if type_sets is not None:
            vertical_left_ids, vertical_right_ids = type_sets
            vl_trials = [(p, tid) for p, tid in trials if tid in vertical_left_ids]
            vr_trials = [(p, tid) for p, tid in trials if tid in vertical_right_ids]
            if vl_trials:
                _write_trajectory_plot(
                    vl_trials,
                    session_out / "vertical_left" / "trajectory_xy_z.png",
                    f"{session_folder} — vertical left ({len(vl_trials)} trials)",
                    use_uv, a1, a2, l1, l2,
                )
                print(f"    -> vertical_left: {len(vl_trials)} trials")
            if vr_trials:
                _write_trajectory_plot(
                    vr_trials,
                    session_out / "vertical_right" / "trajectory_xy_z.png",
                    f"{session_folder} — vertical right ({len(vr_trials)} trials)",
                    use_uv, a1, a2, l1, l2,
                )
                print(f"    -> vertical_right: {len(vr_trials)} trials")

    print(f"Wrote {n_plots} session plots under {out_root}")


if __name__ == "__main__":
    main()
