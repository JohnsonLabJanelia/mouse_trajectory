#!/usr/bin/env python3
"""
Create early / mid / late phase subfolders under each animal in trajectory_analysis/
and write one aggregated trajectory plot per phase.

Trials are ordered chronologically (by session folder name, then trial id) and split
evenly into three phases (early = first third, mid = second third, late = last third).
Uses (u, v) from trajectory_filtered.csv with z as color, same as plot_session_trajectories.py.

Output layout:
  trajectory_analysis/
    rory/
      phases/
        early/
          trajectory_xy_z.png      # all trials in early phase
          vertical_left/
          vertical_right/
        mid/
        late/
    wilfred/
      phases/
        ...

Trial type (vertical left/right) from trial_types.csv (export_trial_types_for_trajectories.py). Use --trial-types to set path.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

SESSION_PATTERN = re.compile(r"^([a-z]+)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")
TRIAL_PATTERN = re.compile(r"^Predictions_3D_trial_(\d+)_(\d+)-\d+$")

# Trajectory filters (see docs/TRAJECTORY_FILTERS.md)
MAX_Z = 150.0
U_LOW_THRESHOLD = 1250.0
Z_CAP_WHEN_U_LOW = 50.0


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


def trials_per_animal_chronological(predictions_root: Path, animals_filter: set | None):
    """Return dict: animal -> list of (csv_path, trial_id) in chronological order."""
    by_animal: dict[str, list[tuple[Path, str]]] = {}
    for animal, session_folder, trials in iter_sessions_and_trials(predictions_root):
        if animals_filter and animal.lower() not in animals_filter:
            continue
        if animal not in by_animal:
            by_animal[animal] = []
        for csv_path, trial_id in trials:
            by_animal[animal].append((csv_path, trial_id))
    return by_animal


def load_trial_type_sets(trial_types_path: Path) -> tuple[set[str], set[str]] | None:
    """Load trial_types.csv; return (vertical_left_ids, vertical_right_ids) or None."""
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


def split_into_three(trial_list: list, animal: str = "") -> tuple[list, list, list]:
    """Split list evenly into early, mid, late (each gets roughly n/3 trials)."""
    n = len(trial_list)
    if n == 0:
        return [], [], []
    k = max(1, n // 3)
    early = trial_list[:k]
    mid = trial_list[k : 2 * k] if 2 * k <= n else []
    late = trial_list[2 * k :] if 2 * k < n else []
    return early, mid, late


def main():
    parser = argparse.ArgumentParser(
        description="Create early/mid/late subfolders per animal and aggregate trajectory plots."
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
        help="Plot 3D world (x, y) instead of camera (u, v).",
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
        help="Path to trial_types.csv (default: <output-dir>/trial_types.csv).",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    predictions_root = Path(args.predictions_root).resolve()
    if not predictions_root.is_dir():
        raise SystemExit(f"Not a directory: {predictions_root}")

    trial_types_path = Path(args.trial_types) if args.trial_types is not None else out_root / "trial_types.csv"
    type_sets = load_trial_type_sets(trial_types_path)
    if type_sets is None:
        print("No trial_types.csv found; skipping vertical_left / vertical_right in phases.")
    else:
        print(f"Trial types loaded: {len(type_sets[0])} vertical-left, {len(type_sets[1])} vertical-right trials")

    animals_filter = set(a.lower() for a in args.animals) if args.animals else None
    by_animal = trials_per_animal_chronological(predictions_root, animals_filter)

    for animal, all_trials in by_animal.items():
        if len(all_trials) < 1:
            continue
        early_trials, mid_trials, late_trials = split_into_three(all_trials, animal)
        phases = [
            ("early", early_trials),
            ("mid", mid_trials),
            ("late", late_trials),
        ]

        # Detect u/v from first available CSV
        sample_csv = all_trials[0][0]
        df0 = load_trajectory_csv(sample_csv)
        use_uv = not args.use_world_xy and "u" in df0.columns and "v" in df0.columns
        if use_uv:
            a1, a2, l1, l2 = "u", "v", "u (px)", "v (px)"
        else:
            a1, a2, l1, l2 = "x", "y", "x", "y"

        for phase_name, trials in phases:
            if not trials:
                continue
            phase_dir = out_root / animal / "phases" / phase_name
            phase_dir.mkdir(parents=True, exist_ok=True)

            _write_trajectory_plot(
                trials,
                phase_dir / "trajectory_xy_z.png",
                f"{animal} — {phase_name} phase ({len(trials)} trials), trajectory (z = color)",
                use_uv, a1, a2, l1, l2,
            )
            print(f"  {animal}/phases/{phase_name}: {len(trials)} trials -> {phase_dir / 'trajectory_xy_z.png'}")

            if type_sets is not None:
                vertical_left_ids, vertical_right_ids = type_sets
                vl_trials = [(p, tid) for p, tid in trials if tid in vertical_left_ids]
                vr_trials = [(p, tid) for p, tid in trials if tid in vertical_right_ids]
                if vl_trials:
                    _write_trajectory_plot(
                        vl_trials,
                        phase_dir / "vertical_left" / "trajectory_xy_z.png",
                        f"{animal} — {phase_name} phase, vertical left ({len(vl_trials)} trials)",
                        use_uv, a1, a2, l1, l2,
                    )
                    print(f"    -> vertical_left: {len(vl_trials)} trials")
                if vr_trials:
                    _write_trajectory_plot(
                        vr_trials,
                        phase_dir / "vertical_right" / "trajectory_xy_z.png",
                        f"{animal} — {phase_name} phase, vertical right ({len(vr_trials)} trials)",
                        use_uv, a1, a2, l1, l2,
                    )
                    print(f"    -> vertical_right: {len(vr_trials)} trials")

    print(f"Phase aggregates written under {out_root}")


if __name__ == "__main__":
    main()
