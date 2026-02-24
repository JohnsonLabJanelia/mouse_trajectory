#!/usr/bin/env python3
"""
One composite figure per animal: 3×2 grid = phase (early, mid, late) × side (vertical left, vertical right).
Uses the same trajectory data and trial_types.csv as plot_phase_trajectories.py.
Saves trajectory_analysis/{animal}/phase_vs_side.png.
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

MAX_Z = 150.0
U_LOW_THRESHOLD = 1250.0
Z_CAP_WHEN_U_LOW = 50.0


def load_trajectory_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "z" not in df.columns or len(df) == 0:
        return df
    keep = (df["z"] >= 0) & (df["z"] <= MAX_Z)
    if "u" in df.columns:
        keep = keep & ((df["u"] >= U_LOW_THRESHOLD) | (df["z"] <= Z_CAP_WHEN_U_LOW))
    return df.loc[keep].copy()


def iter_sessions_and_trials(predictions_root: Path):
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
    by_animal: dict[str, list[tuple[Path, str]]] = {}
    for animal, _sf, trials in iter_sessions_and_trials(predictions_root):
        if animals_filter and animal.lower() not in animals_filter:
            continue
        if animal not in by_animal:
            by_animal[animal] = []
        for csv_path, trial_id in trials:
            by_animal[animal].append((csv_path, trial_id))
    return by_animal


def load_trial_type_sets(trial_types_path: Path) -> tuple[set[str], set[str]] | None:
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


def split_into_three(trial_list: list, animal: str = "") -> tuple[list, list, list]:
    n = len(trial_list)
    if n == 0:
        return [], [], []
    k = max(1, n // 3)
    early = trial_list[:k]
    mid = trial_list[k : 2 * k] if 2 * k <= n else []
    late = trial_list[2 * k :] if 2 * k < n else []
    return early, mid, late


def draw_trajectory_on_ax(
    ax,
    trials: list[tuple[Path, str]],
    a1: str,
    a2: str,
    norm: plt.Normalize,
    a1_min: float,
    a1_max: float,
    a2_min: float,
    a2_max: float,
    use_uv: bool,
    cmap,
    title: str,
) -> None:
    """Draw trajectory (u-v or x-y, z as color) on the given axes with shared limits and norm."""
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
            lc = LineCollection(segments, array=z_seg, cmap=cmap, norm=norm, linewidth=1.0, alpha=0.9)
            ax.add_collection(lc)
    ax.set_xlim(a1_min, a1_max)
    ax.set_ylim(a2_min, a2_max)
    if use_uv:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description="One 3×2 (phase × side) figure per animal.")
    parser.add_argument("--predictions-root", type=Path, default=Path("/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("trajectory_analysis"))
    parser.add_argument("--use-world-xy", action="store_true")
    parser.add_argument("--animals", nargs="*", default=None)
    parser.add_argument("--trial-types", type=Path, default=None)
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    predictions_root = Path(args.predictions_root).resolve()
    if not predictions_root.is_dir():
        raise SystemExit(f"Not a directory: {predictions_root}")

    trial_types_path = Path(args.trial_types) if args.trial_types is not None else out_root / "trial_types.csv"
    type_sets = load_trial_type_sets(trial_types_path)
    if type_sets is None:
        raise SystemExit("trial_types.csv required for phase vs side. Run cbot_climb_log/export_trial_types_for_trajectories.py -o trajectory_analysis/trial_types.csv")

    vertical_left_ids, vertical_right_ids = type_sets
    animals_filter = set(a.lower() for a in args.animals) if args.animals else None
    by_animal = trials_per_animal_chronological(predictions_root, animals_filter)

    for animal, all_trials in by_animal.items():
        if len(all_trials) < 1:
            continue
        early_trials, mid_trials, late_trials = split_into_three(all_trials, animal)
        phases = [("early", early_trials), ("mid", mid_trials), ("late", late_trials)]

        sample_csv = all_trials[0][0]
        df0 = load_trajectory_csv(sample_csv)
        use_uv = not args.use_world_xy and "u" in df0.columns and "v" in df0.columns
        a1, a2 = ("u", "v") if use_uv else ("x", "y")
        l1, l2 = ("u (px)", "v (px)") if use_uv else ("x", "y")

        # 3×2 grid: row = phase (early, mid, late), col = side (vertical_left, vertical_right)
        grid: list[list[list[tuple[Path, str]]]] = []
        for phase_name, trials in phases:
            vl = [(p, t) for p, t in trials if t in vertical_left_ids]
            vr = [(p, t) for p, t in trials if t in vertical_right_ids]
            grid.append([vl, vr])

        # Global limits and z norm across all 6 panels
        all_a1, all_a2, all_z = [], [], []
        for row in grid:
            for trials in row:
                for csv_path, _ in trials:
                    df = load_trajectory_csv(csv_path)
                    if len(df) < 1 or a1 not in df.columns or a2 not in df.columns:
                        continue
                    all_a1.extend(df[a1].tolist())
                    all_a2.extend(df[a2].tolist())
                    all_z.extend(df["z"].tolist())
        if not all_a1 or not all_z:
            continue
        a1_min = min(all_a1) - 0.05 * ((max(all_a1) - min(all_a1)) or 1)
        a1_max = max(all_a1) + 0.05 * ((max(all_a1) - min(all_a1)) or 1)
        a2_min = min(all_a2) - 0.05 * ((max(all_a2) - min(all_a2)) or 1)
        a2_max = max(all_a2) + 0.05 * ((max(all_a2) - min(all_a2)) or 1)
        z_min, z_max = min(all_z), max(all_z)
        norm = plt.Normalize(vmin=z_min, vmax=z_max)
        cmap = plt.colormaps.get_cmap("viridis")

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        phase_names = ["early", "mid", "late"]
        side_names = ["vertical left", "vertical right"]
        for i, phase_name in enumerate(phase_names):
            for j, side_name in enumerate(side_names):
                ax = axes[i, j]
                trials = grid[i][j]
                title = f"{phase_name} — {side_name} (n={len(trials)})"
                draw_trajectory_on_ax(ax, trials, a1, a2, norm, a1_min, a1_max, a2_min, a2_max, use_uv, cmap, title)
                if j == 0:
                    ax.set_ylabel(l2)
                if i == 0:
                    ax.set_xlabel("")
                if i == 2:
                    ax.set_xlabel(l1)

        fig.suptitle(f"{animal} — phase × side", fontsize=12)
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="z (elevation)")
        plt.tight_layout(rect=[0, 0, 0.88, 0.96])
        out_path = out_root / animal / "phase_vs_side.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {animal} -> {out_path}")

        # For rory: save zoomed late vertical left panel to flow_field/
        if animal.lower() == "rory":
            late_vl = grid[2][0]
            if late_vl:
                zoom_a1, zoom_a2 = [], []
                for csv_path, _ in late_vl:
                    df = load_trajectory_csv(csv_path)
                    if len(df) < 1 or a1 not in df.columns or a2 not in df.columns:
                        continue
                    zoom_a1.extend(df[a1].tolist())
                    zoom_a2.extend(df[a2].tolist())
                if zoom_a1 and zoom_a2:
                    zoom_a1_arr = np.array(zoom_a1, dtype=float)
                    zoom_a2_arr = np.array(zoom_a2, dtype=float)
                    p1_low, p1_hi = np.percentile(zoom_a1_arr, [5, 95])
                    p2_low, p2_hi = np.percentile(zoom_a2_arr, [5, 95])
                    margin = 0.08
                    d1, d2 = (p1_hi - p1_low) or 1, (p2_hi - p2_low) or 1
                    z1_min, z1_max = p1_low - margin * d1, p1_hi + margin * d1
                    z2_min, z2_max = p2_low - margin * d2, p2_hi + margin * d2
                    fig_zoom, ax_zoom = plt.subplots(figsize=(8, 8))
                    draw_trajectory_on_ax(ax_zoom, late_vl, a1, a2, norm, z1_min, z1_max, z2_min, z2_max, use_uv, cmap, f"rory — late, vertical left (n={len(late_vl)}) — zoomed")
                    ax_zoom.set_xlabel(l1)
                    ax_zoom.set_ylabel(l2)
                    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_zoom, label="z (elevation)")
                    plt.tight_layout()
                    zoom_dir = out_root / animal / "flow_field"
                    zoom_dir.mkdir(parents=True, exist_ok=True)
                    zoom_path = zoom_dir / "late_vertical_left_zoomed.png"
                    fig_zoom.savefig(zoom_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"  rory -> {zoom_path} (zoomed)")


if __name__ == "__main__":
    main()
