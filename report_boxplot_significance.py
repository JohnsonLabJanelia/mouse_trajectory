#!/usr/bin/env python3
"""
Collect data used in all boxplot analyses, run statistical tests (Kruskal-Wallis for 3 groups,
Mann-Whitney U for 2 groups; post-hoc with Bonferroni), and write SIGNIFICANCE_REPORT.md to
trajectory_analysis/<animal>/.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import analyze_trajectories as at
from plot_head_direction import (
    get_animal_trials,
    _split_phase_trials,
    _get_reward_frame_by_trial,
    _resolve_calib_path,
    load_calib,
    load_head_direction_per_trial,
    load_body_direction_per_trial,
    _load_trial_start_to_reward_with_head,
    _load_trial_start_to_reward_with_body,
    _load_trial_start_to_reward_with_head_and_body,
    _parse_trial_frames,
    _crossing_events_with_frame,
    _first_goal_entries,
    _build_body_vs_head_dataframe,
    _point_goal_region_local,
)

def _bout_region_name(u: float, v: float, params: dict) -> str:
    lab = _point_goal_region_local(u, v, params)
    if lab == 1:
        return "Goal 1"
    if lab == 2:
        return "Goal 2"
    v_mid = params.get("v_mid")
    if v_mid is not None and v < v_mid:
        return "Above midline"
    return "Below midline"

PHASE_NAMES = ["Early", "Mid", "Late"]
ALPHA = 0.05


def _kruskal_posthoc(groups: list[np.ndarray], names: list[str]) -> list[tuple[str, str, float, str]]:
    """Pairwise Mann-Whitney U with Bonferroni. groups = [early, mid, late]. Returns list of (g1, g2, p, sig)."""
    n_pairs = 3  # Early-Mid, Early-Late, Mid-Late
    bonferroni_alpha = ALPHA / n_pairs
    results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if len(groups[i]) < 2 or len(groups[j]) < 2:
                results.append((names[i], names[j], float("nan"), "—"))
                continue
            stat, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            sig = "Yes" if p < bonferroni_alpha else "No"
            results.append((names[i], names[j], float(p), sig))
    return results


def run_tests_and_report(
    trials: list,
    early: list,
    mid: list,
    late: list,
    reward_frame_by_trial: dict,
    params: dict,
    calib_root: Path | None,
    camera: str,
    out_dir: Path,
    animal: str,
) -> None:
    trial_to_phase = {}
    for t in early:
        trial_to_phase[t[1]] = 0
    for t in mid:
        trial_to_phase[t[1]] = 1
    for t in late:
        trial_to_phase[t[1]] = 2
    v_mid = params.get("v_mid") if params else None
    g1_u = params.get("goal1_u") if params else None

    lines: list[str] = [
        "# Significance report: boxplot analyses",
        "",
        f"**Animal:** {animal}  ",
        f"**Trials:** {len(trials)} (Early: {len(early)}, Mid: {len(mid)}, Late: {len(late)})  ",
        "",
        "**Tests:** Kruskal-Wallis for 3 groups (Early / Mid / Late); Mann-Whitney U for 2 groups.  ",
        "Post-hoc after significant Kruskal-Wallis: pairwise Mann-Whitney with Bonferroni (α = 0.05/3).  ",
        f"Significance: α = {ALPHA}.  ",
        "",
        "---",
        "",
    ]
    summary_sig: list[str] = []

    # ----- Head direction -----
    crossing_head: dict[int, list[float]] = {0: [], 1: [], 2: []}
    toward1_head: list[float] = []
    toward2_head: list[float] = []
    goal1_head: dict[int, list[float]] = {0: [], 1: [], 2: []}
    goal2_head: dict[int, list[float]] = {0: [], 1: [], 2: []}
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
        for (frame, u, v, toward) in _crossing_events_with_frame(df, params):
            row = df[df["frame_number"] == frame]
            if len(row) > 0:
                crossing_head[phase_id].append(float(row["head_angle_deg"].iloc[0]))
                if toward == 1:
                    toward1_head.append(float(row["head_angle_deg"].iloc[0]))
                else:
                    toward2_head.append(float(row["head_angle_deg"].iloc[0]))
        f1, u1, v1, f2, u2, v2 = _first_goal_entries(df, params)
        if f1 is not None:
            r = df[df["frame_number"] == f1]
            if len(r) > 0:
                goal1_head[phase_id].append(float(r["head_angle_deg"].iloc[0]))
        if f2 is not None:
            r = df[df["frame_number"] == f2]
            if len(r) > 0:
                goal2_head[phase_id].append(float(r["head_angle_deg"].iloc[0]))

    lines.append("## 1. Head direction")
    lines.append("")
    if any(crossing_head[i] for i in range(3)):
        g = [crossing_head[i] for i in range(3)]
        try:
            stat, p = stats.kruskal(*g)
            lines.append(f"### 1.1 Head at midline crossing (by phase)")
            lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
            lines.append(f"- **Significant difference across phases:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                for a, b, pval, sig in _kruskal_posthoc(g, PHASE_NAMES):
                    lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")
    if toward1_head and toward2_head:
        try:
            stat, p = stats.mannwhitneyu(toward1_head, toward2_head, alternative="two-sided")
            lines.append(f"### 1.2 Head at midline crossing (toward goal 1 vs 2)")
            lines.append(f"- Mann-Whitney U, p = {p:.4f}  ")
            lines.append(f"- **Significant difference:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                summary_sig.append(f"- **1.2** Head at midline: toward goal 1 vs 2 differ (p = {p:.4f}).  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")
    for goal_name, gdict in [("Goal 1", goal1_head), ("Goal 2", goal2_head)]:
        g = [gdict[i] for i in range(3)]
        if not any(g):
            continue
        try:
            stat, p = stats.kruskal(*g)
            lines.append(f"### 1.3 Head when entering {goal_name} (by phase)")
            lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
            lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                for a, b, pval, sig in _kruskal_posthoc(g, PHASE_NAMES):
                    lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")

    # ----- Body direction -----
    crossing_body: dict[int, list[float]] = {0: [], 1: [], 2: []}
    toward1_body: list[float] = []
    toward2_body: list[float] = []
    goal1_body: dict[int, list[float]] = {0: [], 1: [], 2: []}
    goal2_body: dict[int, list[float]] = {0: [], 1: [], 2: []}
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
        for (frame, u, v, toward) in _crossing_events_with_frame(df, params):
            row = df[df["frame_number"] == frame]
            if len(row) > 0:
                crossing_body[phase_id].append(float(row["body_angle_deg"].iloc[0]))
                if toward == 1:
                    toward1_body.append(float(row["body_angle_deg"].iloc[0]))
                else:
                    toward2_body.append(float(row["body_angle_deg"].iloc[0]))
        f1, u1, v1, f2, u2, v2 = _first_goal_entries(df, params)
        if f1 is not None:
            r = df[df["frame_number"] == f1]
            if len(r) > 0:
                goal1_body[phase_id].append(float(r["body_angle_deg"].iloc[0]))
        if f2 is not None:
            r = df[df["frame_number"] == f2]
            if len(r) > 0:
                goal2_body[phase_id].append(float(r["body_angle_deg"].iloc[0]))

    lines.append("## 2. Body direction")
    lines.append("")
    if any(crossing_body[i] for i in range(3)):
        g = [crossing_body[i] for i in range(3)]
        try:
            stat, p = stats.kruskal(*g)
            lines.append(f"### 2.1 Body at midline crossing (by phase)")
            lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
            lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                for a, b, pval, sig in _kruskal_posthoc(g, PHASE_NAMES):
                    lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")
    if toward1_body and toward2_body:
        try:
            stat, p = stats.mannwhitneyu(toward1_body, toward2_body, alternative="two-sided")
            lines.append(f"### 2.2 Body at midline crossing (toward goal 1 vs 2)")
            lines.append(f"- Mann-Whitney U, p = {p:.4f}  ")
            lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                summary_sig.append(f"- **2.2** Body at midline: toward goal 1 vs 2 differ (p = {p:.4f}).  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")
    for goal_name, gdict in [("Goal 1", goal1_body), ("Goal 2", goal2_body)]:
        g = [gdict[i] for i in range(3)]
        if not any(g):
            continue
        try:
            stat, p = stats.kruskal(*g)
            lines.append(f"### 2.3 Body when entering {goal_name} (by phase)")
            lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
            lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                summary_sig.append(f"- **2.3** Body when entering Goal 2: differs by phase (p = {p:.4f}).  ")
                for a, b, pval, sig in _kruskal_posthoc(g, PHASE_NAMES):
                    lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")

    # ----- Body vs head (need full df and bout counts) -----
    df_bv = _build_body_vs_head_dataframe(trials, early, mid, late, reward_frame_by_trial, params, calib_root, camera, fps=180.0)
    if len(df_bv) > 0:
        lines.append("## 3. Body vs head")
        lines.append("")
        by_phase_abs = [df_bv[df_bv["phase_id"] == i]["abs_head_body_diff"].values for i in range(3)]
        try:
            stat, p = stats.kruskal(*by_phase_abs)
            lines.append(f"### 3.1 |Head − body| by phase (decoupling)")
            lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
            lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                summary_sig.append(f"- **3.1** |Head − body| by phase: differs across Early/Mid/Late (p = {p:.4f}).  ")
                for a, b, pval, sig in _kruskal_posthoc(by_phase_abs, PHASE_NAMES):
                    lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")

        regions_order = ["Above midline", "Below midline", "Goal 1", "Goal 2"]
        by_region = [df_bv[df_bv["region"] == r]["abs_head_body_diff"].values for r in regions_order]
        by_region = [x for x in by_region if len(x) >= 2]
        if len(by_region) >= 2:
            try:
                stat, p = stats.kruskal(*by_region)
                lines.append(f"### 3.2 |Head − body| by region")
                lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
                lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
                if p < ALPHA:
                    summary_sig.append(f"- **3.2** |Head − body| by region: differs across regions (p = {p:.4f}).  ")
                lines.append("")
            except Exception as e:
                lines.append(f"- Error: {e}  ")
                lines.append("")

        goal_entry_align: list[tuple[int, int, float]] = []
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
            for goal_idx in [1, 2]:
                sub = align_df[align_df["goal_idx"] == goal_idx]
                g = [sub[sub["phase_id"] == i]["abs_head_body_deg"].values for i in range(3)]
                if not all(len(x) >= 2 for x in g):
                    continue
                try:
                    stat, p = stats.kruskal(*g)
                    lines.append(f"### 3.3 Head–body alignment at first entry to goal {goal_idx} (by phase)")
                    lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
                    lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
                    if p < ALPHA:
                        summary_sig.append(f"- **3.3** Head–body alignment at goal {goal_idx} entry: differs by phase (p = {p:.4f}).  ")
                        for a, b, pval, sig in _kruskal_posthoc(g, PHASE_NAMES):
                            lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
                    lines.append("")
                except Exception as e:
                    lines.append(f"- Error: {e}  ")
                    lines.append("")

    # ----- Scanning bout count by phase and by region -----
    body_still_deg, head_scanning_deg = 15.0, 25.0
    regions_order = ["Above midline", "Below midline", "Goal 1", "Goal 2"]
    bout_count_per_trial: list[tuple[str, int, int]] = []
    bout_count_per_trial_region: list[dict] = []
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
        i, count_bouts = 0, 0
        region_counts: dict[str, int] = {r: 0 for r in regions_order}
        while i < len(tdf):
            if not is_scan[i]:
                i += 1
                continue
            j = i
            while j < len(tdf) and is_scan[j]:
                j += 1
            reg = _bout_region_name(float(u_vals[i]), float(v_vals[i]), params)
            region_counts[reg] = region_counts.get(reg, 0) + 1
            count_bouts += 1
            i = j
        bout_count_per_trial.append((trial_id, phase_id, count_bouts))
        for r in regions_order:
            bout_count_per_trial_region.append({"trial_id": trial_id, "phase_id": phase_id, "region": r, "bout_count": region_counts[r]})
    if bout_count_per_trial:
        bc_df = pd.DataFrame(bout_count_per_trial, columns=["trial_id", "phase_id", "bout_count"])
        by_phase_bouts = [bc_df[bc_df["phase_id"] == i]["bout_count"].values for i in range(3)]
        try:
            stat, p = stats.kruskal(*by_phase_bouts)
            lines.append("### 3.4 Scanning bout count per trial (by phase)")
            lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
            lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
            if p < ALPHA:
                summary_sig.append(f"- **3.4** Scanning bout count per trial: differs by phase (p = {p:.4f}).  ")
                for a, b, pval, sig in _kruskal_posthoc(by_phase_bouts, PHASE_NAMES):
                    lines.append(f"  - {a} vs {b}: p = {pval:.4f} ({sig})  ")
            lines.append("")
        except Exception as e:
            lines.append(f"- Error: {e}  ")
            lines.append("")

    if bout_count_per_trial_region:
        bc_region_df = pd.DataFrame(bout_count_per_trial_region)
        by_region_only = [bc_region_df[bc_region_df["region"] == r]["bout_count"].values for r in regions_order]
        by_region_only = [x for x in by_region_only if len(x) >= 2]
        if len(by_region_only) >= 2:
            try:
                stat, p = stats.kruskal(*by_region_only)
                lines.append("### 3.5 Scanning bout count per trial by region only (all phases)")
                lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
                lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
                if p < ALPHA:
                    summary_sig.append(f"- **3.5** Scanning bout count by region only: differs across regions (p = {p:.4f}).  ")
                lines.append("")
            except Exception as e:
                lines.append(f"- Error: {e}  ")
                lines.append("")
        for phase_id in range(3):
            sub = bc_region_df[bc_region_df["phase_id"] == phase_id]
            by_region = [sub[sub["region"] == r]["bout_count"].values for r in regions_order]
            by_region = [x for x in by_region if len(x) >= 2]
            if len(by_region) < 2:
                continue
            try:
                stat, p = stats.kruskal(*by_region)
                lines.append(f"### 3.6 Scanning bout count by phase (comparing regions): {PHASE_NAMES[phase_id]}")
                lines.append(f"- Kruskal-Wallis H = {stat:.4f}, p = {p:.4f}  ")
                lines.append(f"- **Significant:** {'Yes' if p < ALPHA else 'No'}  ")
                if p < ALPHA:
                    summary_sig.append(f"- **3.6** Scanning bout count in {PHASE_NAMES[phase_id]}: differs by region (p = {p:.4f}).  ")
                lines.append("")
            except Exception as e:
                lines.append(f"- Error: {e}  ")
                lines.append("")

    # Insert summary after first "---"
    idx = next(i for i, s in enumerate(lines) if s.strip() == "---")
    summary_block = ["## Summary (significant findings)", ""] + (summary_sig if summary_sig else ["No significant differences at α = 0.05."]) + ["", "---", ""]
    lines = lines[: idx + 1] + [""] + summary_block + lines[idx + 1 :]

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by report_boxplot_significance.py. Non-parametric tests; no correction for multiple comparisons across sections.*")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "SIGNIFICANCE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Significance report -> {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run significance tests on boxplot analyses and write SIGNIFICANCE_REPORT.md")
    parser.add_argument("--animal", type=str, default="rory")
    parser.add_argument("--predictions-root", type=Path, default=Path("/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("trajectory_analysis"))
    parser.add_argument("--calib-root", type=Path, default=None)
    parser.add_argument("--camera", type=str, default="Cam2005325")
    parser.add_argument("--midline-goals-json", type=Path, default=None)
    parser.add_argument("--reward-times", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    args = parser.parse_args()

    animal = args.animal.strip()
    predictions_root = Path(args.predictions_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_dir = out_root / animal

    trials = get_animal_trials(predictions_root, animal)
    if not trials:
        print(f"No {animal} trials found.")
        return
    early, mid, late = _split_phase_trials(trials)

    params = None
    if args.midline_goals_json and args.midline_goals_json.exists():
        with open(args.midline_goals_json) as f:
            params = json.load(f)
    if params is None and (out_root / animal / "midline_and_goals" / "midline_and_goals.json").exists():
        with open(out_root / animal / "midline_and_goals" / "midline_and_goals.json") as f:
            params = json.load(f)
    if params is None:
        print("No params (midline_and_goals); skipping significance report.")
        return

    calib_root = Path(args.calib_root).resolve() if args.calib_root else None
    reward_times_path = args.reward_times or out_root / "reward_times.csv"
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else None
    reward_frame_by_trial = _get_reward_frame_by_trial(trials, Path(reward_times_path).resolve(), logs_dir, animal)
    if not reward_frame_by_trial:
        print("No reward frames; skipping.")
        return

    run_tests_and_report(
        trials, early, mid, late,
        reward_frame_by_trial, params,
        calib_root=calib_root,
        camera=args.camera,
        out_dir=out_dir,
        animal=animal,
    )


if __name__ == "__main__":
    main()
