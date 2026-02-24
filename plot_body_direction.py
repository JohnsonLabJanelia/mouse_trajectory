#!/usr/bin/env python3
"""
Body direction analysis (tail → ear midpoint) and body_vs_head (where/when head scans while body is still).
Outputs: trajectory_analysis/<animal>/body_direction/, trajectory_analysis/<animal>/body_vs_head/
"""
from pathlib import Path
import argparse
import json

from plot_head_direction import (
    get_animal_trials,
    _split_phase_trials,
    plot_head_direction_angle_reference,
    plot_body_direction_by_phase,
    _run_all_body_direction_analyses,
    run_body_vs_head_analysis,
    _get_reward_frame_by_trial,
)
import analyze_trajectories as at


def main():
    parser = argparse.ArgumentParser(
        description="Body direction (tail→ear mid) and body_vs_head → trajectory_analysis/<animal>/body_direction/ and body_vs_head/"
    )
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
    out_dir_body = out_root / animal / "body_direction"
    out_dir_vs = out_root / animal / "body_vs_head"
    out_dir_body.mkdir(parents=True, exist_ok=True)
    out_dir_vs.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir_body / "body_direction_angle_reference.png"
    plot_head_direction_angle_reference(
        ref_path,
        direction_label="Body direction",
        sublabel="tail → ear midpoint",
    )
    print(f"  body direction angle reference -> {ref_path}")

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
    if params is None and (out_root / animal / "midline_and_goals" / "midline_and_goals.json").is_file():
        with open(out_root / animal / "midline_and_goals" / "midline_and_goals.json") as f:
            params = json.load(f)

    calib_root = Path(args.calib_root).resolve() if args.calib_root else None
    reward_times_path = args.reward_times if args.reward_times is not None else out_root / "reward_times.csv"
    reward_times_path = Path(reward_times_path).resolve()
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else None
    reward_frame_by_trial = _get_reward_frame_by_trial(trials, reward_times_path, logs_dir, animal)
    if reward_frame_by_trial:
        print(f"  reward frame: {len(reward_frame_by_trial)}/{len(trials)} trials")

    out_path_full = out_dir_body / "body_direction_by_phase.png"
    plot_body_direction_by_phase(
        early, mid, late,
        u_min, u_max, v_min, v_max,
        calib_root=calib_root,
        camera=args.camera,
        out_path=out_path_full,
        animal=animal,
        params=params,
        title_suffix=" (full path)",
    )
    print(f"  body direction by phase (full path) -> {out_path_full}")

    if reward_frame_by_trial:
        out_path_str = out_dir_body / "body_direction_by_phase_start_to_reward.png"
        plot_body_direction_by_phase(
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
        print(f"  body direction by phase (start to reward) -> {out_path_str}")
        if params is not None and "v_mid" in params and "goal1_u" in params:
            _run_all_body_direction_analyses(
                trials, early, mid, late,
                reward_frame_by_trial, params,
                calib_root=calib_root,
                camera=args.camera,
                u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max,
                out_dir=out_dir_body,
                animal=animal,
            )
        run_body_vs_head_analysis(
            trials,
            reward_frame_by_trial,
            params=params,
            calib_root=calib_root,
            camera=args.camera,
            u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max,
            out_dir=out_dir_vs,
            animal=animal,
            early=early,
            mid=mid,
            late=late,
        )


if __name__ == "__main__":
    main()
