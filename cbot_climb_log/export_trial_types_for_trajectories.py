#!/usr/bin/env python3
"""
Find the 50 trajectory trials (for mickey) in predictions3D, match them to cbot_climb_log
trial_frames.csv and robot_manager logs, and export trial type (vertical vs angle, side) to CSV.

Usage (from repo root):
  python cbot_climb_log/export_trial_types_for_trajectories.py
  python cbot_climb_log/export_trial_types_for_trajectories.py --predictions-dir predictions3D --output trajectory_analysis/trial_types.csv

Reads:
  - cbot_climb_log/trial_frames.csv (animal, video_folder, session, trial_index, frame_id_start, frame_id_end)
  - cbot_climb_log/logs/{animal}/{session}/robot_manager.log (via analyze_logs.parse_robot_manager_log)

Writes:
  - trajectory_analysis/trial_types.csv (trial_id, animal, session, session_trial_index, left_angle_deg, right_angle_deg, trial_type)
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from analyze_logs import parse_robot_manager_log

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _format_trial_type(left_deg: float | None, right_deg: float | None) -> str:
    """Describe trial as vertical vs angle and which side. 360 = vertical."""
    if left_deg is None or right_deg is None:
        return "unknown"
    left_v = left_deg == 360
    right_v = right_deg == 360
    if left_v and right_v:
        return "vertical vs vertical"
    if left_v and not right_v:
        return f"vertical (left) vs {right_deg:.0f} deg (right)"
    if right_v and not left_v:
        return f"vertical (right) vs {left_deg:.0f} deg (left)"
    return f"left {left_deg:.0f} deg vs right {right_deg:.0f} deg"


def main():
    parser = argparse.ArgumentParser(description="Export trial types (angle/side) for trajectory trials to CSV.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "predictions3D",
        help="Path to predictions3D (contains Predictions_3D_trial_* folders)",
    )
    parser.add_argument(
        "--trial-frames",
        type=Path,
        default=Path(__file__).resolve().parent / "trial_frames.csv",
        help="Path to trial_frames.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "trajectory_analysis" / "trial_types.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--animal",
        default="mickey",
        help="Animal to match (default: mickey)",
    )
    args = parser.parse_args()

    predictions_dir = args.predictions_dir.resolve()
    trial_frames_path = args.trial_frames.resolve()
    out_path = args.output.resolve()
    animal = args.animal

    if not trial_frames_path.is_file():
        raise FileNotFoundError(f"trial_frames.csv not found: {trial_frames_path}")

    tf = pd.read_csv(trial_frames_path)
    tf = tf[tf["animal"] == animal].copy()
    if tf.empty:
        raise ValueError(f"No rows for animal={animal!r} in {trial_frames_path}")

    # Build lookup: (frame_id_start, frame_id_end) -> (session, trial_index)
    frame_to_session = {}
    for _, row in tf.iterrows():
        key = (int(row["frame_id_start"]), int(row["frame_id_end"]))
        frame_to_session[key] = (row["session"], int(row["trial_index"]))

    # List trajectory trials: folders matching Predictions_3D_trial_N_Start-End with trajectory_filtered.csv
    pattern = re.compile(r"^Predictions_3D_trial_(\d+)_(\d+)-(\d+)$")
    trials_in_order = []
    for d in sorted(predictions_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        if not (d / "trajectory_filtered.csv").exists():
            continue
        trial_num = int(m.group(1))
        frame_start = int(m.group(2))
        frame_end = int(m.group(3))
        trials_in_order.append((d.name, trial_num, frame_start, frame_end))

    # Cache parsed logs per (animal, session)
    log_cache: dict[tuple[str, str], list] = {}

    def get_training_trials(anim: str, session: str) -> list:
        key = (anim, session)
        if key not in log_cache:
            log_path = LOGS_DIR / anim / session / "robot_manager.log"
            if not log_path.is_file():
                log_cache[key] = []
                return []
            _, training = parse_robot_manager_log(log_path)
            log_cache[key] = training
        return log_cache[key]

    rows = []
    for trial_id, _trial_num, frame_start, frame_end in trials_in_order:
        key = (frame_start, frame_end)
        if key not in frame_to_session:
            rows.append({
                "trial_id": trial_id,
                "animal": animal,
                "session": "",
                "session_trial_index": -1,
                "left_angle_deg": None,
                "right_angle_deg": None,
                "trial_type": "no_match_in_trial_frames",
            })
            continue
        session, session_trial_index = frame_to_session[key]
        training = get_training_trials(animal, session)
        if session_trial_index >= len(training):
            rows.append({
                "trial_id": trial_id,
                "animal": animal,
                "session": session,
                "session_trial_index": session_trial_index,
                "left_angle_deg": None,
                "right_angle_deg": None,
                "trial_type": "trial_index_out_of_range",
            })
            continue
        _, _, _, _, _, left_angle, right_angle = training[session_trial_index]
        trial_type = _format_trial_type(left_angle, right_angle)
        rows.append({
            "trial_id": trial_id,
            "animal": animal,
            "session": session,
            "session_trial_index": session_trial_index,
            "left_angle_deg": left_angle,
            "right_angle_deg": right_angle,
            "trial_type": trial_type,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} trials to {out_path}")


if __name__ == "__main__":
    main()
