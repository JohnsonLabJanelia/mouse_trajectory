#!/usr/bin/env python3
"""
Export first reward time (time from door open to "reward:") per trial from
robot_manager.log for each animal/session. Saves a CSV that trajectory
scripts can use without re-parsing logs.

Output CSV columns: animal, video_folder, trial_index, time_to_target_sec
- video_folder is derived from log session name (session_2025-12-23_16-57-09 -> 2025_12_23_16_57_09)
  so trajectory trials (session_folder = animal_video_folder) can look up by (animal, video_folder, trial_index).
- time_to_target_sec = seconds from door open to first reward in that trial (None if no reward).

Usage:
  python cbot_climb_log/export_reward_times.py
  python cbot_climb_log/export_reward_times.py --logs-dir /path/to/logs -o trajectory_analysis/reward_times.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from analyze_logs import parse_robot_manager_log

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def session_name_to_video_folder(session_name: str) -> str:
    """Convert log session dir name to video_folder style for lookup.
    session_2025-12-23_16-57-09 -> 2025_12_23_16_57_09
    """
    if not session_name.startswith("session_"):
        return session_name
    rest = session_name.replace("session_", "")
    return rest.replace("-", "_")


def export_reward_times(logs_dir: Path) -> pd.DataFrame:
    """Parse all robot_manager.log files under logs_dir and return DataFrame
    with columns: animal, video_folder, trial_index, time_to_target_sec.
    Includes pretraining and training trials; time_to_target_sec is None when no reward.
    """
    logs_dir = Path(logs_dir)
    rows = []
    for animal_dir in sorted(logs_dir.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        for session_dir in sorted(animal_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_name = session_dir.name
            log_path = session_dir / "robot_manager.log"
            if not log_path.is_file():
                continue
            video_folder = session_name_to_video_folder(session_name)
            pretraining, training = parse_robot_manager_log(log_path)
            # pretraining[i] = (ts_tuple, trial_idx, duration_sec, time_to_reward, time_to_target, None, None)
            for trial in pretraining:
                trial_index = trial[1]
                time_to_target = trial[4]
                rows.append({
                    "animal": animal,
                    "video_folder": video_folder,
                    "trial_index": trial_index,
                    "time_to_target_sec": time_to_target,
                })
            for trial in training:
                trial_index = trial[1]
                time_to_target = trial[4]
                rows.append({
                    "animal": animal,
                    "video_folder": video_folder,
                    "trial_index": trial_index,
                    "time_to_target_sec": time_to_target,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export first reward time per trial from robot_manager logs (per animal)."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Root directory containing animal/session_.../robot_manager.log",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "trajectory_analysis" / "reward_times.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = export_reward_times(args.logs_dir)
    if df.empty:
        print("No log sessions or trials found")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    n_with_reward = df["time_to_target_sec"].notna().sum()
    print(f"Saved {len(df)} trials ({n_with_reward} with reward time) to {args.output}")


if __name__ == "__main__":
    main()
