#!/usr/bin/env python3
"""
Match trajectory trials in predictions3D to cbot_climb_log robot_manager logs and
export trial type (vertical vs angle, left/right) to CSV for use by analyze_trajectories.

Supports:
  - Flat layout: predictions_dir contains Predictions_3D_trial_* directly (single animal).
  - Nested layout: predictions_dir contains {animal}_{video_folder}/Predictions_3D_trial_*/
    (e.g. rory_2025_12_23_16_57_09, wilfred_2026_01_08_16_05_24).
  - Multiple animals: use --animals rory wilfred (or leave unset to auto-detect from nested dirs).

Matching:
  1. If trial_frames.csv exists and has (animal, frame_id_start, frame_id_end) -> (session, trial_index),
     use that for exact frame-range match.
  2. Else: find log session for (animal, video_folder) via same-day closest session
     (extract_trial_frames.find_matching_log_session), and use prediction trial number
     (Predictions_3D_trial_0000 -> 0) as session_trial_index. Reliable only if trial order matches log.

Usage:
  python cbot_climb_log/export_trial_types_for_trajectories.py
  python cbot_climb_log/export_trial_types_for_trajectories.py --predictions-dir /path/to/predictions3D --animals rory wilfred -o trajectory_analysis/trial_types.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from analyze_logs import parse_robot_manager_log
from extract_trial_frames import find_matching_log_session

LOGS_DIR = Path(__file__).resolve().parent / "logs"

# Nested session folder: rory_2025_12_23_16_57_09 -> (rory, 2025_12_23_16_57_09)
SESSION_PATTERN = re.compile(r"^([a-z]+)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")
TRIAL_PATTERN = re.compile(r"^Predictions_3D_trial_(\d+)_(\d+)-(\d+)$")


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


def _collect_trials_from_predictions(predictions_dir: Path) -> list[tuple[str, str, str, int, int, int]]:
    """
    Return list of (trial_id, animal, video_folder, trial_num, frame_start, frame_end)
    for each folder with trajectory_filtered.csv. Supports flat and nested layout.
    """
    predictions_dir = Path(predictions_dir)
    out: list[tuple[str, str, str, int, int, int]] = []

    for d in sorted(predictions_dir.iterdir()):
        if not d.is_dir():
            continue
        # Nested: session folder like rory_2025_12_23_16_57_09
        m_session = SESSION_PATTERN.match(d.name)
        if m_session:
            animal, video_folder = m_session.group(1), m_session.group(2)
            for t in sorted(d.iterdir()):
                if not t.is_dir():
                    continue
                mt = TRIAL_PATTERN.match(t.name)
                if not mt:
                    continue
                if not (t / "trajectory_filtered.csv").exists():
                    continue
                trial_num = int(mt.group(1))
                frame_start = int(mt.group(2))
                frame_end = int(mt.group(3))
                out.append((t.name, animal, video_folder, trial_num, frame_start, frame_end))
            continue
        # Flat: Predictions_3D_trial_* directly under predictions_dir (no animal/video_folder)
        mt = TRIAL_PATTERN.match(d.name)
        if mt and (d / "trajectory_filtered.csv").exists():
            trial_num = int(mt.group(1))
            frame_start = int(mt.group(2))
            frame_end = int(mt.group(3))
            out.append((d.name, "", "", trial_num, frame_start, frame_end))

    return out


def main():
    parser = argparse.ArgumentParser(description="Export trial types (angle/side) from logs for trajectory trials.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "predictions3D",
        help="Path to predictions3D (flat or nested by animal_video_folder)",
    )
    parser.add_argument(
        "--trial-frames",
        type=Path,
        default=Path(__file__).resolve().parent / "trial_frames.csv",
        help="Path to trial_frames.csv (optional; used for frame-range match)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "trajectory_analysis" / "trial_types.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--animals",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Animals to include (e.g. rory wilfred). For nested layout only; flat layout uses --animal.",
    )
    parser.add_argument(
        "--animal",
        default=None,
        help="Single animal for flat predictions layout (default: mickey if unset and flat)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Logs directory (logs/animal/session_.../robot_manager.log)",
    )
    args = parser.parse_args()

    predictions_dir = args.predictions_dir.resolve()
    trial_frames_path = args.trial_frames.resolve()
    out_path = args.output.resolve()
    logs_dir = args.logs_dir.resolve()

    trials_in_order = _collect_trials_from_predictions(predictions_dir)
    if not trials_in_order:
        print("No trials with trajectory_filtered.csv found under predictions_dir")
        return

    # Optional: frame range -> (session, trial_index) from trial_frames, keyed by (animal, frame_start, frame_end)
    frame_to_session: dict[tuple[str, int, int], tuple[str, int]] = {}
    if trial_frames_path.is_file():
        tf = pd.read_csv(trial_frames_path)
        for _, row in tf.iterrows():
            anim = str(row["animal"]).strip()
            key = (anim, int(row["frame_id_start"]), int(row["frame_id_end"]))
            frame_to_session[key] = (str(row["session"]).strip(), int(row["trial_index"]))

    # Flat layout: trials have animal="", video_folder="". Use --animal for lookups.
    flat_animal = args.animal or "mickey"
    if args.animals:
        animals_set = set(a.lower() for a in args.animals)
        trials_in_order = [t for t in trials_in_order if not t[1] or t[1].lower() in animals_set]

    log_cache: dict[tuple[str, str], list] = {}

    def get_training_trials(anim: str, session: str) -> list:
        key = (anim, session)
        if key not in log_cache:
            log_path = logs_dir / anim / session / "robot_manager.log"
            if not log_path.is_file():
                log_cache[key] = []
                return []
            _, training = parse_robot_manager_log(log_path)
            log_cache[key] = training
        return log_cache[key]

    def resolve_session_and_index(animal: str, video_folder: str, frame_start: int, frame_end: int, trial_num: int) -> tuple[str, int]:
        """Get (session, trial_index). Prefer trial_frames; else match log session by video_folder and use trial_num."""
        anim = animal if animal else flat_animal
        key = (anim, frame_start, frame_end)
        if key in frame_to_session:
            return frame_to_session[key]
        if video_folder and anim:
            log_path = find_matching_log_session(video_folder, anim, logs_dir)
            if log_path is not None:
                session_name = log_path.parent.name
                return (session_name, trial_num)
        return ("", -1)

    rows = []
    for trial_id, animal, video_folder, trial_num, frame_start, frame_end in trials_in_order:
        anim = animal if animal else flat_animal
        session, session_trial_index = resolve_session_and_index(
            animal, video_folder, frame_start, frame_end, trial_num
        )
        if session == "" or session_trial_index < 0:
            rows.append({
                "trial_id": trial_id,
                "animal": anim,
                "session": "",
                "session_trial_index": -1,
                "left_angle_deg": None,
                "right_angle_deg": None,
                "trial_type": "no_match_trial_frames_or_log_session",
            })
            continue
        training = get_training_trials(anim, session)
        if session_trial_index >= len(training):
            rows.append({
                "trial_id": trial_id,
                "animal": anim,
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
            "animal": anim,
            "session": session,
            "session_trial_index": session_trial_index,
            "left_angle_deg": left_angle,
            "right_angle_deg": right_angle,
            "trial_type": trial_type,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_with_type = sum(1 for r in rows if r.get("left_angle_deg") is not None)
    print(f"Saved {len(rows)} trials to {out_path} ({n_with_type} with trial type from logs)")


if __name__ == "__main__":
    main()
