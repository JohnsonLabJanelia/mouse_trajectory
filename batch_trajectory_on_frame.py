#!/usr/bin/env python3
"""
Batch: for all JARVIS 3D predictions (rory/wilfred), extract the trial start frame
from video and plot trajectory on frame. Writes trajectory_filtered.csv per trial
so analyze_trajectories.py can run on the same root.

Videos are read from video_root (e.g. /mnt/mouse2). Session folders are
{predictions_root}/{animal}_{video_folder}/ with trial folders
Predictions_3D_trial_XXXX_frameStart-frameEnd/. The frame number used is
frameStart from the folder name; video is video_root/{animal}/{video_folder}/.

Usage:
  python batch_trajectory_on_frame.py --predictions-root /path/to/predictions3D --video-root /mnt/mouse2
  python batch_trajectory_on_frame.py --animals rory wilfred --camera Cam2005325
"""

from pathlib import Path
import argparse
import re
import subprocess
import sys

# Session folder pattern: rory_2025_12_23_16_57_09, wilfred_2026_01_08_16_05_24
SESSION_PATTERN = re.compile(r"^([a-z]+)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")
TRIAL_PATTERN = re.compile(r"^Predictions_3D_trial_\d+_\d+-\d+$")

DEFAULT_PREDICTIONS_ROOT = Path(
    "/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D"
)
DEFAULT_VIDEO_ROOT = Path("/mnt/mouse2")
DEFAULT_CAMERA = "Cam2005325"


def iter_sessions_and_trials(predictions_root: Path, animals: list[str]):
    """Yield (session_dir, trial_dir) for each trial under session dirs matching animals."""
    predictions_root = Path(predictions_root)
    for session_dir in sorted(predictions_root.iterdir()):
        if not session_dir.is_dir():
            continue
        m = SESSION_PATTERN.match(session_dir.name)
        if not m:
            continue
        animal, video_folder = m.group(1), m.group(2)
        if animal not in animals:
            continue
        for trial_dir in sorted(session_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            if not TRIAL_PATTERN.match(trial_dir.name):
                continue
            yield session_dir, trial_dir


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract frame + plot trajectory for JARVIS predictions (rory/wilfred)."
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help=f"Root containing session folders (default: {DEFAULT_PREDICTIONS_ROOT})",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help=f"Root for videos: video_root/animal/video_folder/ (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        default=["rory", "wilfred"],
        help="Animals to process (default: rory wilfred)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help=f"Camera name (default: {DEFAULT_CAMERA})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials that already have trajectory_filtered.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list trials that would be processed",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    predictions_root = Path(args.predictions_root).resolve()
    video_root = Path(args.video_root).resolve()

    if not predictions_root.is_dir():
        raise SystemExit(f"Not a directory: {predictions_root}")

    trials = list(iter_sessions_and_trials(predictions_root, args.animals))
    print(f"Found {len(trials)} trials under {predictions_root} for animals {args.animals}")

    if args.dry_run:
        for _session_dir, trial_dir in trials:
            print(f"  {trial_dir.relative_to(predictions_root)}")
        return

    extract_script = script_dir / "extract_first_frame.py"
    plot_script = script_dir / "plot_trajectory_on_frame.py"
    for script in (extract_script, plot_script):
        if not script.is_file():
            raise SystemExit(f"Script not found: {script}")

    ok = 0
    err = 0
    for session_dir, trial_dir in trials:
        rel = trial_dir.relative_to(predictions_root)
        if args.skip_existing and (trial_dir / "trajectory_filtered.csv").exists():
            continue
        if not (trial_dir / "data3D.csv").exists():
            print(f"  [skip] {rel} (no data3D.csv)")
            continue

        # Recording path: video_root/animal/video_folder
        m = SESSION_PATTERN.match(session_dir.name)
        animal, video_folder = m.group(1), m.group(2)
        recording_path = video_root / animal / video_folder

        # 1) Extract trial start frame to trial_dir/frame.png
        frame_png = trial_dir / "frame.png"
        if not frame_png.exists():
            cmd_extract = [
                sys.executable,
                str(extract_script),
                str(trial_dir),
                str(frame_png),
                "--camera",
                args.camera,
                "--recording-path",
                str(recording_path),
            ]
            r = subprocess.run(cmd_extract, cwd=script_dir, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [FAIL] {rel} extract_first_frame: {r.stderr or r.stdout}")
                err += 1
                continue
        try:
            # 2) Plot trajectory on frame and write trajectory_filtered.csv
            cmd_plot = [
                sys.executable,
                str(plot_script),
                str(trial_dir),
                "--camera",
                args.camera,
                "-o",
                str(trial_dir / "trajectory_on_frame.png"),
                "--output-trajectory",
                str(trial_dir / "trajectory_filtered.csv"),
            ]
            r = subprocess.run(cmd_plot, cwd=script_dir, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [FAIL] {rel} plot_trajectory_on_frame: {r.stderr or r.stdout}")
                err += 1
                continue
        except Exception as e:
            print(f"  [FAIL] {rel} {e}")
            err += 1
            continue
        print(f"  [ok] {rel}")
        ok += 1

    print(f"\nDone: {ok} ok, {err} failed.")


if __name__ == "__main__":
    main()
