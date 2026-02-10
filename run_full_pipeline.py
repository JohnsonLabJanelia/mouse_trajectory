#!/usr/bin/env python3
"""
Full pipeline:

1. **Extract trial frame ranges per session (training trials only by default)**  
   Step 1 uses `cbot_climb_log/extract_trial_frames.py`: for each animal, find
   door open/close trials and write one CSV per session labelled
   `{animal}_{video_folder}.csv` (videos under `video_root/{animal}/`).  
   - Pretraining trials (log lines containing `pretraining`) are **ignored** by
     default; pass `--include-pretraining` if you want them included.

2. **Run JARVIS 3D prediction per session with per-date calibration**  
   Step 2 uses `trainJarvisNoGui/predict_trials.py`: for each session CSV, run
   3D prediction with `--output-subdir` so predictions go under
   `predictions/predictions3D/{animal}_{video_folder}/`.  
   - If `--dataset-name` is provided, that single calibration folder is used
     for all sessions (legacy behaviour).  
   - Otherwise, calibrations are chosen **per video date** from `--calib-root`
     (default `/home/user/red_data/calib`), preferring same-day calibration and
     falling back to the **nearest available date** when same-day is missing.

Videos are expected under: `video_root/{animal}/YYYY_MM_DD_HH_MM_SS/`.
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

# Default paths relative to this script (analyzeMiceTrajectory)
SCRIPT_DIR = Path(__file__).resolve().parent
CBOT_CLIMB_LOG = SCRIPT_DIR / "cbot_climb_log"
DEFAULT_LOGS_DIR = CBOT_CLIMB_LOG / "logs"
DEFAULT_VIDEO_ROOT = Path("/mnt/mouse2")

# Calibration roots / behaviour
DEFAULT_CALIB_ROOT = Path("/home/user/red_data/calib")
CALIB_PARAMS_ROOT = SCRIPT_DIR / "calib_params"
# Known-bad calibration dates to ignore when auto-selecting
CALIB_IGNORE_DATES = {"2025_12_19", "2025_12_21"}


def discover_animals(video_root: Path) -> list[str]:
    """List subdirs of video_root as animal names (skip hidden and non-dirs)."""
    if not video_root.is_dir():
        return []
    return sorted(
        p.name for p in video_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def run_extract_trial_frames(
    animal: str,
    video_root: Path,
    logs_dir: Path,
    output_csv: Path,
    cbot_dir: Path,
    verbose: bool,
    include_pretraining: bool = False,
    video_time_offset_sec: float = 0.0,
) -> bool:
    """Run extract_trial_frames.py for one animal; write combined CSV. Returns True on success."""
    cmd = [
        sys.executable,
        str(cbot_dir / "extract_trial_frames.py"),
        animal,
        "--video-root", str(video_root),
        "--logs-dir", str(logs_dir),
        "-o", str(output_csv),
    ]
    if include_pretraining:
        cmd.append("--include-pretraining")
    if video_time_offset_sec != 0.0:
        cmd.extend(["--video-time-offset-sec", str(video_time_offset_sec)])
    if not verbose:
        cmd.append("-q")
    result = subprocess.run(cmd, cwd=str(cbot_dir))
    return result.returncode == 0 and output_csv.is_file()


def split_trials_by_session(combined_csv: Path, output_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Read combined trial CSV and write one CSV per (animal, video_folder).
    Returns list of (animal, video_folder, path_to_session_csv).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_session: dict[tuple[str, str], list[dict]] = {}
    fieldnames = [
        "animal", "video_folder", "session", "trial_index",
        "door_open_sec", "door_close_sec", "frame_id_start", "frame_id_end", "frame_count",
    ]
    with open(combined_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with no frames (defensive: keep session CSVs clean)
            try:
                if int(row.get("frame_count") or 0) <= 0 and not (row.get("frame_id_start") or "").strip():
                    continue
            except (ValueError, TypeError):
                continue
            key = (row["animal"], row["video_folder"])
            rows_by_session.setdefault(key, []).append(row)

    out_list: list[tuple[str, str, Path]] = []
    for (animal, video_folder), rows in sorted(rows_by_session.items()):
        if not rows:
            continue
        safe_name = f"{animal}_{video_folder}"
        session_csv = output_dir / f"{safe_name}.csv"
        with open(session_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        out_list.append((animal, video_folder, session_csv))
    return out_list


def session_csv_has_valid_trials(trials_csv: Path) -> bool:
    """Return True if the CSV has at least one row with frame_count > 0 / valid frame_id_start."""
    with open(trials_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("frame_count", 0) or 0) > 0:
                    return True
                if row.get("frame_id_start", "").strip():
                    return True
            except (ValueError, TypeError):
                continue
    return False


def count_valid_trials(trials_csv: Path) -> int:
    """Count trials in a session CSV that have a valid frame range (frame_count > 0 or frame_id_start set)."""
    n = 0
    with open(trials_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_count = int(row.get("frame_count", 0) or 0)
            except (ValueError, TypeError):
                frame_count = 0
            has_start = bool((row.get("frame_id_start") or "").strip())
            if frame_count > 0 or has_start:
                n += 1
    return n


def run_predict_trials(
    recording_path: Path,
    trials_csv: Path,
    project: str,
    dataset_name: Path,
    output_subdir: str,
    train_jarvis_dir: Path,
    jarvis_hybridnet_dir: Path,
    verbose: bool,
) -> bool:
    """Run predict_trials.py for one session. Returns True on success."""
    if not session_csv_has_valid_trials(trials_csv):
        if verbose:
            print(f"[JARVIS] Skip {output_subdir}: no trials with valid frames in CSV")
        return False
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = f"{jarvis_hybridnet_dir}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(train_jarvis_dir / "predict_trials.py"),
        "--project", project,
        "--recording-path", str(recording_path),
        "--dataset-name", str(dataset_name),
        "--trials-csv", str(trials_csv),
        "--output-subdir", output_subdir,
    ]
    result = subprocess.run(cmd, cwd=str(train_jarvis_dir), env=env)
    return result.returncode == 0


def parse_video_date(video_folder: str) -> date | None:
    """Parse YYYY_MM_DD_HH_MM_SS video folder name to a date."""
    try:
        return datetime.strptime(video_folder, "%Y_%m_%d_%H_%M_%S").date()
    except ValueError:
        return None


def list_calib_dates(calib_root: Path) -> list[tuple[date, Path]]:
    """
    List available calibration dates under calib_root.

    We look for subdirectories named YYYY_MM_DD that contain a 'calibration/'
    folder, and ignore known-bad dates.
    """
    out: list[tuple[date, Path]] = []
    if not calib_root.is_dir():
        return out
    for p in calib_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name in CALIB_IGNORE_DATES:
            continue
        try:
            d = datetime.strptime(name, "%Y_%m_%d").date()
        except ValueError:
            continue
        if not (p / "calibration").is_dir():
            continue
        out.append((d, p))
    out.sort(key=lambda t: t[0])
    return out


def find_closest_calib_date(
    video_date: date | None,
    calib_dates: list[tuple[date, Path]],
) -> tuple[date, Path] | None:
    """Return (date, path) of calibration closest in days to video_date."""
    if video_date is None or not calib_dates:
        return None
    return min(calib_dates, key=lambda t: abs((t[0] - video_date).days))


def ensure_jarvis_calib(
    opencv_calib_dir: Path,
    date_name: str,
    verbose: bool,
) -> Path | None:
    """
    Ensure JARVIS-format calibration exists for this date.

    Uses convert_calibration.py to convert from OpenCV-format YAMLs under
    opencv_calib_dir into calib_params/{date_name}/.
    """
    CALIB_PARAMS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = CALIB_PARAMS_ROOT / date_name
    # Reuse if already converted
    if any(out_dir.glob("*.yaml")):
        return out_dir

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "convert_calibration.py"),
        "--input-folder",
        str(opencv_calib_dir),
        "--output-folder",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode == 0 and any(out_dir.glob("*.yaml")):
        if verbose:
            print(f"[Calib] Converted calibration for {date_name} -> {out_dir}")
        return out_dir
    if verbose:
        print(f"[Calib] Failed to convert calibration for {date_name} from {opencv_calib_dir}")
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Full pipeline: extract trial frames per session, then JARVIS 3D prediction per session.",
    )
    ap.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help=f"Root under which videos live as {{animal}}/YYYY_MM_DD_HH_MM_SS/ (default: {DEFAULT_VIDEO_ROOT})",
    )
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Logs directory with {animal}/session_YYYY-MM-DD_HH-MM-SS/robot_manager.log",
    )
    ap.add_argument(
        "--animals",
        nargs="*",
        default=None,
        metavar="A",
        help="Animals to process (default: discover from --video-root)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline_output"),
        help="Directory for per-session trial CSVs (default: pipeline_output)",
    )
    ap.add_argument(
        "--project",
        default="mouseClimb4",
        help="JARVIS project name (default: mouseClimb4)",
    )
    ap.add_argument(
        "--calib-root",
        type=Path,
        default=DEFAULT_CALIB_ROOT,
        help=f"Root with per-date calibration folders YYYY_MM_DD (default: {DEFAULT_CALIB_ROOT})",
    )
    ap.add_argument(
        "--dataset-name",
        type=Path,
        default=None,
        help=(
            "Calibration params folder for JARVIS. If set, this single folder is "
            "used for ALL sessions (legacy behaviour). If omitted, per-date "
            "calibration under --calib-root is used."
        ),
    )
    ap.add_argument(
        "--step",
        type=int,
        choices=(1, 2),
        default=None,
        metavar="N",
        help="Run only step N: 1=extract trial frames only, 2=JARVIS only (use existing CSVs). Default: run both.",
    )
    ap.add_argument(
        "--train-jarvis-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "trainJarvisNoGui",
        help="Path to trainJarvisNoGui repo",
    )
    ap.add_argument(
        "--jarvis-hybridnet-dir",
        type=Path,
        default=None,
        help="Path to JARVIS-HybridNet (for PYTHONPATH); default: SCRIPT_DIR.parent / JARVIS-HybridNet",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip step 1; use existing CSVs in output-dir/trial_frames/",
    )
    ap.add_argument(
        "--skip-jarvis",
        action="store_true",
        help="Skip step 2; only extract and write per-session CSVs",
    )
    ap.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N sessions per animal (for testing)",
    )
    ap.add_argument(
        "--include-pretraining",
        action="store_true",
        help="Include pretraining trials in extract step (default: training only)",
    )
    ap.add_argument(
        "--video-time-offset-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Add SEC to video frame timestamps when matching to log (extract step)",
    )
    ap.add_argument(
        "--confirm-before-jarvis",
        action="store_true",
        help="After step 1, prompt before running JARVIS (step 2). Use to run step-by-step in one command.",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Less console output",
    )
    args = ap.parse_args()

    if args.step == 1:
        args.skip_jarvis = True
    elif args.step == 2:
        args.skip_extract = True

    video_root = args.video_root.resolve()
    logs_dir = args.logs_dir.resolve()
    output_dir = args.output_dir.resolve()
    trial_frames_dir = output_dir / "trial_frames"
    dataset_name = args.dataset_name.resolve() if args.dataset_name is not None else None
    calib_root = args.calib_root.resolve() if args.calib_root is not None else None
    train_jarvis_dir = args.train_jarvis_dir.resolve()
    jarvis_hybridnet_dir = (args.jarvis_hybridnet_dir or SCRIPT_DIR.parent / "JARVIS-HybridNet").resolve()
    verbose = not args.quiet

    if not CBOT_CLIMB_LOG.is_dir():
        print(f"Error: cbot_climb_log not found at {CBOT_CLIMB_LOG}", file=sys.stderr)
        sys.exit(2)
    if not args.skip_jarvis:
        if not train_jarvis_dir.is_dir():
            print(f"Error: trainJarvisNoGui not found at {train_jarvis_dir}", file=sys.stderr)
            sys.exit(2)
        if not jarvis_hybridnet_dir.is_dir():
            print(f"Error: JARVIS-HybridNet not found at {jarvis_hybridnet_dir}", file=sys.stderr)
            sys.exit(2)
        # Either a single dataset-name is provided, or we rely on per-date calibration under calib_root
        if args.dataset_name is not None:
            if not dataset_name.is_dir():
                print(f"Error: --dataset-name path not found: {dataset_name}", file=sys.stderr)
                sys.exit(2)
        else:
            if calib_root is None or not calib_root.is_dir():
                print(
                    "Error: either --dataset-name or a valid --calib-root with per-date calibrations "
                    "is required for step 2 / JARVIS",
                    file=sys.stderr,
                )
                sys.exit(2)

    animals = args.animals or discover_animals(video_root)
    if not animals:
        print("No animals to process (check --video-root and --animals)", file=sys.stderr)
        sys.exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract trial frames, split into per-session CSVs ---
    sessions_to_run: list[tuple[str, str, Path]] = []
    if not args.skip_extract:
        for animal in animals:
            if verbose:
                print(f"[Extract] {animal}")
            combined = output_dir / f"{animal}_all_trials.csv"
            if not run_extract_trial_frames(
                animal, video_root, logs_dir, combined, CBOT_CLIMB_LOG, verbose,
                include_pretraining=args.include_pretraining,
                video_time_offset_sec=args.video_time_offset_sec,
            ):
                if verbose:
                    print(f"  Skipping {animal} (no trials or error)")
                continue
            sessions = split_trials_by_session(combined, trial_frames_dir)
            sessions_to_run.extend(sessions)
            if verbose:
                print(f"  -> {len(sessions)} session(s)")
    else:
        # Load existing per-session CSVs
        if not trial_frames_dir.is_dir():
            print("Error: --skip-extract but no trial_frames dir; run without --skip-extract first", file=sys.stderr)
            sys.exit(2)
        for csv_path in sorted(trial_frames_dir.glob("*.csv")):
            stem = csv_path.stem
            if "_" in stem:
                parts = stem.split("_", 1)
                animal = parts[0]
                video_folder = parts[1]
                sessions_to_run.append((animal, video_folder, csv_path))
        if verbose:
            print(f"[Skip extract] Using {len(sessions_to_run)} existing session CSVs")

    if args.limit_sessions is not None:
        sessions_to_run = sessions_to_run[: args.limit_sessions]
        if verbose:
            print(f"Limited to first {args.limit_sessions} session(s)")

    if not sessions_to_run:
        print("No sessions to run. Check video-root, logs, and extract step.")
        sys.exit(0)

    run_jarvis = not args.skip_jarvis
    if run_jarvis and args.confirm_before_jarvis:
        try:
            reply = input(f"Step 1 done. Run JARVIS for {len(sessions_to_run)} session(s)? [y/N]: ").strip().lower()
            run_jarvis = reply in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            run_jarvis = False
        if not run_jarvis and verbose:
            print("Skipping JARVIS (step 2). Run with --step 2 when ready.")

    # Precompute available calibration dates (if we are using per-date calibration)
    calib_dates: list[tuple[date, Path]] = []
    if run_jarvis and args.dataset_name is None and calib_root is not None:
        calib_dates = list_calib_dates(calib_root)
        if verbose:
            print(f"[Calib] Found {len(calib_dates)} calibration date(s) under {calib_root}")

    # --- Step 2: JARVIS predict_trials per session ---
    if run_jarvis:
        # Base directory where JARVIS saves predictions for this project.
        predictions_base = jarvis_hybridnet_dir / "projects" / args.project / "predictions" / "predictions3D"

        # Track per-animal session counts so progress is clearer than a single global index.
        from collections import Counter

        total_sessions_by_animal: Counter[str] = Counter(a for a, _, _ in sessions_to_run)
        seen_sessions_by_animal: Counter[str] = Counter()

        for i, (animal, video_folder, session_csv) in enumerate(sessions_to_run):
            recording_path = video_root / animal / video_folder
            if not recording_path.is_dir():
                if verbose:
                    print(f"[JARVIS] Skip {animal}/{video_folder}: recording path not found")
                continue

            # Choose calibration / dataset for this session
            if dataset_name is not None:
                # Legacy behaviour: single dataset for all sessions
                session_dataset: Path | None = dataset_name
            else:
                video_date = parse_video_date(video_folder)
                best = find_closest_calib_date(video_date, calib_dates)
                if best is None:
                    if verbose:
                        print(f"[JARVIS] Skip {animal}/{video_folder}: no calibration available for date {video_date}")
                    continue
                calib_date, calib_dir = best
                # Convert OpenCV-format calibration to JARVIS format on demand
                jarvis_calib = ensure_jarvis_calib(
                    calib_dir / "calibration",
                    calib_dir.name,
                    verbose,
                )
                if jarvis_calib is None:
                    if verbose:
                        print(
                            f"[JARVIS] Skip {animal}/{video_folder}: failed to prepare JARVIS calibration "
                            f"from {calib_dir}"
                        )
                    continue
                # Log when we are forced to use a different date's calibration
                if verbose and video_date is not None and calib_date != video_date:
                    print(
                        f"[Calib] Using nearest-date calibration {calib_dir.name} "
                        f"for video {video_folder} (video date {video_date})"
                    )
                session_dataset = jarvis_calib

            output_subdir = f"{animal}_{video_folder}"

            # If this session already has predictions for all valid trials, skip re-running JARVIS.
            session_output_dir = predictions_base / output_subdir
            if session_output_dir.is_dir():
                existing_trials = len(list(session_output_dir.glob("Predictions_3D_trial_*")))
                total_trials = count_valid_trials(session_csv)
                if total_trials > 0 and existing_trials >= total_trials:
                    seen_sessions_by_animal[animal] += 1
                    animal_idx = seen_sessions_by_animal[animal]
                    animal_total = total_sessions_by_animal[animal]
                    if verbose:
                        print(
                            f"[JARVIS] [{animal} {animal_idx}/{animal_total}] {output_subdir} "
                            f"(calib={session_dataset}) — already has {existing_trials}/{total_trials} trials, skipping"
                        )
                    continue

            if verbose:
                seen_sessions_by_animal[animal] += 1
                animal_idx = seen_sessions_by_animal[animal]
                animal_total = total_sessions_by_animal[animal]
                print(
                    f"[JARVIS] [{animal} {animal_idx}/{animal_total}] {output_subdir} "
                    f"(calib={session_dataset})"
                )
            run_predict_trials(
                recording_path,
                session_csv,
                args.project,
                session_dataset,
                output_subdir,
                train_jarvis_dir,
                jarvis_hybridnet_dir,
                verbose,
            )
        if verbose:
            print("Done.")
    else:
        if verbose:
            print(f"Wrote {len(sessions_to_run)} session CSVs to {trial_frames_dir}. Run with --step 2 when ready to run JARVIS.")


if __name__ == "__main__":
    main()
