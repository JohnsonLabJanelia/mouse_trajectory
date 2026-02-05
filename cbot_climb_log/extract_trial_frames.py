#!/usr/bin/env python3
"""
Extract video frame IDs that fall between "door: opened" and "door: closed"
for each trial in the cbot robot_manager log.

INVARIANT: Every output frame is guaranteed to match the log. Specifically, for
each row we write, every frame_id has timestamp_sys (plus any offset) in
[door_open_sec, door_close_sec] from that trial's log entries. We only use
frames that strictly fall within the logged door-open to door-close interval.

Video folders: /mnt/ssd2/{animal}/YYYY_MM_DD_HH_MM_SS/ with Cam*_meta.csv
containing frame_id and timestamp_sys (system time in nanoseconds).
Log sessions: logs/{animal}/session_YYYY-MM-DD_HH-MM-SS/robot_manager.log

We match each video folder to the closest same-day log session. For each trial
(door open → door closed) we find frames whose timestamp_sys falls in that
interval. Only trials that have at least one such frame are written; trials
with no matching frames are skipped. Frames are 180 Hz; one meta.csv is used
as reference (synchronized cameras share the same frame indices).
"""

import re
from datetime import datetime
from pathlib import Path
import csv
import argparse

# Reuse timestamp parsing from analyze_logs
from analyze_logs import parse_timestamp_to_seconds

LOGS_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_VIDEO_ROOT = Path("/mnt/ssd2")


def parse_door_open_close_pairs(
    log_path: Path,
    include_pretraining: bool = False,
) -> list[tuple[float, float]]:
    """
    Parse robot_manager.log and return a list of (door_open_sec, door_close_sec)
    in chronological order (Unix seconds).
    """
    pairs: list[tuple[float, float]] = []
    door_open_sec: float | None = None

    with open(log_path, "r") as f:
        for line in f:
            if ",  robot_manager_logger, info, " not in line:
                continue
            parts = line.split(",  robot_manager_logger, info, ", 1)
            if len(parts) != 2:
                continue
            ts_str, message = parts[0].strip(), parts[1].strip()
            ts_sec = parse_timestamp_to_seconds(ts_str)

            is_pretraining = "pretraining" in message

            if message.startswith("door: opened"):
                # By default, ignore pretraining trials; only include them when
                # explicitly requested via include_pretraining=True.
                if is_pretraining and not include_pretraining:
                    continue
                door_open_sec = ts_sec
            elif message.strip() == "door: closed" and door_open_sec is not None:
                pairs.append((door_open_sec, ts_sec))
                door_open_sec = None

    return pairs


def video_folder_to_datetime(folder_name: str) -> datetime | None:
    """Parse YYYY_MM_DD_HH_MM_SS to datetime."""
    try:
        return datetime.strptime(folder_name, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return None


def session_folder_to_datetime(folder_name: str) -> datetime | None:
    """Parse session_YYYY-MM-DD_HH-MM-SS to datetime."""
    if not folder_name.startswith("session_"):
        return None
    rest = folder_name.replace("session_", "")
    try:
        return datetime.strptime(rest, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def find_matching_log_session(
    video_folder_name: str,
    animal: str,
    logs_dir: Path,
) -> Path | None:
    """
    Find the log session that best matches the video folder (same date,
    closest session time). Returns path to robot_manager.log or None.
    """
    v_dt = video_folder_to_datetime(video_folder_name)
    if v_dt is None:
        return None
    animal_logs = logs_dir / animal
    if not animal_logs.is_dir():
        return None

    best_path: Path | None = None
    best_diff_sec: float = float("inf")

    for session_dir in animal_logs.iterdir():
        if not session_dir.is_dir():
            continue
        s_dt = session_folder_to_datetime(session_dir.name)
        if s_dt is None:
            continue
        if s_dt.date() != v_dt.date():
            continue
        log_path = session_dir / "robot_manager.log"
        if not log_path.is_file():
            continue
        diff_sec = abs((v_dt - s_dt).total_seconds())
        if diff_sec < best_diff_sec:
            best_diff_sec = diff_sec
            best_path = log_path

    return best_path


def list_video_folders(video_root: Path, animal: str) -> list[tuple[str, Path]]:
    """List (folder_name, path) for directories matching YYYY_MM_DD_HH_MM_SS."""
    base = video_root / animal
    if not base.is_dir():
        return []
    pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
    out: list[tuple[str, Path]] = []
    for p in base.iterdir():
        if p.is_dir() and pattern.match(p.name):
            out.append((p.name, p))
    return sorted(out, key=lambda x: x[0])


def load_meta_timestamps(meta_csv: Path) -> list[tuple[int, float]]:
    """
    Load (frame_id, timestamp_sys_seconds) from a Cam*_meta.csv.
    timestamp_sys is assumed to be in nanoseconds; convert to seconds.
    """
    rows: list[tuple[int, float]] = []
    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            try:
                fid = int(line["frame_id"])
                ts_ns = int(line["timestamp_sys"])
                ts_sec = ts_ns / 1e9
                rows.append((fid, ts_sec))
            except (KeyError, ValueError):
                continue
    return rows


def frames_in_interval(
    meta_rows: list[tuple[int, float]],
    open_sec: float,
    close_sec: float,
    video_offset_sec: float = 0.0,
) -> list[int]:
    """
    Return frame_ids whose timestamp (plus offset) is strictly within the log
    interval: open_sec <= (timestamp_sys_sec + video_offset_sec) <= close_sec.
    Only frames that match the log interval are returned.
    """
    return [
        fid for fid, ts_sec in meta_rows
        if open_sec <= (ts_sec + video_offset_sec) <= close_sec
    ]


def _frames_match_log(
    meta_rows: list[tuple[int, float]],
    frame_ids: list[int],
    open_sec: float,
    close_sec: float,
    video_offset_sec: float,
) -> bool:
    """
    Verify that every frame_id has timestamp in [open_sec, close_sec].
    Ensures output always matches the log.
    """
    ts_by_fid = {fid: ts for fid, ts in meta_rows}
    for fid in frame_ids:
        ts = ts_by_fid.get(fid)
        if ts is None:
            return False
        t = ts + video_offset_sec
        if not (open_sec <= t <= close_sec):
            return False
    return True


def get_first_meta_csv(video_folder: Path) -> Path | None:
    """Return path to first Cam*_meta.csv in the folder (alphabetically)."""
    metas = sorted(video_folder.glob("Cam*_meta.csv"))
    return metas[0] if metas else None


def run(
    animal: str,
    video_root: Path = DEFAULT_VIDEO_ROOT,
    logs_dir: Path = LOGS_DIR,
    output_csv: Path | None = None,
    verbose: bool = True,
    video_time_offset_sec: float = 0.0,
    include_pretraining: bool = False,
) -> list[dict]:
    """
    For each video folder, match the closest same-day log session, extract door
    open/close pairs, and write only trials where at least one frame has
    timestamp in [door_open_sec, door_close_sec]. Every written frame is
    verified to match the log interval; trials with no matching frames are skipped.
    """
    video_folders = list_video_folders(video_root, animal)
    if not video_folders and verbose:
        print(f"No video folders found under {video_root / animal}")
    results: list[dict] = []

    for folder_name, folder_path in video_folders:
        log_path = find_matching_log_session(folder_name, animal, logs_dir)
        if log_path is None:
            continue
        pairs = parse_door_open_close_pairs(
            log_path, include_pretraining=include_pretraining
        )
        if not pairs:
            continue
        meta_path = get_first_meta_csv(folder_path)
        if meta_path is None:
            continue
        meta_rows = load_meta_timestamps(meta_path)
        if not meta_rows:
            continue

        session_name = log_path.parent.name
        n_written = 0
        for trial_idx, (open_sec, close_sec) in enumerate(pairs):
            frame_ids = frames_in_interval(
                meta_rows, open_sec, close_sec, video_time_offset_sec
            )
            if not frame_ids:
                continue
            # Only write if every frame strictly matches this trial's log interval
            if not _frames_match_log(
                meta_rows, frame_ids, open_sec, close_sec, video_time_offset_sec
            ):
                continue
            n_written += 1
            results.append({
                "animal": animal,
                "video_folder": folder_name,
                "session": session_name,
                "trial_index": trial_idx,
                "door_open_sec": open_sec,
                "door_close_sec": close_sec,
                "frame_id_start": min(frame_ids),
                "frame_id_end": max(frame_ids),
                "frame_count": len(frame_ids),
                "frame_ids": frame_ids,
            })
        if verbose and n_written > 0:
            print(f"  {folder_name} -> {session_name}: {n_written} trials, from {meta_path.name}")

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "animal", "video_folder", "session", "trial_index",
                "door_open_sec", "door_close_sec", "frame_id_start", "frame_id_end", "frame_count",
            ])
            writer.writeheader()
            for r in results:
                row = {k: r.get(k) for k in writer.fieldnames}
                writer.writerow(row)
        if verbose:
            print(f"Wrote {len(results)} rows to {output_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frame IDs between door open and door closed per trial.",
    )
    parser.add_argument("animal", nargs="?", default="mickey", help="Animal name (e.g. mickey)")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help=f"Root directory for video folders (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Logs directory containing animal/session_.../robot_manager.log",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV path (trial -> frame_id_start, frame_id_end, frame_count)",
    )
    parser.add_argument(
        "--include-pretraining",
        action="store_true",
        help="Include pretraining trials (default: training only)",
    )
    parser.add_argument(
        "--video-time-offset-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Add SEC to each video frame timestamp when matching to log",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    run(
        animal=args.animal,
        video_root=args.video_root,
        logs_dir=args.logs_dir,
        output_csv=args.output,
        verbose=not args.quiet,
        video_time_offset_sec=args.video_time_offset_sec,
        include_pretraining=args.include_pretraining,
    )


if __name__ == "__main__":
    main()
