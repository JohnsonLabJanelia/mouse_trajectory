# Extracting video frames by door open/closed

This note describes how we identify video frames that fall between **door open** and **door closed** for each trial, and how to run the extraction script.

## Overview

- **Log**: `robot_manager.log` in each session folder records events `door: opened` and `door: closed` with timestamps.
- **Videos**: Recordings live under `/mnt/ssd2/{animal}/YYYY_MM_DD_HH_MM_SS/` with multiple synchronized cameras. Each camera has a `Cam*_meta.csv` with `frame_id` and **system timestamp** (`timestamp_sys`, in nanoseconds).
- **Goal**: For each trial (one door open → door closed interval), find which frame IDs have a system time inside that interval. Those frame IDs can then be used to cut or analyze only the trial segment from the 180 Hz video.

## How frame extraction works

1. **Door open/closed pairs from the log**  
   We parse `robot_manager.log` and collect every pair where a line starts with `door: opened` (possibly with extra text like `(auto)` or `(pretraining)`) followed later by a line that is exactly `door: closed`.  
   - **By default** we treat only **training** trials: any `door: opened` line whose message contains `pretraining` is **ignored**.  
   - If you want to include pretraining trials as well, run the script with `--include-pretraining`.  
   The log timestamps are in the form `YYYY_MM_DD_HH_MM_SS_mmm`. We convert them to Unix seconds (using `parse_timestamp_to_seconds` from `analyze_logs.py`) so we have `(door_open_sec, door_close_sec)` for each trial.

2. **Matching video folders to log sessions**  
   Video folders are named `YYYY_MM_DD_HH_MM_SS`. For each such folder we look for the cbot log session for the **same animal** and **same calendar date** whose session time is **closest** to the video folder time. That session’s `robot_manager.log` is used for door open/closed events.

3. **Frame timestamps**  
   In the chosen video folder we use one `Cam*_meta.csv` (e.g. the first alphabetically). It has:
   - `frame_id`: 0, 1, 2, …
   - `timestamp_sys`: system time in **nanoseconds** (same clock as the log when synchronized).

   We convert `timestamp_sys` to seconds: `ts_sec = timestamp_sys / 1e9`.

4. **Which frames belong to a trial**  
   For each trial `(door_open_sec, door_close_sec)` we keep every row in the meta CSV where  
   `door_open_sec <= (ts_sec + video_offset) <= door_close_sec`  
   and record their `frame_id`s. By default `video_offset` is 0. If the video system clock is not aligned with the log (e.g. `timestamp_sys` is seconds since boot), you can pass `--video-time-offset-sec` to shift video times. Because all cameras are synchronized, the same frame indices apply to all `Cam*.mp4` in that folder; we only need one meta file to get the frame range.

5. **Output**  
   For each trial we get:
   - `frame_id_start`, `frame_id_end`: first and last frame in the interval
   - `frame_count`: number of frames
   - Optionally the full list of `frame_id`s (e.g. from the script’s return value or a future JSON export).

## How to run the script

**Script**: `extract_trial_frames.py` (in this repo).

**Dependencies**: Uses `analyze_logs.parse_timestamp_to_seconds`; run from the repo root so that `analyze_logs` is importable.

**Default paths**:
- Logs: `logs/` (next to the script), i.e. `logs/{animal}/session_YYYY-MM-DD_HH-MM-SS/robot_manager.log`
- Videos: `/mnt/ssd2/{animal}/YYYY_MM_DD_HH_MM_SS/`

**Examples**:

```bash
# From the repo root (cbot_climb_log)
cd /path/to/cbot_climb_log

# Mickey, write results to CSV
python3 extract_trial_frames.py mickey -o trial_frames.csv

# Another animal
python3 extract_trial_frames.py rory -o rory_trial_frames.csv

# Custom video root and logs directory
python3 extract_trial_frames.py mickey --video-root /mnt/ssd2 --logs-dir ./logs -o trial_frames.csv

# Less console output
python3 extract_trial_frames.py mickey -o trial_frames.csv -q

# Video clock is seconds since boot; add offset so frames align with log (e.g. first door open at 1767131013)
python3 extract_trial_frames.py rory --video-time-offset-sec 1767131013 -o rory_trial_frames.csv

# Include pretraining trials as well (normally they are skipped)
python3 extract_trial_frames.py rory --include-pretraining -o rory_trial_frames_with_pretraining.csv
```

**Output CSV columns**:  
`animal`, `video_folder`, `session`, `trial_index`, `door_open_sec`, `door_close_sec`, `frame_id_start`, `frame_id_end`, `frame_count`.

Use `frame_id_start` and `frame_id_end` (or the full frame list if you add JSON export) to slice the corresponding segment from any of the synchronized `Cam*.mp4` files for that video folder.
