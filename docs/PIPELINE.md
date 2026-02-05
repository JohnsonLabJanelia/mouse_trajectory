# Mouse Trajectory Analysis Pipeline

Complete documentation for the full pipeline that extracts trial frames from videos, runs JARVIS 3D prediction, and analyzes mouse climbing trajectories.

## Overview

The pipeline processes video recordings of mice performing climbing trials, extracts 3D trajectories using JARVIS, and enables downstream analysis. It consists of two main steps:

1. **Extract trial frame ranges** from videos based on door open/close events in robot logs
2. **Run JARVIS 3D prediction** for each trial using per-date camera calibrations

## Prerequisites

- Video recordings under `video_root/{animal}/YYYY_MM_DD_HH_MM_SS/` with synchronized cameras
- Robot manager logs under `logs/{animal}/session_YYYY-MM-DD_HH-MM-SS/robot_manager.log`
- Camera calibration folders under `calib_root/YYYY_MM_DD/calibration/` (OpenCV format YAMLs)
- JARVIS-HybridNet and trainJarvisNoGui repositories configured

## Step 1: Extract Trial Frames

**Script**: `cbot_climb_log/extract_trial_frames.py`

Extracts video frame IDs that fall between "door: opened" and "door: closed" events for each trial.

### How it works

1. **Matches video folders to log sessions**: For each video folder `YYYY_MM_DD_HH_MM_SS`, finds the closest same-day log session
2. **Parses door events**: Reads `robot_manager.log` and extracts `(door_open_sec, door_close_sec)` pairs
   - **By default, only training trials are included** (pretraining trials are ignored)
   - Use `--include-pretraining` to include pretraining trials
3. **Finds matching frames**: For each trial interval, finds frames in `Cam*_meta.csv` whose `timestamp_sys` falls within `[door_open_sec, door_close_sec]`
   - Only trials with **at least one matching frame** are written
   - Every written frame is verified to match the log interval
4. **Outputs per-session CSVs**: Writes `{animal}_{video_folder}.csv` with columns:
   - `animal`, `video_folder`, `session`, `trial_index`
   - `door_open_sec`, `door_close_sec`
   - `frame_id_start`, `frame_id_end`, `frame_count`

### Key features

- **Training-only by default**: Pretraining trials are excluded unless `--include-pretraining` is used
- **Strict log matching**: Every output frame is guaranteed to have timestamp in `[door_open_sec, door_close_sec]` from the log
- **Skips invalid trials**: Trials with no matching frames are silently skipped (not written, not logged)

### Usage

```bash
# Extract for one animal (training only)
python cbot_climb_log/extract_trial_frames.py rory -o rory_trials.csv

# Include pretraining trials
python cbot_climb_log/extract_trial_frames.py rory --include-pretraining -o rory_all_trials.csv

# Custom paths
python cbot_climb_log/extract_trial_frames.py rory \
    --video-root /mnt/mouse2 \
    --logs-dir ./logs \
    -o output.csv
```

See `cbot_climb_log/EXTRACT_TRIAL_FRAMES.md` for detailed documentation.

## Step 2: JARVIS 3D Prediction

**Script**: `trainJarvisNoGui/predict_trials.py`

Runs JARVIS 3D pose prediction for each trial, generating 3D coordinates for keypoints (Snout, EarL, EarR, Tail, etc.).

### Calibration selection

The pipeline supports two modes:

#### Per-date calibration (recommended)

- For each session `animal/YYYY_MM_DD_HH_MM_SS`:
  - Parses video date `YYYY_MM_DD`
  - Looks under `--calib-root` (default `/home/user/red_data/calib`) for `YYYY_MM_DD/calibration/`
  - **Prefers exact same-day calibration**
  - **Falls back to nearest available date** if same-day is missing
  - Converts OpenCV-format YAMLs to JARVIS format via `convert_calibration.py`
  - Stores converted calibrations in `calib_params/YYYY_MM_DD/`

#### Single calibration (legacy)

- Use `--dataset-name` to specify one calibration folder for all sessions
- Overrides per-date selection

### Output structure

Predictions are saved under:
```
JARVIS-HybridNet/projects/{project}/predictions/predictions3D/{animal}_{video_folder}/
  Predictions_3D_trial_{trial_index:04d}_{frame_start}-{frame_end}/
    data3D.csv          # 3D coordinates per frame
    info.yaml           # Metadata (recording path, calibration, frame range)
    frame.png           # Visualization (if generated)
```

### Known issues

- **Cam710040**: Always skipped in calibration (different resolution)
- **Corrupted videos**: Some sessions may have 0×0 resolution videos that cause JARVIS to fail
  - See `ISSUE_LOG.md` for details on problematic sessions

## Full Pipeline Usage

**Script**: `run_full_pipeline.py`

Orchestrates both steps, handling per-session CSV generation and per-date calibration selection.

### Basic usage

```bash
# Run both steps for all animals
python run_full_pipeline.py

# Run for specific animals
python run_full_pipeline.py --animals rory wilfred

# Run only step 1 (extract)
python run_full_pipeline.py --step 1 --animals rory wilfred

# Run only step 2 (JARVIS, using existing CSVs)
python run_full_pipeline.py --step 2 --animals rory wilfred
```

### Key options

- `--video-root PATH`: Root directory for videos (default: `/mnt/mouse2`)
- `--logs-dir PATH`: Root directory for robot logs (default: `cbot_climb_log/logs`)
- `--calib-root PATH`: Root for per-date calibrations (default: `/home/user/red_data/calib`)
- `--dataset-name PATH`: Override with single calibration folder (legacy mode)
- `--include-pretraining`: Include pretraining trials in step 1
- `--output-dir PATH`: Output directory (default: `pipeline_output`)
- `--project NAME`: JARVIS project name (default: `mouseClimb4`)
- `--step {1,2}`: Run only one step
- `--limit-sessions N`: Process only first N sessions (for testing)
- `--confirm-before-jarvis`: Prompt before running step 2
- `-q, --quiet`: Less verbose output

### Example: Full run with per-date calibration

```bash
python run_full_pipeline.py \
    --animals rory wilfred \
    --video-root /mnt/mouse2 \
    --calib-root /home/user/red_data/calib
```

### Example: Legacy single-calibration mode

```bash
python run_full_pipeline.py \
    --animals rory wilfred \
    --dataset-name calib_params/2026_01_31_22_50_54
```

## Output Files

### Step 1 outputs

- `pipeline_output/{animal}_all_trials.csv`: Combined trials for one animal
- `pipeline_output/trial_frames/{animal}_{video_folder}.csv`: Per-session trial CSVs

### Step 2 outputs

- `JARVIS-HybridNet/projects/{project}/predictions/predictions3D/{animal}_{video_folder}/Predictions_3D_trial_*/data3D.csv`

## Troubleshooting

See `ISSUE_LOG.md` for known issues and workarounds:

- **Corrupted videos**: Some sessions have 0×0 resolution videos that prevent JARVIS from running
- **Missing calibration**: Sessions without same-day calibration use nearest-date fallback (logged in console)
- **Pretraining sessions**: Excluded by default; use `--include-pretraining` if needed

## Related Scripts

- `cbot_climb_log/extract_trial_frames.py`: Step 1 extraction
- `convert_calibration.py`: Converts OpenCV calibration to JARVIS format
- `plot_trajectory_on_frame.py`: Visualizes trajectories on video frames
- `batch_trajectory_on_frame.py`: Batch visualization for all trials

## Data Flow

```
Videos (video_root/{animal}/YYYY_MM_DD_HH_MM_SS/)
  ↓ [Step 1: extract_trial_frames.py]
Trial CSVs (pipeline_output/trial_frames/{animal}_{video_folder}.csv)
  ↓ [Step 2: predict_trials.py + per-date calibration]
3D Predictions (predictions3D/{animal}_{video_folder}/Predictions_3D_trial_*/data3D.csv)
  ↓ [Downstream analysis]
Trajectory metrics, visualizations, etc.
```
