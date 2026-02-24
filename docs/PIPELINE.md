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

**Wrapper script**: `trainJarvisNoGui/predict_trials.py`  
**Core JARVIS function**: `JARVIS-HybridNet/jarvis/prediction/predict3D.py`

Step 2 runs JARVIS 3D pose prediction for each trial, generating 3D coordinates for keypoints (Snout, EarL, EarR, Tail, etc.).

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

### Trial-wise prediction wrapper (`predict_trials.py`)

`predict_trials.py` is a thin wrapper that:

- Reads a **per-session trial CSV** (from step 1) with at least:
  - `frame_id_start`, `frame_id_end`
  - Optional `trial_index` (if missing, indices `0..N-1` are assigned)
- For each row / trial:
  - Computes `num_frames = frame_id_end - frame_id_start + 1`
  - Builds a **session base directory** for predictions:
    ```text
    JARVIS-HybridNet/projects/{project}/predictions/predictions3D/{animal}_{video_folder}/
    ```
  - Builds a **per-trial subfolder**:
    ```text
    Predictions_3D_trial_{trial_index:04d}_{frame_start}-{frame_end}/
    ```
  - Sets `Predict3DParams.output_dir` to that per-trial folder and calls `predict3D(params)`.
- Prints progress per trial:
  ```text
  [i/N] Trial {trial_index}: frames start-end (num_frames) -> Predictions_3D_trial_...
  ```

This is the script `run_full_pipeline.py` calls for each session.

### JARVIS core (`predict3D.py`)

The actual 3D prediction work is done by `JARVIS-HybridNet/jarvis/prediction/predict3D.py`:

- Loads the JARVIS project and model weights (CenterDetect + HybridNet).
- Builds the reprojection tool from the chosen calibration (`dataset_name`).
- **Output directory behaviour**:
  - If `Predict3DParams.output_dir` is already set (e.g. by `predict_trials.py`), it is **respected**.
  - If it is not set (e.g. when calling `predict3D` directly), it falls back to a timestamp folder:
    ```text
    JARVIS-HybridNet/projects/{project}/predictions/predictions3D/Predictions_3D_{YYYYMMDD-HHMMSS}/
    ```
- Writes:
  - `data3D.csv` — 3D coordinates for all keypoints and frames in the trial.
  - `info.yaml` — metadata: project, recording path, calibration dataset, frame range, etc.
  - Additional visualization files if enabled.

### Output structure

When run via `predict_trials.py` (as in the full pipeline), predictions are saved under:
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

### Step 2 resume behaviour

`run_full_pipeline.py` is designed so you can safely restart step 2 after an error or interrupt:

- For each session CSV:
  - Counts **valid trials** in the CSV (rows with `frame_count > 0` or a non-empty `frame_id_start`).
  - Counts existing `Predictions_3D_trial_*` folders under:
    ```text
    JARVIS-HybridNet/projects/{project}/predictions/predictions3D/{animal}_{video_folder}/
    ```
- **Skip rule**:
  - If `existing_trials >= total_valid_trials`, the session is **skipped**:
    ```text
    [JARVIS] [wilfred 5/9] wilfred_2026_01_04_16_35_15 (calib=...) — already has 6/6 trials, skipping
    ```
  - Otherwise the session is re-run, and all its trials are predicted again.

This makes it safe to rerun:

```bash
python run_full_pipeline.py --step 2 --animals rory wilfred --skip-extract
```

after a crash; completed sessions are not recomputed.

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
- `batch_trajectory_on_frame.py`: Batch extract frame + plot trajectory for all JARVIS predictions (rory/wilfred)

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

## Running on existing JARVIS predictions (predictions3D)

When 3D predictions already exist (e.g. under JARVIS-HybridNet), you can run frame extraction, trajectory-on-frame plotting, and then full trajectory analysis without re-running JARVIS.

### Paths and layout

- **Predictions root**: `/home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D`
- **Session/trial layout**: `predictions3D/{animal}_{video_folder}/Predictions_3D_trial_XXXX_frameStart-frameEnd/`  
  Example: `rory_2025_12_23_16_57_09/Predictions_3D_trial_0000_15186-145628`. The trial start frame index is **frameStart** (e.g. 15186) from the folder name.
- **Videos**: `/mnt/mouse2/{animal}/{video_folder}/` (e.g. `/mnt/mouse2/rory/2025_12_23_16_57_09/`). Videos must be available at this path for frame extraction.

### Step 1: Batch extract frame + plot trajectory

**Script**: `batch_trajectory_on_frame.py`

For each trial (rory and wilfred by default) the script:

1. Extracts the trial start frame from the camera video at `video_root/{animal}/{video_folder}/{camera}.mp4` and saves it as **`frame.png`** inside the trial folder.
2. Runs `plot_trajectory_on_frame.py` to overlay the filtered Snout trajectory on that frame and writes **`trajectory_on_frame.png`** and **`trajectory_filtered.csv`** in the same trial folder.

**Outputs per trial** (all inside the trial folder, e.g. `.../Predictions_3D_trial_0000_15186-145628/`):

| File | Description |
|------|-------------|
| `frame.png` | Extracted video frame (trial start frame). |
| `trajectory_on_frame.png` | Frame with Snout trajectory overlaid (colored by frame number). |
| `trajectory_filtered.csv` | Filtered trajectory points (frame_number, x, y, z, u, v, segment_id) for analysis. Elevation and region filters apply (see **Trajectory filters** below). |

**Commands** (run from the `analyzeMiceTrajectory` directory):

```bash
# Process all trials (rory and wilfred). Requires /mnt/mouse2 mounted.
python batch_trajectory_on_frame.py \
  --predictions-root /home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D \
  --video-root /mnt/mouse2 \
  --animals rory wilfred \
  --camera Cam2005325
```

- **Skip trials that already have** `trajectory_filtered.csv`: add `--skip-existing`.
- **List trials only** (no extraction or plotting): add `--dry-run`.
- **Different camera**: set `--camera` (e.g. `Cam2005326`).

This workflow has been run successfully on the full JARVIS predictions set (400 trials for rory and wilfred).

### Step 2: Run trajectory analysis

After the batch has written `trajectory_filtered.csv` for the trials you care about, run the analysis on the same predictions root. `analyze_trajectories.py` supports the nested layout (session folders under the root).

```bash
python analyze_trajectories.py \
  /home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D \
  -o trajectory_analysis
```

This produces summary CSVs and plots in `trajectory_analysis/` (see main pipeline docs and `analyze_trajectories.py --help`).

### Trajectory filters

Before plotting or analysis, trajectory points are filtered so that: **z** is in [0, 150]; and any point with **u < 1250** (px) and **z > 50** is dropped. See **`docs/TRAJECTORY_FILTERS.md`** for the full list and rationale.

### Trajectory plots: u-v (camera view) everywhere

All trajectory path plots use **(u, v)** from `trajectory_filtered.csv` when available — the same 2D camera/image coordinates used for the frame overlay — so orientation matches the arena view. The v-axis is inverted so the plot matches the frame (v=0 at top). This applies to:

- **`plot_session_trajectories.py`**: per-session plots (default; use `--use-world-xy` for 3D world x-y).
- **`analyze_trajectories.py`**: example trajectories, all-trials, by animal×phase, vertical-on-left/right, etc.

Flow field and time-spent heatmaps remain in world (x, y) for spatial binning.

### Trajectory analysis output (sessions + phases)

The trajectory plotting scripts write under **`trajectory_analysis/{animal}/sessions/`** and **`trajectory_analysis/{animal}/phases/`**. For the full folder layout, file descriptions, and commands, see **`docs/TRAJECTORY_ANALYSIS_OUTPUT.md`**.

### Per-session trajectory plots

**Script**: `plot_session_trajectories.py`

Writes **one subfolder per session** under `trajectory_analysis/{animal}/sessions/`, with a single plot per session: trajectory (u, v) from `trajectory_filtered.csv` with **z as color** for all trials in that session. No aggregate stats or cross-session plots.

**Output layout:**

```
trajectory_analysis/
  rory/
    sessions/
      rory_2025_12_23_16_57_09/
        trajectory_xy_z.png    # all trials in this session, u-v colored by z
      ...
    phases/
      early/
      mid/
      late/
  wilfred/
    sessions/
      wilfred_2026_01_08_16_05_24/
        trajectory_xy_z.png
      ...
    phases/
      early/
      mid/
      late/
```

**Commands:**

```bash
python plot_session_trajectories.py \
  --predictions-root /home/user/src/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D \
  -o trajectory_analysis
```

- **Default:** We plot **(u, v)** — the same 2D camera/image coordinates used when drawing the trajectory on the video frame — so the orientation matches the arena view. The v-axis is inverted so the plot matches the frame (v=0 at top). Use **`--use-world-xy`** to plot 3D world (x, y) instead.
- Restrict to specific animals: `--animals rory wilfred`.

### Phase aggregates (early / mid / late)

**Script**: `plot_phase_trajectories.py`

Creates **early**, **mid**, and **late** subfolders under `trajectory_analysis/{animal}/phases/` and writes one aggregated trajectory plot per phase. When **`trial_types.csv`** is present (from `cbot_climb_log/export_trial_types_for_trajectories.py`), each session and phase also gets **vertical_left** and **vertical_right** subfolders with trajectory plots for those trial types (from robot_manager logs: left_angle_deg/right_angle_deg, 360 = vertical).

```bash
python plot_phase_trajectories.py \
  --predictions-root /path/to/predictions3D \
  -o trajectory_analysis
```

Options: `--use-world-xy`, `--animals rory wilfred`, `--trial-types <path>`.

To generate trial types from logs (run before or alongside the plotters):  
`python cbot_climb_log/export_trial_types_for_trajectories.py --predictions-dir /path/to/predictions3D -o trajectory_analysis/trial_types.csv`
