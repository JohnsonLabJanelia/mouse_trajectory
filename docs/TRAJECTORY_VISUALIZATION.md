# Trajectory Visualization Scripts

Documentation for scripts that visualize 3D trajectory data from JARVIS predictions.

## Overview

After running JARVIS 3D prediction (Step 2 of the pipeline), you get `data3D.csv` files with 3D coordinates for each keypoint (Snout, EarL, EarR, Tail, etc.) per frame. These scripts help visualize and analyze those trajectories.

## Scripts

### 1. `plot_trajectory_xy.py`

**Location**: `predictions3D/plot_trajectory_xy.py`

**Purpose**: Plots 3D trajectory projected onto the **x-y plane** (top-down view).

**What it does**:
- Parses `data3D.csv` with multi-level header (body part names, then coordinate labels)
- Extracts per-body-part DataFrames (Snout, EarL, EarR, Tail, etc.)
- Projects 3D (x, y, z) → 2D (x, y) by simply using x and y coordinates
- Plots trajectory as colored line/scatter, optionally colored by frame index (time)

**Usage**:
```bash
# Plot all body parts from a trial folder
python predictions3D/plot_trajectory_xy.py Predictions_3D_trial_0000_11572-20491/

# Plot only Snout
python predictions3D/plot_trajectory_xy.py Predictions_3D_trial_0000_11572-20491/ --parts Snout

# Plot with confidence filtering
python predictions3D/plot_trajectory_xy.py Predictions_3D_trial_0000_11572-20491/ --min-confidence 0.5

# Output to specific file
python predictions3D/plot_trajectory_xy.py Predictions_3D_trial_0000_11572-20491/ -o my_trajectory.png
```

**Output**: Saves `trajectory_xy.png` in the trial folder (or specified path).

---

### 2. `plot_trajectory_on_frame.py` ✅ **Current**

**Location**: `plot_trajectory_on_frame.py` (project root)

**Purpose**: Overlays the **Snout** 2D trajectory on a video frame image. Uses camera calibration to project 3D points into this camera’s image, then applies a fixed pipeline of trims and filters so the plotted path starts in the start region, ends in the end region, and avoids jumps and stray segments.

---

#### Pipeline (order is fixed)

| Step | What it does |
|------|------------------|
| 1 | Load Snout (x, y, z, confidence) from `trial_dir/data3D.csv`. |
| 2 | **Frame numbers**: From trial folder name `..._frameStart-frameEnd`, set `frame_start`; per-row frame = `frame_start + row_index`. Carried through all later steps so the colorbar is actual frame number. |
| 3 | **Project 3D → 2D**: JARVIS formula (intrinsicMatrix, distortionCoefficients, R, T) → image (u, v) in pixels. |
| 4 | **Start trim** (if `arena_start_end.npz`): Drop all points before the **first** point inside the start circle. |
| 5 | **End trim** (if start/end loaded): Drop all points after the **last** point inside the end circle. |
| 6 | **Confidence**: Keep points with `confidence >= min_confidence` (default 0.15). |
| 7 | **Arena mask**: Keep only points where `predictions3D/arena_mask.npz` mask is non-zero (and in image bounds). |
| 8 | **Jump filter**: Consecutive 2D distance > `jump_threshold` px (default 100) splits into segments. Drop segments with fewer than `min_segment_points` (default 50). |
| 9 | **Segment filter**: Keep only segments that have at least one point in the start circle **or** in the end circle. |
| 10 | **Plot**: Matplotlib `imshow` of frame + `scatter(u, v, c=frame_numbers)` + yellow line per segment. Colorbar = "Frame number". Save PNG. |
| (optional) | **Export trajectory**: If `--output-trajectory CSV` is set, save the **filtered** points to CSV for analysis (same points as plotted). |

If any step removes all points/segments, the script saves the frame image only (no trajectory) and exits.

---

#### Inputs

- **Trial folder**: e.g. `predictions3D/Predictions_3D_trial_0000_11572-20491` containing:
  - `data3D.csv` (Snout and other keypoints)
  - `info.yaml` (optional: `dataset_name` for calibration path, `recording_path` for video)
  - `frame.png` (background image; default)
- **Camera**: e.g. `Cam2005325` (required).
- **Calibration**: JARVIS-format OpenCV YAML for this camera. Default: `info.yaml` → `dataset_name` → `{dataset_name}/{camera}.yaml`.
- **Arena mask** (optional): `predictions3D/arena_mask.npz` with key `mask` (H×W uint8). Same (H,W) as frame.
- **Start/end** (optional): `predictions3D/arena_start_end.npz` with `average_start`, `average_end` (2D center in image coords), `radius_start`, `radius_end` (pixels). Used for trim and segment filter.

---

#### Key functions in the script

| Function | Role |
|----------|------|
| `get_trial_frame_and_recording(trial_dir)` | Parse `frame_start` from folder name `_(\d+)-(\d+)$` and `recording_path` from `info.yaml`. |
| `load_arena_mask(path)`, `load_arena_start_end(path)` | Load mask or start/end circles from .npz. |
| `point_in_start_region(u, v, start_end)` | Boolean array: point inside start circle. |
| `point_in_end_region(u, v, start_end)` | Boolean array: point inside end circle. |
| `segment_has_points_in_start_or_end_region(u_seg, v_seg, start_end)` | True if segment has any point in start or end circle. |
| `load_calib(path)` | Load JARVIS calibration (OpenCV FileStorage). |
| `project_3d_to_2d(points_3d, calib)` | JARVIS reprojection: world → image (u, v). |
| `get_continuous_segments(u, v, frame_indices, jump_threshold_px)` | Split at jumps; return list of `(u_seg, v_seg, frame_indices_seg)`. |
| `run(...)` | Runs the full pipeline; `main()` parses CLI and calls `run`. |

---

#### Usage

```bash
# Default: frame from trial_dir/frame.png, calibration from info.yaml, arena mask and start/end from predictions3D/
python plot_trajectory_on_frame.py predictions3D/Predictions_3D_trial_0000_11572-20491 --camera Cam2005325

# Save filtered trajectory to CSV for analysis (same points as the plot)
python plot_trajectory_on_frame.py predictions3D/Predictions_3D_trial_0000_11572-20491 --camera Cam2005325 --output-trajectory predictions3D/Predictions_3D_trial_0000_11572-20491/trajectory_filtered.csv

# Custom output path (e.g. avoid overwriting ground truth)
python plot_trajectory_on_frame.py predictions3D/Predictions_3D_trial_0001_28678-31180 --camera Cam2005325 -o predictions3D/Predictions_3D_trial_0001_28678-31180/trajectory_on_frame_analysis.png

# Disable arena or start/end filtering
python plot_trajectory_on_frame.py predictions3D/Predictions_3D_trial_0000_11572-20491 --camera Cam2005325 --no-arena-mask --no-arena-start-end

# Tune jump and segment length
python plot_trajectory_on_frame.py predictions3D/Predictions_3D_trial_0000_11572-20491 --camera Cam2005325 --jump-threshold 80 --min-segment-points 30
```

**CLI options** (summary):

- `trial_dir` (positional), `--camera` (required)
- `--calib-path`, `--frame-path`, `-o` (output image)
- `--output-trajectory` path: save filtered trajectory CSV (frame_number, x, y, z, u, v, segment_id) for analysis
- `--arena-mask`, `--no-arena-mask`
- `--arena-start-end`, `--no-arena-start-end`
- `--jump-threshold` (default 100 px), `--min-segment-points` (default 50), `--min-confidence` (default 0.15)

**Output**:
- **Image**: `trial_dir/trajectory_on_frame.png` (or `-o`): frame with Snout trajectory overlaid, color = actual frame number.
- **Trajectory CSV** (if `--output-trajectory` is set): one row per filtered point. Columns: `frame_number`, `x`, `y`, `z` (3D world), `u`, `v` (2D image for this camera), `segment_id` (0, 1, …). This is the **analysis-ready** trajectory (start/end trimmed, confidence, arena, jump and segment filtered), not the raw `data3D.csv`.

**Status**: ✅ Implemented and documented; can be extended (e.g. other body parts, different colormaps, or exporting filtered points).

**Rebuilding / extending**:
- Pipeline order is fixed in `run()`: trim start/end first, then confidence, arena mask, then segment by jumps and filter segments. Changing order will change which points remain.
- Frame numbers come from `get_trial_frame_and_recording(csv_path.parent)`; the same index array is sliced at every step so the colorbar always shows actual video frame indices.
- To add a new filter: apply it in `run()` at the right place and slice `u`, `v`, and `frame_indices` (and any new arrays) together.
- Calibration and projection: `load_calib()` + `project_3d_to_2d()` match JARVIS `ReprojectionTool` (see JARVIS-HybridNet if the formula must be re-derived).

---

### 3. `analyze_trajectories.py` — Study filtered trajectories

**Location**: `analyze_trajectories.py` (project root)

**Purpose**: Read all `trajectory_filtered.csv` files (from `plot_trajectory_on_frame.py --output-trajectory`), compute per-trial statistics, and produce summary tables and plots for exploration.

**What it does**:
1. **Finds trials**: Scans `predictions_dir` for `Predictions_3D_trial_*` folders containing `trajectory_filtered.csv`.
2. **Per-trial stats**: For each CSV, computes:
   - `n_points`, `frame_start`, `frame_end`, `duration_frames`
   - `path_length_3d` (sum of consecutive 3D Euclidean distances)
   - `mean_speed_per_step`, `n_segments`, `min/max_segment_points`
3. **Summary CSV**: Writes `trajectory_analysis/trajectory_stats_summary.csv` (or `-o` dir).
4. **Plots** (all saved under output dir):
   - `path_length_per_trial.png` — bar chart of 3D path length per trial
   - `duration_per_trial.png` — bar chart of frame span per trial
   - `path_length_vs_duration.png` — scatter
   - `n_segments_per_trial.png` — bar chart of segment count
   - `path_length_histogram.png` — distribution of path lengths
   - `example_trajectories_xy.png` — top-down (x, y) view of first 6 trials
   - `summary_boxplots.png` — boxplots of path length, duration, n_points

**Usage**:
```bash
# Default: predictions3D, output to trajectory_analysis/
python analyze_trajectories.py

# Custom paths
python analyze_trajectories.py /path/to/predictions3D -o /path/to/trajectory_analysis
```

**Requires**: `pandas`, `matplotlib`, `numpy`. Input: `trajectory_filtered.csv` in each trial folder (generate with `plot_trajectory_on_frame.py --output-trajectory`).

---

### 4. `extract_first_frame.py` ✅ **Restored**

**Location**: `/home/user/src/analyzeMiceTrajectory/extract_first_frame.py`

**Purpose**: Extracts a single frame from a video file, typically used to get the first frame of a trial for visualization.

**What it does**:
1. **Two modes**:
   - **Direct video mode**: Extract frame 0 (or specified frame) from a video file
   - **Trial folder mode**: Parse `frame_start` from trial folder name (e.g., `Predictions_3D_trial_0000_11572-20491` → frame 11572) and read `recording_path` from `info.yaml` to find the video
2. **Extracts frame**: Uses `ffmpeg` to extract the frame: `ffmpeg -i video.mp4 -vf "select=eq(n\,FRAME)" -vframes 1 output.png`
3. **Saves**: Writes frame as PNG image

**Expected usage** (based on bytecode):
```bash
# Extract frame 0 from a video (original behavior)
python extract_first_frame.py video.mp4 [output.png]

# Extract the trial start frame from a prediction folder
python extract_first_frame.py Predictions_3D_trial_0000_11572-20491/ --camera Cam2005325 [output.png]
```

**Key functions**:
- `parse_trial_frame_and_recording()`: Parse `frame_start` from folder name and read `recording_path` from `info.yaml`
- `extract_frame()`: Uses `ffmpeg` to extract a single frame from video

**Status**: ✅ **Restored** — source file recreated from bytecode analysis.

---

### 5. `batch_trajectory_on_frame.py` ⚠️ **Missing source file**

**Location**: Should be at `/home/user/src/analyzeMiceTrajectory/batch_trajectory_on_frame.py` (currently only `.pyc` bytecode exists)

**Purpose**: Batch processes multiple trials, running `plot_trajectory_on_frame.py` for each trial folder.

**Expected usage**:
```bash
# Process all trials in a predictions3D directory
python batch_trajectory_on_frame.py predictions3D/

# Process trials for a specific animal/session
python batch_trajectory_on_frame.py --animal mickey --session 2026_01_01_15_18_31
```

**Status**: ⚠️ **Source file missing** — only `.pyc` bytecode exists.

---

## Data Format

### `data3D.csv` structure

```
Snout,Snout,Snout,Snout,EarL,EarL,EarL,EarL,...
x,y,z,confidence,x,y,z,confidence,...
-13.23,-215.77,42.84,0.172,-22.86,-200.98,32.75,0.106,...
-13.98,-212.94,41.85,0.173,-20.99,-202.45,35.73,0.105,...
...
```

- **Row 1**: Body part names (repeated 4× per part: x, y, z, confidence)
- **Row 2**: Coordinate labels (`x`, `y`, `z`, `confidence`)
- **Row 3+**: Numeric data (one row per frame)

### `info.yaml` structure

```yaml
recording_path: /mnt/ssd2/mickey/2026_01_01_15_18_31
dataset_name: /home/user/red_data/climb_jarvis_merge_all/calib_params/2026_02_01_14_10_44
frame_start: 384780
number_frames: 3360
```

---

## Workflow

1. **Run JARVIS prediction** (Step 2 of pipeline) → generates `Predictions_3D_trial_*/data3D.csv`
2. **Extract first frame** (optional): Use `extract_first_frame.py` to extract the trial start frame from video
3. **Plot x-y trajectory**: Use `plot_trajectory_xy.py` for top-down 2D view
4. **Overlay on video frame**: Use `plot_trajectory_on_frame.py` to see trajectory on actual camera view (requires calibration and frame image)

---

## Missing Scripts

✅ **Status**: `plot_trajectory_on_frame.py` and `extract_first_frame.py` have been **restored** by reconstructing them from bytecode analysis, function signatures, and extracted strings.

⚠️ **Action needed**: `batch_trajectory_on_frame.py` is still missing. It likely batch processes multiple trials by calling `plot_trajectory_on_frame.py` for each trial folder. You can create it or use a simple shell loop:

```bash
for trial_dir in predictions3D/Predictions_3D_trial_*/; do
    python plot_trajectory_on_frame.py "$trial_dir" --camera Cam2005325
done
```
