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

### 2. `plot_trajectory_on_frame.py` ✅ **Restored**

**Location**: `/home/user/src/analyzeMiceTrajectory/plot_trajectory_on_frame.py`

**Purpose**: Overlays 2D snout trajectory on a **video frame image** by projecting 3D points into camera image space using calibration.

**What it does** (inferred from bytecode and outputs):
1. **Loads calibration**: Reads JARVIS-format camera calibration YAML (intrinsicMatrix, distortionCoefficients, R, T)
2. **Reads first frame**: Extracts a frame image from the video (likely frame 0 or `frame_start` from `info.yaml`)
3. **Projects 3D to 2D**: Uses camera calibration to project 3D snout (x,y,z) coordinates into image (u,v) pixel coordinates
   - Uses JARVIS ReprojectionTool formula: `P = [R; T] @ K`, then projects `(X,Y,Z,1) @ P → (u*z, v*z, z)`, divides by z, applies radial distortion
4. **Filters trajectory**: 
   - Splits trajectory at large jumps (frame-to-frame steps above a threshold)
   - Keeps top-k longest continuous segments
   - Optionally applies arena mask filtering
5. **Overlays on frame**: Draws trajectory points/lines on the frame image, colored by frame index (time)
6. **Saves**: Writes `trajectory_on_frame.png` in the trial folder

**Key functions** (from bytecode):
- `get_continuous_segments()`: Split trajectory at jumps, return top-k longest segments
- `refine_segments_drop_jumps()`: Within each segment, drop outlier jumps
- `load_calib_opencv()`: Load calibration YAML with cv2.FileStorage
- `project_world_to_image_jarvis()`: Project 3D world points to 2D image coordinates
- `overlay_trajectory_on_frame()`: Main function that orchestrates the overlay

**Expected usage** (based on bytecode structure):
```bash
# Overlay trajectory on first frame for a trial
python plot_trajectory_on_frame.py Predictions_3D_trial_0000_11572-20491/

# Specify camera calibration and frame image
python plot_trajectory_on_frame.py \
    --trial-dir Predictions_3D_trial_0000_11572-20491/ \
    --calib-path calib_params/2026_01_01/Cam2005325.yaml \
    --frame-path /path/to/frame.png \
    --camera-name Cam2005325
```

**Output**: Saves `trajectory_on_frame.png` showing the video frame with colored trajectory overlaid.

**Status**: ⚠️ **Source file missing** — only `.pyc` bytecode exists. The script needs to be restored or recreated from the bytecode.

---

### 3. `extract_first_frame.py` ✅ **Restored**

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

### 4. `batch_trajectory_on_frame.py` ⚠️ **Missing source file**

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
