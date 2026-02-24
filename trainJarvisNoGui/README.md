# Train JARVIS (No GUI)

Scripts to **train** and **run prediction** with [JARVIS-HybridNet](https://github.com/JARVIS-MoCap/JARVIS-HybridNet) from the command line, without the Streamlit GUI. This repo assumes JARVIS-HybridNet is available (e.g. as a sibling directory or on `PYTHONPATH`).

---

## Setup

- **Python env:** Use the same conda/env as JARVIS (e.g. `conda activate jarvis`).
- **JARVIS on path:** When running scripts from this repo, point Python to JARVIS-HybridNet so `jarvis` can be imported:

  ```bash
  export PYTHONPATH=/path/to/JARVIS-HybridNet:$PYTHONPATH
  ```

  Example if both repos are under `src`:

  ```bash
  export PYTHONPATH=/home/user/src/JARVIS-HybridNet:$PYTHONPATH
  ```

- **Projects and config:** Project configs and weights live under `JARVIS-HybridNet/projects/<project_name>/`. Create/configure projects there (e.g. via the JARVIS GUI once, or by copying an existing project).

---

## Training

### Script: `train.py`

Trains CenterDetect, KeypointDetect, and/or HybridNet. You can skip stages and control pretrained weights.

**Usage:**

```bash
python train.py --project PROJECT_NAME [options]
```

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--project` | fin1 | Project name (must exist under JARVIS-HybridNet/projects/). |
| `--epochs-center` | 1 | Epochs for CenterDetect. |
| `--epochs-keypoint` | 1 | Epochs for KeypointDetect. |
| `--epochs-3d` | 1 | Epochs for HybridNet (3D). |
| `--epochs-finetune` | 1 | Epochs for HybridNet finetune (if used). |
| `--pretrain-center` | None | CenterDetect: path to `.pth` or `None`. |
| `--pretrain-keypoint` | None | KeypointDetect: path to `.pth` or `None`. |
| `--weights-keypoint` | latest | KeypointDetect weights for HybridNet: path or `latest`. |
| `--weights-hybridnet` | None | HybridNet initial weights: path or `latest` or `None`. |
| `--mode` | 3D_only | HybridNet: `3D_only` or `all`. |
| `--skip-center` | - | Skip CenterDetect. |
| `--skip-keypoint` | - | Skip KeypointDetect. |
| `--skip-3d` | - | Skip HybridNet. |

**Examples:**

- Train only HybridNet (3D-only, 100 epochs), latest KeypointDetect + latest HybridNet weights:

  ```bash
  python train.py --project mouse3dmodel --skip-center --skip-keypoint \
    --epochs-3d 100 --weights-keypoint latest --weights-hybridnet latest --mode 3D_only
  ```

- Full pipeline with pretrained Center and Keypoint from another project:

  ```bash
  python train.py --project mouse3dmodel \
    --epochs-center 10 --epochs-keypoint 10 --epochs-3d 50 \
    --pretrain-center /path/to/CenterDetect/EfficientTrack-medium_final.pth \
    --pretrain-keypoint /path/to/KeypointDetect/EfficientTrack-medium_final.pth
  ```

**GPU:**

- **Single GPU (recommended if you have one faster GPU):**  
  Set `CUDA_VISIBLE_DEVICES` to the device index (e.g. `0` or `1`):

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py --project mouse3dmodel ...
  ```

- **Multi-GPU:**  
  If multiple GPUs are visible, training uses DataParallel. Note: speed is limited by the slowest GPU.

---

## 3D prediction

### Prediction only on certain frames

You can get 3D pose **only for the frames you care about** in three ways:

1. **Single contiguous range** (e.g. frames 1000–2000)  
   Use **`predict.py`** with `--frame-start` and `--number-frames`:

   ```bash
   python predict.py --project mouseClimb4 --recording-path /path/to/videos \
     --dataset-name /path/to/calib_params \
     --frame-start 1000 --number-frames 1001
   ```
   This runs the model only on frames 1000 through 2000 (inclusive) and writes one folder with `data3D.csv` and `info.yaml`.

2. **Multiple ranges (e.g. one per trial)**  
   Use **`predict_trials.py`** with a CSV that has `frame_id_start` and `frame_id_end` (and optionally `trial_index`) per row. Prediction runs only on those ranges and saves **one folder per row** (e.g. per trial). See **Per-trial prediction** below.

3. **Specific frame numbers (non-contiguous)**  
   Use **`predict_trials.py`** with a CSV where each row is a single frame: set `frame_id_start` and `frame_id_end` to the same value (e.g. 100, 500, 1000). You get one prediction folder per frame; each `data3D.csv` has one row.

**Output details (what you get):**

- **`data3D.csv`**  
  One row per predicted frame. Columns: for each keypoint, four values `x, y, z, confidence` (repeated for all keypoints in project order). Row order = frame order (row 1 = first frame in the range, etc.). There is no frame-index column; use `info.yaml` to know the range.

- **`info.yaml`**  
  - `recording_path`: video folder  
  - `dataset_name`: calibration used  
  - `frame_start`: global first frame index in the video  
  - `number_frames`: number of frames processed  

  So **global frame index** for row `i` (0-based) = `frame_start + i`.

---

### 1. Full video (or frame range): `predict.py`

Runs 3D pose prediction on a recording for a contiguous frame range. Output: one folder per run with `data3D.csv` and `info.yaml`.

**Usage:**

```bash
python predict.py --project PROJECT --recording-path /path/to/video_folder \
  --dataset-name /path/to/calib_params [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--project` | none | Project name. |
| `--recording-path` | - | Folder containing camera videos. |
| `--dataset-name` | - | Calibration params folder (e.g. `.../calib_params/2026_02_01_14_10_44`). |
| `--weights-center` | latest | CenterDetect weights. |
| `--weights-hybridnet` | latest | HybridNet weights. |
| `--frame-start` | 3800 | First frame index. |
| `--number-frames` | -1 | Number of frames (-1 = all from `frame-start`). |
| `--trt` | off | TensorRT: off, new, previous. |

**Example:**

```bash
PYTHONPATH=/path/to/JARVIS-HybridNet:$PYTHONPATH python predict.py \
  --project mouseClimb4 \
  --recording-path /mnt/ssd2/mickey/2026_01_01_15_18_31 \
  --dataset-name /home/user/red_data/climb_jarvis_merge_all/calib_params/2026_02_01_14_10_44
```

Output is written under  
`JARVIS-HybridNet/projects/<project>/predictions/predictions3D/Predictions_3D_<timestamp>/`.

---

### 2. Per-trial prediction: `predict_trials.py`

Runs 3D prediction **only for the frame ranges** defined in a CSV (e.g. one row per trial). Saves **one prediction folder per trial**.

**Trials CSV:** Must have at least:

- `frame_id_start` – first frame index of the trial  
- `frame_id_end` – last frame index (inclusive)  
- Optional: `trial_index` – used in output folder names (defaults to row index)

**Usage:**

```bash
python predict_trials.py --project PROJECT --recording-path /path/to/videos \
  --dataset-name /path/to/calib_params --trials-csv trial_frames.csv [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--project` | - | Project name. |
| `--recording-path` | - | Folder with camera videos. |
| `--dataset-name` | - | Calibration params folder. |
| `--trials-csv` | - | CSV with frame_id_start, frame_id_end [, trial_index]. |
| `--weights-center` | latest | CenterDetect weights. |
| `--weights-hybridnet` | latest | HybridNet weights. |
| `--start-col` | frame_id_start | Column name for start frame. |
| `--end-col` | frame_id_end | Column name for end frame. |
| `--trial-index-col` | trial_index | Column name for trial index. |

**Example:**

```bash
PYTHONPATH=/path/to/JARVIS-HybridNet:$PYTHONPATH python predict_trials.py \
  --project mouseClimb4 \
  --recording-path /mnt/ssd2/mickey/2026_01_01_15_18_31 \
  --dataset-name /home/user/red_data/climb_jarvis_merge_all/calib_params/2026_02_01_14_10_44 \
  --trials-csv trial_frames.csv
```

Output folders (one per trial):  
`JARVIS-HybridNet/projects/<project>/predictions/predictions3D/Predictions_3D_trial_0000_11572-20491/`,  
`Predictions_3D_trial_0001_28678-31180/`, etc.  
Each contains `data3D.csv` and `info.yaml` (with `frame_start`, `number_frames`, `recording_path`, `dataset_name`).

---

## 2D prediction

### Script: `predict2D.py`

Runs 2D keypoint prediction (CenterDetect + KeypointDetect only), no 3D. Writes one CSV per run with 2D keypoints per camera and frame.

**Usage:**

```bash
python predict2D.py --project PROJECT --recording-path /path/to/videos [options]
```

**Options:** `--weights-center`, `--weights-keypoint`, `--frame-start`, `--number-frames`, `--trt`, `--cameras` (space-separated list).  
Output is written under the recording path, e.g.  
`<recording_path>/predictions/jarvis/<project>/Predictions_2D_<timestamp>/data2D.csv`.

---

## Creating annotated videos from predictions

### 1. Single prediction run: `createvideos.py`

Builds annotated 3D videos from one prediction folder (e.g. from `predict.py`). Reads `info.yaml` in that folder for recording path and dataset.

**Usage:**

```bash
python createvideos.py --project PROJECT --prediction-dir FOLDER_NAME [options]
```

- `FOLDER_NAME`: name of the prediction folder, e.g. `Predictions_3D_20260202-130925` or `Predictions_3D_trial_0000_11572-20491`.
- Script looks for the folder under `JARVIS-HybridNet/projects/<project>/predictions/predictions3D/` (and, if missing, under `trainJarvisNoGui/../JARVIS-HybridNet/...`).

**Options:** `--cams` (comma-separated), `--frame-start`, `--number-frames`.

**Example:**

```bash
PYTHONPATH=/path/to/JARVIS-HybridNet:$PYTHONPATH python createvideos.py \
  --project mouseClimb4 --prediction-dir Predictions_3D_20260202-130925
```

---

### 2. Trial-based (multiple clips): `createvideos_new.py`

Renders annotated videos for **each trial** in a trials CSV. Expects one prediction folder that contains 3D predictions for the same frame ranges as the CSV (e.g. from `predict_trials.py` you run this once per trial folder, or use a single combined prediction if you have one).

**Usage:**

```bash
python createvideos_new.py --project-name PROJECT --prediction-dir /full/path/to/Predictions_3D_* \
  --trials-csv /path/to/trials.csv [options]
```

- `--prediction-dir`: **full path** to the prediction folder (e.g. containing `data3D.csv`, `info.yaml`).
- `--trials-csv`: CSV with columns `trial_start` and `trial_end` (or `--start-col` / `--end-col`).

**Options:** `--start-col`, `--end-col`, `--cams`, `--min-conf`, `--clip-prefix`.

**Example (for one trial folder from predict_trials):**

```bash
# Build a one-row trials CSV for that trial’s frame range, then:
PYTHONPATH=/path/to/JARVIS-HybridNet:$PYTHONPATH python createvideos_new.py \
  --project-name mouseClimb4 \
  --prediction-dir /path/to/JARVIS-HybridNet/projects/mouseClimb4/predictions/predictions3D/Predictions_3D_trial_0000_11572-20491 \
  --trials-csv trials_one_trial.csv
```

Where `trials_one_trial.csv` has columns e.g. `trial_start`,`trial_end` with one row `11572,20491`.

---

## Summary of scripts

| Script | Purpose |
|--------|--------|
| `train.py` | Train CenterDetect / KeypointDetect / HybridNet (CLI, no GUI). |
| `predict.py` | 3D prediction on a full video or contiguous frame range. |
| `predict_trials.py` | 3D prediction per trial from a CSV; one output folder per trial. |
| `predict2D.py` | 2D keypoint prediction only. |
| `createvideos.py` | Annotated 3D videos from one prediction folder. |
| `createvideos_new.py` | Annotated 3D videos per trial from a prediction folder + trials CSV. |
| `jarvis_predict3D_new.py` | Alternative trial-wise 3D prediction (different I/O layout). |

---

## Changes and fixes applied (in this repo / JARVIS-HybridNet)

- **train.py**
  - When `--pretrain-keypoint` is set, HybridNet training uses that path for KeypointDetect weights instead of `--weights-keypoint`/latest.
- **createvideos.py**
  - If the prediction folder is not under `trainJarvisNoGui/../projects/`, it is looked up under `trainJarvisNoGui/../JARVIS-HybridNet/projects/`.
- **JARVIS-HybridNet (if you applied the same edits)**
  - **HybridNet load_weights:** Checkpoints that saved meshgrid buffers (`xx`, `yy`, `zz`) can cause “shared memory” errors when loading. The loader now skips loading those buffers and uses `strict=False` so the rest of the weights load correctly.
  - **Multi-GPU:** CenterDetect, KeypointDetect, and HybridNet training use `DataParallel` when multiple GPUs are visible.
  - **Single GPU:** If `CUDA_VISIBLE_DEVICES` is set to a single device (e.g. `0` or `1`), training uses that GPU only and does not wrap the model in DataParallel.
- **predict3D (JARVIS):**
  - If `params.output_dir` is already set (e.g. by `predict_trials.py`), it is not overwritten, so per-trial output directories work as intended.

---

## Quick reference: typical workflow

1. **Train** (e.g. HybridNet only, 100 epochs, latest weights):

   ```bash
   export CUDA_VISIBLE_DEVICES=0   # optional: force one GPU
   export PYTHONPATH=/path/to/JARVIS-HybridNet:$PYTHONPATH
   python train.py --project mouse3dmodel --skip-center --skip-keypoint \
     --epochs-3d 100 --weights-keypoint latest --weights-hybridnet latest --mode 3D_only
   ```

2. **3D predict** (full video or per-trial):

   ```bash
   # Full video
   python predict.py --project mouseClimb4 --recording-path /path/to/videos \
     --dataset-name /path/to/calib_params

   # Per trial (from trial_frames.csv)
   python predict_trials.py --project mouseClimb4 --recording-path /path/to/videos \
     --dataset-name /path/to/calib_params --trials-csv trial_frames.csv
   ```

3. **Videos from predictions:**

   ```bash
   python createvideos.py --project mouseClimb4 --prediction-dir Predictions_3D_20260202-130925
   ```

All of the above assume you are in the `trainJarvisNoGui` directory and that `PYTHONPATH` includes the path to JARVIS-HybridNet when required.

