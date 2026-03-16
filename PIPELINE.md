# Dataset Generation Pipeline — Runbook

This document covers the full pipeline for generating the 24-keypoint 3D pose dataset
from raw climbing videos (rory + wilfred), including all known issues and how to run
it yourself from scratch.

---

## Overview

```
NAS videos (rory/wilfred)
        ↓  rsync to scratch (per cluster node)
make_dataset.py   ← JARVIS mouseHybrid24 (24 kp) + DLT triangulation
        ↓
dataset/predictions_raw/{animal}_{session}/trial_{idx}_{start}-{end}/data3D.csv
        ↓
filter_dataset.py  ← arena mask + confidence gating
        ↓
dataset/rory.csv  +  dataset/wilfred.csv
```

**Input:**
- Videos: `NAS (10.10.1.28):/mouse2/climb/{animal}/{video_folder}/Cam*.mp4`
- Trials:  `dataset/rory_trials.csv`, `dataset/wilfred_trials.csv`
- Calibration: `calib_params/{YYYY_MM_DD}/*.yaml`

**Output:**
- Per-trial 3D predictions: `dataset/predictions_raw/`
- Filtered dataset: `dataset/rory.csv`, `dataset/wilfred.csv`

---

## Prerequisites

### On the cluster

- Conda env `jarvis_repro` at `~/miniconda3/envs/jarvis_repro`
  - PyTorch + JARVIS-HybridNet installed
  - `ffmpeg` available at `/usr/bin/ffmpeg` (system)
- JARVIS project `mouseHybrid24` loaded at `~/JARVIS-HybridNet/projects/mouseHybrid24/`
  - Trained weights for CenterDetect + KeypointDetect present
- `~/analyzeMiceTrajectory/` is this repo
- `~/analyzeMiceTrajectory/dataset/{rory,wilfred}_trials.csv` exist
- `~/analyzeMiceTrajectory/cluster/sessions.txt` exists (one `animal/video_folder` per line)
- `~/analyzeMiceTrajectory/calib_params/` contains dated calibration dirs

### sessions.txt

26 sessions total (17 rory, 9 wilfred), one per line, format `animal/video_folder`:

```
rory/2025_12_23_16_57_09
rory/2025_12_24_13_29_06
...
wilfred/2026_01_08_16_05_24
```

The LSF array index maps to the line number (1-indexed).

---

## Step 1 — Submit the array job

```bash
cd ~/analyzeMiceTrajectory
bsub < cluster/run_session.lsf
```

This submits a 26-element array job (`dataset[1-26]`), one element per session.
Each job:
1. `rsync` the session's videos from NAS → `/scratch/doq/job_$$/`
2. Runs `make_dataset.py --session animal/vf --skip-done --no-trt`
3. Cleans up scratch

### To resubmit only specific indices (e.g. after a failure)

```bash
bsub -J "dataset[3,7,21]" < cluster/run_session.lsf
```

### To rerun a single session interactively (for debugging)

```bash
bsub -Is -J "dataset[10]" -W 2:00 -n 12 -gpu "num=1" -q gpu_l4_large \
  -P johnson ~/analyzeMiceTrajectory/cluster/run_session.lsf
```

---

## Step 2 — Monitor

```bash
# One-shot status snapshot
bash ~/analyzeMiceTrajectory/cluster/monitor.sh

# Auto-refresh every 60s
watch -n 60 bash ~/analyzeMiceTrajectory/cluster/monitor.sh
```

The monitor shows:
- LSF job status table (RUN / PEND / DONE / EXIT counts)
- Per-session prediction progress (trials done / total)
- Cost estimate (GPU-hours consumed + remaining)

### Watching a specific job's live output

```bash
tail -f ~/analyzeMiceTrajectory/cluster/logs/job_<JOBID>_<IDX>.out
```

### Check for failed jobs

```bash
bjobs -noheader -J dataset | grep EXIT
# or after jobs finish:
bacct -u doq | grep EXIT
```

---

## Step 3 — Handle failures / resume

Jobs use `--skip-done`, so resubmitting a failed index is safe — it picks up
where it left off at the trial level.

```bash
# Kill all running dataset jobs
bkill -J dataset

# Resubmit everything (skip-done skips completed trials)
bsub < cluster/run_session.lsf
```

---

## Step 4 — Filter and build final dataset

Once all `data3D.csv` files are generated:

```bash
cd ~/analyzeMiceTrajectory
conda run -n jarvis_repro python3 filter_dataset.py
```

Outputs: `dataset/rory.csv`, `dataset/wilfred.csv`

Each row: `animal, video_folder, trial_index, frame_id, Snout_x, ..., TailTip_conf`

To skip the arena mask filter (if `predictions3D/arena_mask.npz` is missing):

```bash
python3 filter_dataset.py --no-arena-mask
```

---

## Cost and time estimates

| Metric | Value |
|--------|-------|
| Total frames (26 sessions) | ~4.5M |
| Throughput | ~3.9 fps (GPU inference bottleneck) |
| Wall time (longest session, 553k frames) | ~39 hours |
| Wall time limit in LSF script | 48 hours |
| GPU cost (L4, $0.10/GPU-hr) | ~$32 total |

---

## Key files

| File | Purpose |
|------|---------|
| `cluster/run_session.lsf` | LSF batch script for one array element |
| `cluster/sessions.txt` | 26 sessions, one per line (maps index → session) |
| `cluster/monitor.sh` | Live progress + cost dashboard |
| `make_dataset.py` | Core prediction pipeline |
| `filter_dataset.py` | Post-processing: arena mask + confidence filter |
| `compile_trt_hybrid24.py` | (local only) Compile TRT models — NOT for cluster |
| `calib_params/` | Per-date camera calibration YAML files |
| `dataset/predictions_raw/` | Raw per-trial data3D.csv outputs |

---

## Known issues and solutions

### 1. ffmpeg NVDEC unavailable on cluster (CUDA_ERROR_DEVICE_UNAVAILABLE)

**Symptom:** All prediction rows are NaN; job completes in seconds at "3000+ fps"

**Root cause:** PyTorch loads the GPU first when the model is initialized. When
`make_dataset.py` then tries to spawn ffmpeg subprocesses with `-hwaccel cuda`,
the CUDA device is already exclusively held by PyTorch and ffmpeg gets
`CUDA_ERROR_DEVICE_UNAVAILABLE`.

**Fix (already in make_dataset.py):** The `_iter_frames_ffmpeg` function runs a
1-frame probe before starting all 16 camera processes. If hwaccel fails, it
automatically falls back to software (CPU) decode. Software decode is the same
speed as the old cv2 approach, since the bottleneck is GPU inference, not decoding.

**You'll see this in the log:**
```
WARNING ffmpeg probe failed (hw=True): ... CUDA_ERROR_DEVICE_UNAVAILABLE
INFO    Software decode works — switching to no-hwaccel
INFO    ffmpeg probe OK (21172800 bytes/frame, hw=False, res=3208x2200)
```
This is normal — the fallback works correctly.

### 2. TensorRT not supported on cluster

**Symptom:** `torch_tensorrt` compile errors

**Root cause:** pip-installed `torch_tensorrt 2.5.0` ships with cuDNN 9, but the
`ir='ts'` (TorchScript) mode requires cuDNN 8 + system TRT plugins. These are
incompatible on the cluster pip environment.

**Fix:** `--no-trt` flag is hardcoded in `run_session.lsf`. TRT is only useful
locally where cuDNN 8 + TRT plugins are properly installed.

**Workaround if you want TRT speed:** Compile TRT models locally first:
```bash
conda run -n jarvis python3 compile_trt_hybrid24.py
```
Then rsync the `.pt` files to the cluster. But software PyTorch on L4 is fast
enough for the dataset size.

### 3. decord OOM (if you try to re-enable it)

**Symptom:** LSF job killed with `TERM_MEMLIMIT`

**Root cause:** `decord.VideoReader` pre-loads the entire video file into RAM.
16 cameras × ~7 GB per session = ~112 GB, exceeding the 180 GB LSF mem limit.

**Fix:** decord is removed from the pipeline entirely. `make_dataset.py` uses
ffmpeg subprocesses (one per camera) that stream frames without loading the whole
file.

### 4. Wall time too short

**Symptom:** Job killed with `TERM_RUNLIMIT` before session finishes

**Root cause:** Original `-W 12:00` (12 hours). The longest session (wilfred
`2026_01_04_16_35_15`, 553k frames) takes ~39 hours at 3.9 fps.

**Fix:** `run_session.lsf` now uses `-W 48:00`. Since `--skip-done` is set,
a job can be resubmitted safely if it still gets killed.

### 5. NAS SSH connection refused when all jobs start simultaneously

**Symptom:** Job exits in ~1 second with:
```
Connection closed by 10.10.1.28 port 22
rsync: connection unexpectedly closed (0 bytes received so far) [Receiver]
rsync error: unexplained error (code 255) at io.c(228)
```

**Root cause:** All 26 array elements start at the same second and open 26
simultaneous SSH connections to the NAS. The NAS drops the excess connections.

**Fix (already in run_session.lsf):** A random stagger sleep of 0–119 seconds
(seeded by array index) is added before the rsync. Jobs spread out across ~2
minutes, keeping the NAS connection count manageable.

**If it still happens:** Resubmit just the failed indices — `--skip-done` means
no work is duplicated:
```bash
bsub -J "dataset[14,15]" -W 48:00 -n 12 -gpu "num=1" -q gpu_l4_large \
  -P johnson \
  -o ~/analyzeMiceTrajectory/cluster/logs/job_%J_%I.out \
  -e ~/analyzeMiceTrajectory/cluster/logs/job_%J_%I.err \
  ~/analyzeMiceTrajectory/cluster/run_session.lsf
```

### 6. Video files not available during predictions

**Symptom:** `Expected N cameras, found 0` or similar

**Root cause:** rsync from NAS failed or scratch directory was cleaned up early.

**Fix:** Check the `[1/4] Sync` section of the job log. NAS host is
`ratan@10.10.1.28:/mouse2/climb`. If the NAS is unreachable, the rsync will fail
and `set -euo pipefail` will abort the job cleanly.

---

## Updating the code on the cluster

The cluster cannot `git pull` directly (no GitHub SSH key there). Use rsync:

```bash
# From your local machine:
rsync -av /path/to/analyzeMiceTrajectory/make_dataset.py \
    doq@login1:~/analyzeMiceTrajectory/make_dataset.py

rsync -av /path/to/analyzeMiceTrajectory/cluster/ \
    doq@login1:~/analyzeMiceTrajectory/cluster/
```

---

## Running locally (without cluster)

```bash
# Process one session
conda run -n jarvis python3 make_dataset.py \
    --session rory/2026_01_07_17_31_36 \
    --video-root /mnt/mouse2 \
    --trials-dir dataset/ \
    --calib-root calib_params/ \
    --output-dir dataset/ \
    --skip-done

# Process all sessions
conda run -n jarvis python3 make_dataset.py \
    --video-root /mnt/mouse2 \
    --trials-dir dataset/ \
    --calib-root calib_params/ \
    --output-dir dataset/ \
    --skip-done
```

Add `--no-trt` if TRT models are not compiled.
