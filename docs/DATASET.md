# 24-Keypoint 3D Dataset

## Overview

Full-body 3D pose estimates for Rory and Wilfred, derived from the 16-camera
rig using the `mouseHybrid24` model (CenterDetect from mouseClimb4 + KeypointDetect
from mouseJan30) with DLT triangulation. No HybridNet volumetric step — 2D
detections from all cameras are triangulated with robust outlier rejection.

## Directory layout

```
analyzeMiceTrajectory/
  dataset/
    predictions_raw/               ← raw per-trial 3D predictions
      rory_2025_12_23_16_57_09/
        trial_0000_15186-145628/
          data3D.csv               ← 96 cols (24 kp × x,y,z,conf), one row per predicted frame
        trial_0001_155332-193782/
          data3D.csv
        ...
      rory_2026_01_07_17_31_36/
        ...
      wilfred_2026_01_01_14_03_49/
        ...
    rory.csv                       ← filtered clean dataset for rory
    wilfred.csv                    ← filtered clean dataset for wilfred
    sessions.csv                   ← session metadata (video path, calib, fps)
    make_dataset.log               ← pipeline progress log
```

## Clean dataset columns (`rory.csv`, `wilfred.csv`)

| Column | Type | Description |
|---|---|---|
| `animal` | str | `rory` or `wilfred` |
| `video_folder` | str | `YYYY_MM_DD_HH_MM_SS` session identifier |
| `trial_index` | int | Trial number within the session (from log) |
| `frame_id` | int | Absolute video frame number |
| `Snout_x` | float | Snout 3D x (mm) |
| `Snout_y` | float | Snout 3D y (mm) |
| `Snout_z` | float | Snout 3D z (mm) |
| `Snout_conf` | float | Mean 2D keypoint confidence across detecting cameras |
| ... | | Same pattern for all 24 keypoints (see Keypoints below) |

Only frames where the Snout projects **inside the arena mask** (Cam2005325
image space) are included. Frames with NaN Snout (mouse not detected) are
excluded.

## Sessions metadata (`sessions.csv`)

| Column | Description |
|---|---|
| `animal` | Animal name |
| `video_folder` | Session folder name |
| `calib_date` | Which calibration date was used (nearest to recording date) |
| `video_path` | Full path to the raw video folder on disk |
| `calib_dir` | Full path to the calibration YAML files |
| `fps` | Camera frame rate (180 fps) |

## 24 Keypoints

```
Snout, EarL, EarR, Neck, SpineL, TailBase,
ShoulderL, ElbowL, WristL, HandL,
ShoulderR, ElbowR, WristR, HandR,
KneeL, AnkleL, FootL,
KneeR, AnkleR, FootR,
TailTip, TailMid, Tail1Q, Tail3Q
```

## Coordinate system

World coordinates in **mm**, JARVIS camera calibration convention.
The coordinate origin is defined by the calibration procedure for each date.
Within a session the coordinate system is consistent across all 24 keypoints.

To project a 3D point onto camera `CamXXXXXX`:

```python
import cv2, numpy as np

def load_P(calib_dir, cam_name):
    fs = cv2.FileStorage(f'{calib_dir}/{cam_name}.yaml', cv2.FILE_STORAGE_READ)
    K  = fs.getNode('intrinsicMatrix').mat()
    R  = fs.getNode('R').mat()
    T  = fs.getNode('T').mat()
    fs.release()
    RT = np.vstack([R, T.reshape(1, 3)])   # 4x3
    return (RT @ K).T                      # 3x4

P = load_P('/path/to/calib_params/2026_01_07', 'Cam2002486')
X = np.array([[x, y, z, 1.0]])            # 1x4 homogeneous
p = X @ P.T                               # 1x3
u, v = p[0,0]/p[0,2], p[0,1]/p[0,2]     # pixel coords
```

To grab the corresponding video frame:

```python
import cv2, pandas as pd

df = pd.read_csv('dataset/rory.csv')
row = df.iloc[0]

meta = pd.read_csv('dataset/sessions.csv').set_index(['animal','video_folder'])
session = meta.loc[(row.animal, row.video_folder)]

cap = cv2.VideoCapture(f"{session.video_path}/Cam2002486.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, row.frame_id)
ret, frame = cap.read()
cap.release()
```

## Pipeline scripts

| Script | Purpose |
|---|---|
| `make_dataset.py` | Generate raw per-trial predictions (runs for days, resumable) |
| `filter_dataset.py` | Post-process raw predictions → clean per-animal CSVs |
| `predict2D_triangulate.py` | Core 2D+DLT prediction engine |
| `trainJarvisNoGui/predict_trials_2d.py` | Per-session wrapper (used internally) |

### Regenerating the dataset

```bash
# Full run (rory + wilfred, all frames, ~159 hours)
conda run -n jarvis python3 make_dataset.py --skip-done

# Faster run at 60fps effective (stride 3, ~53 hours)
conda run -n jarvis python3 make_dataset.py --stride 3 --skip-done

# One animal only
conda run -n jarvis python3 make_dataset.py --animals rory --skip-done

# After make_dataset.py finishes (or to update while running)
python3 filter_dataset.py
```

### Checking progress

```bash
# How many trials done
find dataset/predictions_raw -name data3D.csv | wc -l

# Live log
tail -f dataset/make_dataset.log
```

## Model details

| Component | Model | Weights |
|---|---|---|
| CenterDetect | mouseClimb4 | Run_20260202-120720 |
| KeypointDetect | mouseJan30 | Run_20260311-151206 (Epoch_40) |
| HybridNet | — | Bypassed; DLT triangulation used instead |
| Speed | TRT FP16 | ~7.9 fps (16 cameras, active frame) |

DLT triangulation: robust iterative outlier rejection, removes cameras with
reprojection error > 200 px per keypoint. Typically 7–16 cameras contribute
to each 3D point depending on occlusion.

## Arena mask

The arena mask (`predictions3D/arena_mask.npz`) is a binary (2200, 3208) image
in Cam2005325's image space. Only frames where the Snout projects inside this
mask are included in the filtered dataset. This gates out frames where the
mouse is in the start/end waiting area or outside the climbing structure.
