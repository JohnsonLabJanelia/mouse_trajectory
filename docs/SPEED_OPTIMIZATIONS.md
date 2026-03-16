# Speed Optimizations: 4-kp 4.73 fps → 24-kp 7.9 fps

Summary of everything done to make the JARVIS inference pipeline faster and
more accurate.

## Baseline

| | |
|---|---|
| Model | mouseClimb4 (4 keypoints) |
| Pipeline | JARVIS HybridNet 3D (CenterDetect → KeypointDetect → V2V volumetric net) |
| Speed | **4.73 fps** (16 cameras, GPU batch) |
| Keypoints | 4 (Snout, EarL, EarR, Tail) |

---

## Optimization 1 — TensorRT FP16 compilation

**Script:** `compile_trt_hybrid24.py`

Both CenterDetect and KeypointDetect are compiled to TRT FP16 JIT modules via
`torch_tensorrt.compile(..., ir='ts', enabled_precisions={torch.float16})`.

Key fix required: the bundled torch_tensorrt 1.x `.so` files in the jarvis
conda env were compiled for Python 3.9 but the env runs Python 3.10 + PyTorch
2.5.1. The fix was to patch
`jarvis/prediction/jarvis3D.py` to use the 2.5.0 `ir='ts'` API instead of
loading the old shared libraries.

| Model | Input shape | Batch |
|---|---|---|
| CenterDetect TRT | (16, 3, 320, 320) | 16 cams at once |
| KeypointDetect TRT | (1–16, 3, 704, 704) | all detected cams at once |

**Result:** mouseClimb4 baseline 4.73 → **6.09 fps** (+29%)

---

## Optimization 2 — Bypass HybridNet; use 2D+DLT triangulation

**Script:** `predict2D_triangulate.py`

The HybridNet volumetric step (V2V-Net on a 100³ grid) is slow and was the
primary bottleneck. It is replaced by:

1. **Batched CenterDetect** — all 16 cameras in one GPU forward pass.
2. **Batched KeypointDetect** — all cameras that detected the mouse in one
   GPU forward pass (M ≤ 16 crops).
3. **Robust DLT triangulation** — per keypoint, per frame, in pure NumPy.
   Iterative outlier rejection removes cameras whose reprojection error
   exceeds `max_reproj_err` (default 200 px). Typically 7–16 cameras
   contribute per keypoint.

```
CenterDetect (all 16 cams) → detected mask
KeypointDetect (M detected cams) → 24×(x,y,conf) per cam
DLT triangulate (24 keypoints) → 24×(x,y,z) world coords
```

DLT math (same as JARVIS ReprojectionTool, but in NumPy for flexibility):

```python
def _dlt_core(pts, Ps, ws):
    A = []
    for (u, v), P, w in zip(pts, Ps, ws):
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]; X /= X[3]
    return X[:3]
```

Calibration convention (JARVIS stores K transposed):
```python
RT = np.vstack([R, T.reshape(1, 3)])  # 4×3
P  = (RT @ K).T                        # 3×4
```

**Result at benchmark frames (15186–15285, few cameras detect mouse):**
- no TRT: **5.10 fps**
- with TRT: **6.12 fps**

**At active trial frames (all 16 cameras detecting mouse, e.g. frame 59035):**
- with TRT: **~7.9 fps**

The reason active frames are faster: fewer NaN branches, GPU batches are
fully packed, and the triangulation loop is tight.

---

## Optimization 3 — mouseHybrid24 composite project

**Config:** `JARVIS-HybridNet/projects/mouseHybrid24/config.yaml`

mouseJan30's own CenterDetect (Run_20260309-113238) does not generalize to
the local rig. mouseClimb4's CenterDetect (Run_20260202-120720) was trained
on the local rig and works perfectly.

Solution: create a composite project that symlinks:
- `models/CenterDetect/` → mouseClimb4 Run_20260202-120720
- `models/KeypointDetect/` → mouseJan30 Run_20260311-151206

This avoids the JARVIS config singleton bug (loading two projects mutates
shared global cfg). Using a single project keeps cfg consistent.

---

## Optimization 4 — Trial-only prediction

**Script:** `trainJarvisNoGui/predict_trials_2d.py` / `make_dataset.py`

The mouse is only in the arena during door-open → door-close trial windows
(extracted by `cbot_climb_log/extract_trial_frames.py`). Running inference on
the full recording (hours of video) would waste most frames. The pipeline
only predicts within trial windows.

Additional arena-mask gating is applied in `filter_dataset.py`: only frames
where the Snout projects inside `predictions3D/arena_mask.npz` in Cam2005325
image space are kept in the final dataset.

---

## Summary table

| Step | Change | fps | kp |
|---|---|---|---|
| Baseline | mouseClimb4 HybridNet 3D | 4.73 | 4 |
| +TRT | compile CenterDetect + KeypointDetect | 6.09 | 4 |
| +DLT | replace HybridNet with triangulation | 6.12 | 24 |
| +Active frames | full GPU batches during climbing | **7.9** | **24** |
| +Trial-only | skip non-trial frames entirely | — | 24 |

---

## Files changed

| File | Change |
|---|---|
| `JARVIS-HybridNet/jarvis/prediction/jarvis3D.py` | Patched torch_tensorrt 1.x → 2.5.0 `ir='ts'` API |
| `compile_trt_hybrid24.py` | Compiles mouseHybrid24 CenterDetect + KeypointDetect to TRT FP16 |
| `predict2D_triangulate.py` | Core 2D+DLT engine; batched GPU inference + robust DLT triangulation |
| `trainJarvisNoGui/predict_trials_2d.py` | Session runner: models loaded once, all trials processed per session |
| `run_full_pipeline.py` | `--trt` flag routes step 2 to predict_trials_2d.py; default project = mouseHybrid24 |
| `make_dataset.py` | Full dataset generation pipeline: models loaded once for all 400 trials |
| `filter_dataset.py` | Arena-mask + confidence gating; outputs clean per-animal CSVs |
| `JARVIS-HybridNet/projects/mouseHybrid24/config.yaml` | Composite project: mouseClimb4 CD + mouseJan30 KD |
