## Pipeline issue log

This file tracks known and detected issues when running the mouse trajectory
pipeline (extraction, calibration, JARVIS prediction), so we can quickly see
what went wrong, where, and what to fix.

---

### Issue: Corrupted / mismatched videos in `rory_2025_12_19_13_33_51`

- **Date noted**: 2026-02-05  
- **Animal / session**: `rory / 2025_12_19_13_33_51`  
- **Stage**: JARVIS prediction (step 2), `predict_trials.py` → `predict3D.py`  
- **Error**:
  - ffmpeg/OpenCV logs:
    - `stream 0, contradictionary STSC and STCO`
    - `error reading header`
  - JARVIS assertion:
    - `AssertionError: All videos need to have the same resolution`
- **What we inspected** (`/mnt/mouse2/rory/2025_12_19_13_33_51`):
  - Several cameras report **0×0 resolution and 0 frames**:
    - `Cam2002486.mp4: 0x0, frames=0`
    - `Cam2005325.mp4: 0x0, frames=0`
    - `Cam2006054.mp4: 0x0, frames=0`
    - `Cam2006055.mp4: 0x0, frames=0`
  - Other cameras look valid:
    - e.g. `Cam2002487.mp4: 3208x2200, frames=579687` (and similar for most others)
  - There is also `Cam710040.mp4` with a **different** resolution:
    - `Cam710040.mp4: 1856x984, frames=96615`
- **Hypothesis / root cause**:
  - MP4 container metadata for some cameras is **corrupted** (bad STSC/STCO tables),
    so ffmpeg/OpenCV cannot read headers and returns 0×0/0 frames, even though file
    sizes on disk look similar.
  - JARVIS requires all used cameras to have the **same non-zero resolution**; once
    it hits a corrupted or mismatched camera, `create_video_reader` asserts.
  - We have already adjusted calibration loading to **always skip `Cam710040`**, but
    the 0×0 cameras are still present in this session and cause the mismatch.
- **Impact**:
  - JARVIS 3D prediction cannot run for this session with the current set of videos.
- **Potential fixes / follow-ups**:
  - Repair or re-export the corrupted MP4s for the affected cameras, or remove them
    and update calibration/dataset to use only the healthy cameras for this day.
  - Optionally add a pre-check script that:
    - Scans each session folder’s `Cam*.mp4` and logs any video with 0×0 resolution
      or unreadable header.
    - Records these findings here so issues are visible immediately after running
      the full pipeline.

---

### Issue: Missing same-day calibration; using nearest-date calibration instead

- **Date noted**: 2026-02-05  
- **Affected scope**: All sessions whose video date (e.g. `2025_12_19`) does **not** have
  a corresponding calibration folder under `/home/user/red_data/calib/YYYY_MM_DD/calibration`.  
- **Stage**: JARVIS prediction (step 2), calibration selection in `run_full_pipeline.py`.  
- **Current behaviour**:
  - For each session `animal/YYYY_MM_DD_HH_MM_SS`, we:
    - Parse the video date `YYYY_MM_DD`.
    - Look under `--calib-root` (default `/home/user/red_data/calib`) for folders named
      `YYYY_MM_DD` that contain a `calibration/` subfolder.
    - Ignore known-bad calibration dates: `2025_12_19`, `2025_12_21`.
    - If an **exact same-day** calibration exists, we use it.
    - Otherwise, we pick the **closest available calibration date in days** and convert
      that folder’s `calibration/` YAMLs to JARVIS format (via `convert_calibration.py`)
      into `analyzeMiceTrajectory/calib_params/YYYY_MM_DD/`.
    - The resulting JARVIS-format folder is passed as `--dataset-name` to
      `predict_trials.py` for that session.
- **Why this is an issue**:
  - Using a calibration from a different day is a **fallback**, not ideal:
    - Camera geometry and positions should be stable, but if the rig moved or cameras
      were re-aligned between days, the nearest-date calibration could introduce
      systematic 3D errors.
    - Some calibration dates are explicitly bad and must be ignored (e.g. 12/19, 12/21),
      so we rely on the “next best” day.
- **Impact**:
  - Predictions for sessions without a same-day calibration may be **less accurate**,
    depending on how much the rig drifted between the true recording date and the
    calibration date we borrowed.
- **Mitigation / current implementation**:
  - `run_full_pipeline.py` now:
    - Uses **exact same-day** calibration when available.
    - Falls back to **nearest-date** calibration when same-day is missing.
    - Prints which calibration folder is used per session:
      - e.g. `[JARVIS] [1/37] rory_2025_12_19_13_33_51 (calib=/home/user/src/analyzeMiceTrajectory/calib_params/2025_12_22)`
  - This makes it clear in the logs which sessions are using fallback calibration so
    we can review them as “potential issue” sessions.
- **Potential future improvements**:
  - Require **confirmation** or a command-line flag when falling back to a different
    date’s calibration (e.g. `--allow-nearest-calib`).
  - Optionally log a summary table at the end of the pipeline run that lists, for each
    session, whether it used same-day or fallback calibration, so these sessions can be
    prioritized for better calibration in the future.


