# Trajectory analysis output layout

This document describes the folder structure and contents under `trajectory_analysis/` when you run the session and phase plotting scripts.

## Overview

Under each **animal** there are two top-level folders:

- **`sessions/`** — one subfolder per recording session: main trajectory plot plus optional **vertical_left** and **vertical_right** breakdowns (when trial type is available).
- **`phases/`** — three subfolders (**early**, **mid**, **late**), each with a main plot and optional **vertical_left** / **vertical_right** subfolders.
- **`phase_vs_side.png`** — one composite figure per animal: 3×2 grid (phase × side: early/mid/late × vertical left/right).

All plots use **(u, v)** camera coordinates with **z as color** (elevation). The same trajectory filters apply everywhere; see **`docs/TRAJECTORY_FILTERS.md`**.

**Trial type (vertical left vs right)** comes from the log: run **`cbot_climb_log/export_trial_types_for_trajectories.py`** to generate **`trajectory_analysis/trial_types.csv`**. The session and phase scripts then split trials into vertical-left (left ladder vertical, right at an angle) and vertical-right (right ladder vertical, left at an angle) and write the extra plots when this file is present.

## Folder structure

```
trajectory_analysis/
  trial_types.csv              # from export_trial_types_for_trajectories.py (optional)
  rory/
    phase_vs_side.png         # 3×2: phase (early/mid/late) × side (vertical left/right)
    sessions/
      rory_2025_12_23_16_57_09/
        trajectory_xy_z.png      # all trials in this session
        vertical_left/           # when trial_types.csv present
          trajectory_xy_z.png    # trials with vertical on left
        vertical_right/
          trajectory_xy_z.png    # trials with vertical on right
      ...
    phases/
      early/
        trajectory_xy_z.png
        vertical_left/
        vertical_right/
      mid/
      late/
  wilfred/
    sessions/
    phases/
```

## What goes where

| Path | Content | Script |
|------|---------|--------|
| `{animal}/sessions/{session_folder}/trajectory_xy_z.png` | All trials in that session, u-v with z as color. | `plot_session_trajectories.py` |
| `{animal}/sessions/{session_folder}/vertical_left/trajectory_xy_z.png` | Trials with vertical on left (from trial_types.csv). | `plot_session_trajectories.py` |
| `{animal}/sessions/{session_folder}/vertical_right/trajectory_xy_z.png` | Trials with vertical on right. | `plot_session_trajectories.py` |
| `{animal}/phases/early/trajectory_xy_z.png` (and mid, late) | All trials in that phase (first/second/last third). | `plot_phase_trajectories.py` |
| `{animal}/phases/{phase}/vertical_left/trajectory_xy_z.png` | Phase subset: vertical on left. | `plot_phase_trajectories.py` |
| `{animal}/phases/{phase}/vertical_right/trajectory_xy_z.png` | Phase subset: vertical on right. | `plot_phase_trajectories.py` |
| `{animal}/phase_vs_side.png` | One 3×2 figure: rows = early/mid/late, cols = vertical left / vertical right. | `plot_phase_vs_side.py` |

Session folder names follow the pattern `{animal}_{YYYY}_{MM}_{DD}_{HH}_{MM}_{SS}` (e.g. `rory_2025_12_23_16_57_09`). Phase assignment is by **trial count**: trials are ordered chronologically (session order, then trial order within session) and split evenly into three groups.

## How to generate

0. **Trial types** (optional, for vertical left/right breakdown):  
   Generate `trajectory_analysis/trial_types.csv` from robot_manager logs so session and phase scripts can split by trial type:

   ```bash
   python cbot_climb_log/export_trial_types_for_trajectories.py \
     --predictions-dir /path/to/predictions3D \
     -o trajectory_analysis/trial_types.csv
   ```

   Use the same predictions root as below. If this file is missing, only the main “all trials” plots are written (no vertical_left / vertical_right subfolders).

1. **Sessions** (per-session plots):

   ```bash
   python plot_session_trajectories.py \
     --predictions-root /path/to/predictions3D \
     -o trajectory_analysis
   ```

2. **Phases** (early / mid / late aggregates):

   ```bash
   python plot_phase_trajectories.py \
     --predictions-root /path/to/predictions3D \
     -o trajectory_analysis
   ```

3. **Phase vs side** (one 3×2 figure per animal; requires trial_types.csv):

   ```bash
   python plot_phase_vs_side.py \
     --predictions-root /path/to/predictions3D \
     -o trajectory_analysis
   ```

Use the same `-o trajectory_analysis` (or your chosen output root) for all. You can run them in either order; both create their own subfolders under `{animal}/sessions/` and `{animal}/phases/` respectively.

Options (both scripts): `--use-world-xy` to plot 3D world (x, y) instead of camera (u, v); `--animals rory wilfred` to restrict to specific animals; `--trial-types <path>` to override the path to trial_types.csv (default: `<output-dir>/trial_types.csv`).

## Filters

All trajectory points are filtered before plotting (see **`docs/TRAJECTORY_FILTERS.md`**):

- **z** must be in **[0, 150]**.
- Points with **u < 1250** (px) and **z > 50** are dropped.

These apply to session plots and phase plots alike.
