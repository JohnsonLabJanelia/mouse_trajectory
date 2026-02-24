# Mouse Trajectory Analysis Documentation

This directory contains documentation for the mouse trajectory analysis pipeline.

## Documentation Files

- **[PIPELINE.md](PIPELINE.md)**: Complete guide to the full pipeline (extraction → JARVIS prediction)
- **[TRAJECTORY_VISUALIZATION.md](TRAJECTORY_VISUALIZATION.md)**: Documentation for trajectory visualization scripts (`plot_trajectory_xy.py`, `plot_trajectory_on_frame.py`, `extract_first_frame.py`)
- **[FLOW_FIELD_PLOTS.md](FLOW_FIELD_PLOTS.md)**: How the two flow-field panels (elevation + flow direction, flow speed + flow direction) are computed and what they show
- **[RORY_MIDLINE_AND_GOALS.md](RORY_MIDLINE_AND_GOALS.md)**: Rory midline-and-goals analysis (output dir `midline_and_goals/`): elevation/flow panels, midline_and_goals.json, path plots by crossing count and late phase/vertical left–right, crossing-location histograms
- **[ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)**: Summary of trajectory analysis: mouse choice (side pick at peak z), side-pick heatmap, vertical-on-left/right trajectory and flow-field plots, and trial-type filtering
- **[../ISSUE_LOG.md](../ISSUE_LOG.md)**: Known issues and troubleshooting

## Quick Start

1. **Extract trial frames**:
   ```bash
   python run_full_pipeline.py --step 1 --animals rory wilfred
   ```

2. **Run JARVIS 3D prediction**:
   ```bash
   python run_full_pipeline.py --step 2 --animals rory wilfred
   ```

3. **Analyze trajectories**:
   - Use `data3D.csv` files from `predictions3D/` directories
   - See example analysis scripts (to be added)

## Pipeline Overview

The pipeline processes video recordings of mice performing climbing trials:

1. **Extract trial frames**: Matches video frames to door open/close events from robot logs
2. **JARVIS 3D prediction**: Generates 3D pose estimates for each trial using multi-camera calibration
3. **Downstream analysis**: Speed, directedness, elevation gain, etc. (see analysis scripts)

For detailed documentation, see [PIPELINE.md](PIPELINE.md).
