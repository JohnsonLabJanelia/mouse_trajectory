# Head direction analysis

Head direction is computed from the **snout–ear triangle** (Snout, EarL, EarR) in `data3D.csv`: the angle in the camera (u,v) plane from the ear midpoint toward the snout. Outputs live under `trajectory_analysis/<animal>/head_direction/`.

## Current outputs

- **`head_direction_angle_reference.png`** — **Angle reference**: diagram showing which angle (degrees) corresponds to which direction in the image (0° = right / u+, 90° = down / v+, ±180° = left, −90° = up). Generated every run.
- **`head_direction_by_phase.png`** — **Full path**: three panels (early / mid / late) using all trajectory in `trajectory_filtered.csv` (valid path segments). Title suffix: “(full path)”.
- **`head_direction_by_phase_start_to_reward.png`** — **Start to reward only**: same layout but only frames from trial start up to the reward frame (requires `reward_times.csv` or `--logs-dir`). Title suffix: “(start to reward)”. Generated only when reward times are available.

Both use (u,v) heatmaps colored by mean head direction (circular mean per bin) and overlay midline and goals from `midline_and_goals.json` when present.

When reward times and `midline_and_goals.json` are available, the script also runs **six start-to-reward analyses** and saves figures in the same folder:

- **`head_at_midline_crossing.png`** — Head direction at each midline crossing: boxplots by phase and by direction (toward goal 1 vs goal 2).
- **`head_when_entering_goal.png`** — Head direction at first entry into goal 1 and goal 2: boxplots by phase.
- **`head_goal_alignment_goal1.png`** / **`head_goal_alignment_goal2.png`** — Heatmap of mean |head − direction to goal| (°) per (u,v) bin (lower = more aligned).
- **`head_vs_movement_direction.png`** — Histogram of (head − movement)° by phase (0° = head aligned with movement).
- **`head_direction_polar_by_region.png`** — Polar histograms of head direction in four regions: above midline, below midline, goal 1, goal 2.
- **`time_facing_each_goal.png`** — Fraction of frames (±45°) facing goal 1 vs goal 2, by phase.

## Running

```bash
# Rory (default); uses trajectory_analysis/rory/midline_and_goals/midline_and_goals.json if present
python plot_head_direction.py --animal rory

# Wilfred; optional overlay from rory’s geometry
python plot_head_direction.py --animal wilfred --midline-goals-json trajectory_analysis/rory/midline_and_goals/midline_and_goals.json

# If calibration is under a root (e.g. calib_params/YYYY_MM_DD)
python plot_head_direction.py --animal rory --calib-root /path/to/calib_params
```

Requires **data3D.csv** in each trial folder (Snout, EarL, EarR) and calibration for 3D→2D projection (from trial `info.yaml` or `--calib-root`). The start-to-reward heatmap requires reward times: use `reward_times.csv` in the output dir (e.g. from `cbot_climb_log/export_reward_times.py`) or `--reward-times` / `--logs-dir`.

---

## Suggested analyses (goals and midline)

1. **Head direction at midline crossing**  
   For each trial, at the first (or every) midline crossing (start → reward), record head angle. Compare distribution early vs mid vs late, or when crossing toward goal 1 vs goal 2.

2. **Head direction when entering goal regions**  
   At first entry into each goal rectangle, record head angle. Test whether head is aligned with the goal (e.g. facing the reward zone) more in late phase.

3. **Head–goal alignment along the path**  
   For each (u,v) bin, compute both mean head direction and the direction toward goal 1 and toward goal 2; plot alignment (e.g. |head − direction_to_goal|) as a function of position and phase.

4. **Head direction vs movement direction**  
   Compare head angle with movement direction (from trajectory_filtered) per frame or per bin. Quantify “head leading” vs “body leading” by phase or by region (e.g. near midline vs near goals).

5. **Polar histograms by region**  
   Split space into: start region, below midline, above midline, goal 1 region, goal 2 region. Plot polar histograms of head direction in each region and phase.

6. **Time spent facing each goal**  
   Define “facing goal” as head within ±θ° of the direction from current position to goal center. Compute fraction of time (or frames) facing goal 1 vs goal 2, by phase and by trial type (vertical left / right if available).

Implementing any of these would follow the same pipeline: load head direction per trial (and optionally reward frame, midline/goal params), then aggregate by the desired grouping (phase, region, trial type) and plot or export statistics.
