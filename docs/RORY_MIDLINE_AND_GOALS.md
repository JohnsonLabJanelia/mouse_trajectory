# Midline-and-goals analysis (per animal)

This document describes the **midline-and-goals** analysis: flow-style panels in u–v, midline + goals JSON, path plots by crossing count (and phase / vertical left–right), crossing-location histograms, goal-region heatmaps, etc. Outputs are written under **`trajectory_analysis/<animal>/midline_and_goals/`** (e.g. `rory` or `wilfred`).

**Script:** `plot_flow_field_rory.py`  
**Animal:** use `--animal rory` (default) or `--animal wilfred`. Session folders in the predictions root must start with `<animal>_` (e.g. `rory_...`, `wilfred_...`).

## Purpose

- **Elevation and flow in u–v**: Two panels (elevation + flow direction; flow speed + flow direction) in camera pixel space; two high-z regions are detected and define the midline and two goal points.
- **Midline and goals**: Saved to `midline_and_goals.json` (`v_mid`, `goal1_u/v`, `goal2_u/v`) and used for all path and crossing analyses.
- **Path plots**: Trajectories from trial start to reward; by number of midline crossings (all trials, late phase, late vertical left, late vertical right).
- **Crossing locations**: Histogram of u positions where paths cross the midline (all trials); combined 3-panel (early/mid/late, vertical left vs right, shared y, trial counts in titles).

## Prerequisites

1. **Trajectory data**: Predictions root (e.g. JARVIS `predictions3D/`) with rory sessions and trials; each trial has trajectory CSV loaded by `analyze_trajectories.load_trajectory_csv`.
2. **Reward times**: **`reward_times.csv`** (default: `trajectory_analysis/reward_times.csv`) from `cbot_climb_log/export_reward_times.py`, or **`--logs-dir`** to resolve from `robot_manager.log`.
3. **Trial types (optional)**: **`trial_types.csv`** for vertical left/right; needed for late vertical-left and late vertical-right path plots and for the combined crossing-location plot.

## Running

```bash
# Rory (default)
python plot_flow_field_rory.py

# Wilfred (same analyses, output under trajectory_analysis/wilfred/midline_and_goals/)
python plot_flow_field_rory.py --animal wilfred
```

Options: `--animal` (default: rory), `--predictions-root`, `-o` / `--output-dir`, `--reward-times`, `--logs-dir`, `--trial-types`.

## Outputs (generated files)

| File | Description |
|------|-------------|
| **flow_field_low_to_high.png** | Two panels: elevation (u–v) + flow arrows; flow speed + flow arrows (upward segments only). |
| **flow_field_high_z_regions.png** | Same with two high-z regions highlighted; defines midline and goals. |
| **midline_and_goals.json** | `v_mid`, `goal1_u`, `goal1_v`, `goal2_u`, `goal2_v`. |
| **paths_trial_start_to_reward.png** | All rory paths from trial start to reward; midline and goals overlaid. |
| **paths_by_crossing_count.png** | 3×4 grid: paths by number of midline crossings (0, 1, …, 10, 11+) — all rory trials. |
| **early/paths_by_crossing_count.png** | Same grid, early phase only. |
| **early/paths_by_crossing_count_vertical_left.png** | Early phase, vertical-left trials only. |
| **early/paths_by_crossing_count_vertical_right.png** | Early phase, vertical-right trials only. |
| **mid/** | Same three files for mid phase. |
| **late/** | Same three files for late phase. |
| **crossing_locations_on_midline.png** | Histogram of u where paths cross the midline (all trials). |
| **crossing_locations_on_midline_combined.png** | Three rows (early/mid/late); each row = vertical left vs vertical right overlaid; shared y-axis; trial count in each phase title. |
| **paths_goal_region_exclusive_early.png** | Two panels: full path (start→reward) for trials that visit **only** goal1 region (red) vs **only** goal2 region (blue) — early phase. |
| **paths_goal_region_exclusive_mid.png** | Same, mid phase. |
| **paths_goal_region_exclusive_late.png** | Same, late phase. |
| **paths_goal_region_switches.png** | Late phase only: paths that approach one goal rectangle then the other (goal1→goal2 red, goal2→goal1 blue). |
| **paths_first_goal_left_vs_right_early.png** | Two panels: full path (start→reward) for trials whose **first goal visited** is left vs right (by u) — early phase. |
| **paths_first_goal_left_vs_right_mid.png** | Same, mid phase. |
| **paths_first_goal_left_vs_right_late.png** | Same, late phase. |
| **goal_regions_visited_histogram.png** | Bar chart: number of goal regions visited (0, 1, or 2) per trial for early, mid, and late phase (3 panels). |
| **both_goals_when_and_where.png** | For trials that visit **both** goals: (1) when they first cross (normalized time 0–1); (2) where in u–v the first transition occurs; colored by phase. |
| **first_goal_visit_heatmap_by_phase.png** | U–v heatmap of **first visit** to each goal rectangle: 3 rows (early/mid/late) × 2 columns (goal1, goal2). Each cell shows the rectangle and density of first-visit (u,v) points for that goal and phase. |
| **first_visit_to_goal_when_other_first_heatmap.png** | First visit **to that goal** when the **other** was visited first: col 1 = where they first enter goal 1 in trials that visited goal 2 first; col 2 = where they first enter goal 2 in trials that visited goal 1 first. 3 rows = early/mid/late. |

## How the plots are produced

### Goal rectangles and GOAL_RECT_GEOM

Goal regions are **rectangles** derived from the elevation grid when drawing **flow_field_high_z_regions.png**:

1. **High-z mask**: Elevation grid is thresholded at z ≥ 50 when max elevation &gt; 50, otherwise at the 80th percentile. Cells above this threshold form a high-z band on each side of the midline.
2. **Midline**: The midline `v_mid` is the mean v of the two highest local maxima (the two “goal” peaks).
3. **Split by midline**: The high-z mask is split into top (v &lt; v_mid) and bottom (v ≥ v_mid); each side contains one peak.
4. **Rectangle geometry**: On each side, vertical extent of the high-z band is computed (with a small margin). A **common rectangle height** is taken as 70% of the minimum of the two band heights, then rectangles are placed with that height, anchored near the midline (v_mid ± 5 px), and **never cross the midline**. Horizontal half-width `half_u` is 5% of (u_max − u_min), same for both rectangles (symmetric). This geometry is stored in the global **GOAL_RECT_GEOM** for the rest of the run (`half_u`, `top_bottom`, `top_top`, `bottom_bottom`, `bottom_top`).
5. **Goal points**: The two peak (u, v) positions are saved to **midline_and_goals.json** as `goal1_u/v` and `goal2_u/v`; they are used for display and for associating “goal1” vs “goal2” with the top vs bottom rectangle in `_point_goal_region`.

So: **GOAL_RECT_GEOM** is set once when the high-z figure is generated and is then used by all path-classification logic (point-in-rectangle, trial goal hits, exclusive goal-region classification).

### Point-in-region and trial classification

- **`_point_goal_region(u, v, params)`**: Uses GOAL_RECT_GEOM and the goal points in `params` to return 1 (inside goal1 rectangle), 2 (inside goal2 rectangle), or 0 (outside). Goal1 is the rectangle on the same side of the midline as `goal1_v`.
- **`_trial_goal_hits(df, params)`**: For a trajectory dataframe (e.g. start→reward), scans each (u, v) and returns `(hits_goal1, hits_goal2)` depending on whether the path ever enters each rectangle.
- **`_classify_trials_by_goal_region(...)`**: For each trial, loads trajectory from trial start to reward frame, calls `_trial_goal_hits`, and assigns the trial to **goal1_only** (hit goal1, never goal2) or **goal2_only** (hit goal2, never goal1). Trials that hit both or neither are omitted.

### Exclusive goal-region path figures (early / mid / late)

The three files **paths_goal_region_exclusive_early.png**, **paths_goal_region_exclusive_mid.png**, and **paths_goal_region_exclusive_late.png** are produced as follows:

1. Trials are split into early / mid / late (chronological thirds by session order then trial id).
2. For each phase, `_plot_paths_by_goal_region_exclusive` is called with that phase’s trial list.
3. It calls `_classify_trials_by_goal_region` to get `goal1_only` and `goal2_only` trial ids for that phase.
4. It draws a **two-panel** figure: **left** = full path from trial start to reward for all trials in `goal1_only` (red); **right** = same for `goal2_only` (blue). Midline and both goal points are drawn on each panel.

So each of these PNGs shows **full trajectories (start→reward)** for trials that entered exactly one of the two goal rectangles and never the other, split by phase.

### First goal visited (left vs right)

- **Left/right** is defined by u: the goal with smaller u is “left”, the other “right”.
- **`_first_goal_visited(df, params)`** returns 1 or 2 for the first goal rectangle the path enters (0 if never).
- **`_classify_trials_by_first_goal(...)`** splits trials into first_goal_left and first_goal_right.
- **paths_first_goal_left_vs_right_{early,mid,late}.png**: two panels each (full path start→reward), teal = first goal left, coral = first goal right.

### Goal regions visited (0 / 1 / 2) histogram

- For each trial, **`_count_goal_regions_visited`** counts how many distinct goal rectangles are entered (0, 1, or 2).
- **goal_regions_visited_histogram.png**: three bar charts (early, mid, late), each with bars for 0, 1, and 2 regions visited.

### Both goals: when and where they cross

- **`_get_both_goals_crossing_events`** finds trials that visit both rectangles and records the **first transition** (first point in the second goal): trial_id, frame, u, v, phase.
- **both_goals_when_and_where.png**: (1) **When**: histogram of normalized time in trial (0 = start, 1 = reward) at first cross, by phase; (2) **Where**: scatter of (u, v) at that first transition in the u–v plane, colored by phase (early/mid/late). Title reports total number of such trials.

### First visit to each goal: u–v heatmap by phase

- **`_get_first_goal_visit_locations`** returns, for each trial, the **(u, v)** of the **first** point where the path enters **either** goal rectangle, plus which goal (1 or 2) and phase (early/mid/late).
- **first_goal_visit_heatmap_by_phase.png**: 3×2 grid. Rows = early, mid, late; columns = goal1, goal2. Each cell shows that goal’s rectangle (black outline) and a 2D density heatmap (YlOrRd) of first-visit (u, v) points for that phase. Same color scale across cells. Title of each cell includes trial count for that goal and phase.

### First visit to each goal when the other was visited first

- For each trial we build the ordered sequence of “entries” into a goal (each time the path transitions into goal 1 or goal 2). The first entry gives “which goal was visited first”.
- **First visit to goal 2 (when goal 1 was visited first)**: trials whose first entry is goal 1; we take the (u, v) of the **first** time they enter goal 2 in that trial.
- **First visit to goal 1 (when goal 2 was visited first)**: trials whose first entry is goal 2; we take the (u, v) of the **first** time they enter goal 1.
- **first_visit_to_goal_when_other_first_heatmap.png**: 3×2 grid. Column 1 = goal 1 rectangle, heatmap of those first-visit (u, v) to goal 1 (trials that visited goal 2 first), by phase. Column 2 = goal 2 rectangle, heatmap of first-visit (u, v) to goal 2 (trials that visited goal 1 first), by phase. So you see **where** in the second-visited goal they first step into it.

## Midline crossing logic

- **Goal-based crossing**: A crossing is counted only when the path goes clearly toward the other goal (past the midpoint between `v_mid` and that goal’s v).
- **Crossing location**: For each crossing, the u coordinate at `v = v_mid` is computed (linear interpolation) and used in the histograms.

## Phase and trial type

- **Phase**: Early / mid / late = chronological thirds (session order, then trial id).
- **Vertical left/right**: From `trial_types.csv` (left_angle_deg == 360 vs right_angle_deg == 360).
