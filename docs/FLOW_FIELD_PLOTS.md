# Flow field plots (low → high elevation)

The script `analyze_trajectories.py` produces a two-panel figure **flow_field_low_to_high.png** in `trajectory_analysis/`. It summarizes where and how mice move when they are **gaining elevation** (dz > 0).

## Output

- **File**: `trajectory_analysis/flow_field_low_to_high.png`
- **Layout**: Two side-by-side panels, same x/y limits and aspect.

### Left panel: Elevation + flow direction

- **Background**: Mean elevation (z) per grid cell over all path points (turbo colormap). Same spatial filters as the rest of the analysis (z ≤ MAX_PEAK_Z, and y/z cap in the high-y region).
- **Overlay**: Flow direction as **black arrows**. Each arrow is the dz-weighted average of horizontal displacement (dx, dy) when the mouse is moving upward (dz > 0) in that cell.
- **Interpretation**: “Where does the mouse tend to go in the horizontal plane when climbing here?” Arrow direction = average (u, v); arrow length is scaled for visibility (same scale as right panel).

### Right panel: Flow speed + flow direction

- **Background**: Mean **raw 3D velocity** per grid cell (turbo colormap). Velocity = √(dx² + dy² + dz²) / dt in distance units per second (e.g. mm/s). Only upward segments (dz > 0) contribute; cells with no flow (speed 0) are drawn as **white**.
- **Overlay**: Same **black** flow-direction arrows as on the left.
- **Interpretation**: “How fast does the mouse move in 3D when climbing here?” Color = speed; arrows = direction.

## Process (how it’s computed)

1. **Grid**: 40×40 cells over the x,y range of the all-trials data (same bounds as the all_trials_xy plot).

2. **Data used**:
   - Same trials and filters as the main trajectory analysis: `trajectory_filtered.csv` per trial, with points excluded where z > MAX_PEAK_Z or (y > Y_FOR_Z_CAP and z > Z_CAP_IN_Y_REGION).
   - Paths are ordered by segment and frame (`_ordered_xyz_frame_path`).

3. **Per-cell quantities** (only segments with **dz > 0**):
   - Segment midpoint must not lie in the excluded y/z region (mid_y, mid_z cap).
   - **Weight** = dz (elevation gain on that segment).
   - For each segment: (dx, dy, dz), dt = Δframe / FPS, raw_vel = √(dx² + dy² + dz²) / dt.
   - Each segment contributes to the cell containing its **midpoint** (mid_x, mid_y):
     - `sum_dx += dx * weight`, `sum_dy += dy * weight`, `sum_w += weight`, `sum_vel += raw_vel * weight`.
   - Elevation grid: all path points (not just upward segments) contribute to mean z per cell for the **left** panel.

4. **Flow field**:
   - **u** = sum_dx / sum_w, **v** = sum_dy / sum_w (horizontal flow direction).
   - **mean_raw_vel** = sum_vel / sum_w (mean 3D speed when climbing, per cell).

5. **Arrow scaling**: Arrow length in data coordinates is (u,v)/scale, with scale set so the 95th percentile of horizontal speed maps to a reasonable length (scale factor 2.5). Same scale on both panels.

6. **View**: Symmetric limits around the data center, with 2% padding, so both panels are zoomed similarly and centered.

7. **Visual choices**:
   - Colormap: **turbo** for both elevation and speed.
   - Zero speed shown as **white** on the right (NaN in the speed array, colormap `set_bad("white")`).
   - Arrows: **black**, thin (width 0.006), with white edge; drawn on top of the background (zorder=10).
   - Color scale for speed: 10th–95th percentile of positive mean_raw_vel (avoids stretching by zeros).

## Constants involved

- **MAX_PEAK_Z**, **Y_FOR_Z_CAP**, **Z_CAP_IN_Y_REGION**: same as in the rest of `analyze_trajectories.py`.
- **FPS** = 180 for dt and velocity in “per second”.
- **n_bins** = 40 for the flow grid.
- **pad_frac** = 0.02 for axis limits.

## Summary

- **Left**: Elevation as background, flow direction as black arrows (low → high elevation only).
- **Right**: Flow speed as background (white where no flow), same flow direction arrows.
- Both panels use the same grid, same filters, and the same flow (u,v); only the scalar background (elevation vs speed) and its color scale differ.
