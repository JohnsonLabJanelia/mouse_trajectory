# Trajectory data filters

All trajectory loading and plotting in this repo applies the following filters so that outlier or physically inconsistent points are excluded. They are applied to **all animals and all sessions**.

## 1. Global elevation cap (z)

- **Rule:** Drop any point with **z > 150**.
- **Reason:** Values above ~150 are treated as artifacts/outliers (e.g. spurious peaks near 200).
- **Applied in:** `load_trajectory_csv()` in `plot_session_trajectories.py`, `plot_phase_trajectories.py`, `analyze_trajectories.py`; elevation filter in `plot_trajectory_on_frame.py` (and thus in any saved `trajectory_filtered.csv` when re-run).

## 2. Region-specific elevation cap (u and z)

- **Rule:** Drop any point where **u < 1250** (px) **and** **z > 50**.
- **Reason:** In the low-u (left) region of the image, elevations above 50 are treated as inconsistent/outliers; this avoids spurious high-z points in that region.
- **Constants:** `U_LOW_THRESHOLD = 1250`, `Z_CAP_WHEN_U_LOW = 50` (or script-specific names, same values).
- **Applied in:** Same scripts as above. When trajectory data has a `u` column (camera horizontal coordinate), points that fail this condition are removed; when only 3D world (x, y, z) is available, this filter is skipped.

## 3. Non-negative elevation (z)

- **Rule:** Drop any point with **z < 0**.
- **Reason:** Elevation is defined non-negative; negative z is not physically meaningful.
- **Applied in:** All of the above (together with the caps).

## Summary table

| Condition              | Action   |
|------------------------|----------|
| z < 0                  | Drop     |
| z > 150                | Drop     |
| u < 1250 **and** z > 50 | Drop (if `u` present) |
| Otherwise              | Keep     |

Constants are defined at the top of each script (or next to `load_trajectory_csv`) so they can be adjusted in one place per file. For a single reference, see `docs/TRAJECTORY_FILTERS.md` (this file).
