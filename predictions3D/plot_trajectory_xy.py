#!/usr/bin/env python3
"""
Parse 3D keypoint CSV and plot trajectory projected onto the x-y plane.

CSV format: row 1 = body part names (repeated per x,y,z,confidence),
            row 2 = coordinate labels (x,y,z,confidence),
            row 3+ = numeric data.
"""

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_data3d_csv(csv_path: Path) -> dict[str, pd.DataFrame]:
    """Parse data3D.csv with multi-level header into per-body-part DataFrames."""
    with open(csv_path) as f:
        line1 = f.readline().strip().split(",")
        line2 = f.readline().strip().split(",")
    names = [n.strip() for n in line1]
    coords = [c.strip() for c in line2]

    # Build (body_part, coord) for each column
    # Pattern: Snout,Snout,Snout,Snout, EarL,EarL,EarL,EarL, ...
    #          x,   y,   z,   confidence, x,  y,  z,  confidence, ...
    tuples = list(zip(names, coords))

    # Read data (skip the 2 header rows)
    df = pd.read_csv(csv_path, skiprows=2, header=None)
    df.columns = pd.MultiIndex.from_tuples(tuples)

    # Extract per-body-part DataFrames with x, y, z, confidence
    body_parts = {}
    for part in df.columns.get_level_values(0).unique():
        part_df = df[part].copy()
        part_df.columns = part_df.columns.str.strip()
        body_parts[part] = part_df

    return body_parts


def plot_trajectory_xy(
    body_parts: dict[str, pd.DataFrame],
    *,
    parts: list[str] | None = None,
    frame_start: int = 0,
    color_by_time: bool = True,
    out_path: Path | None = None,
    min_confidence: float | None = None,
) -> None:
    """
    Plot x-y trajectory for selected body parts.

    parts: which body parts to plot (default: all)
    frame_start: added to frame index for x-axis/time (e.g. from info.yaml)
    color_by_time: use time/frame to color the trajectory
    min_confidence: if set, mask out points below this confidence
    """
    if parts is None:
        parts = list(body_parts.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")

    colors = plt.cm.viridis(np.linspace(0, 1, len(parts)))

    for idx, part in enumerate(parts):
        if part not in body_parts:
            print(f"Warning: body part '{part}' not found. Available: {list(body_parts.keys())}")
            continue
        df = body_parts[part]
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        n = len(x)

        if min_confidence is not None and "confidence" in df.columns:
            conf = df["confidence"].values.astype(float)
            mask = conf >= min_confidence
            x, y = x[mask], y[mask]

        if color_by_time and n > 0:
            scatter = ax.scatter(x, y, c=np.arange(n), cmap="viridis", s=4, alpha=0.7)
            # Plot line in same color as last point
            ax.plot(x, y, color=colors[idx], alpha=0.4, linewidth=0.8, label=part)
        else:
            ax.plot(x, y, color=colors[idx], alpha=0.7, linewidth=1, label=part)
            ax.scatter(x[0], y[0], color=colors[idx], s=40, marker="o", zorder=5, edgecolors="k")
            ax.scatter(x[-1], y[-1], color=colors[idx], s=40, marker="s", zorder=5, edgecolors="k")

    if color_by_time and parts:
        plt.colorbar(scatter, ax=ax, label="Frame index")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory (x-y plane)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot 3D keypoints trajectory on x-y plane")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("Predictions_3D_trial_0000_11572-20491/data3D.csv"),
        help="Path to data3D.csv or to a trial folder containing data3D.csv",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=None,
        help="Body parts to plot (default: all). e.g. Snout EarL EarR Tail",
    )
    parser.add_argument(
        "--no-color-time",
        action="store_true",
        help="Do not color trajectory by time; use solid color per part",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum confidence to include a point (0-1)",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Frame start offset (e.g. from info.yaml) for display",
    )
    args = parser.parse_args()

    path = args.input.resolve()
    if path.is_dir():
        path = path / "data3D.csv"
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    body_parts = parse_data3d_csv(path)
    print(f"Body parts: {list(body_parts.keys())}")
    for part, df in body_parts.items():
        print(f"  {part}: {len(df)} frames")

    out = args.output
    if out is None and path.parent != Path.cwd():
        out = path.parent / "trajectory_xy.png"

    plot_trajectory_xy(
        body_parts,
        parts=args.parts,
        frame_start=args.frame_start,
        color_by_time=not args.no_color_time,
        out_path=out,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
