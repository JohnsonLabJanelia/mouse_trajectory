#!/usr/bin/env python3
"""
Analyze cbot climb log files: parse robot_manager.log for each animal,
plot pretraining vs training trial counts and trial duration learning curves.
Trial duration = time between "door: opened" and "door: closed".
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LOGS_DIR = Path(__file__).resolve().parent / "logs"

# Log line format: TIMESTAMP,  robot_manager_logger, info, MESSAGE
# Trial duration = time from "door: opened" to "door: closed"
# Stage: pretraining if message contains "pretraining", else training


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Convert YYYY_MM_DD_HH_MM_SS_mmm to seconds since epoch for duration math."""
    parts = ts_str.strip().split("_")
    if len(parts) != 7:
        return 0.0
    try:
        y, mo, d, h, mi, s, ms = (int(x) for x in parts)
        dt = datetime(y, mo, d, h, mi, s, ms * 1000)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def parse_timestamp(ts_str: str) -> tuple:
    """Convert YYYY_MM_DD_HH_MM_SS_mmm to a sortable tuple (for ordering)."""
    parts = ts_str.strip().split("_")
    if len(parts) != 7:
        return (0,) * 7
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        return (0,) * 7


def _normalize_angle(angle: float) -> float:
    """Bring angle into [-180, 180]. Keep 360 as 360."""
    if angle == 360:
        return 360.0
    return ((angle + 180) % 360) - 180


def _to_positive_angle(angle: float) -> float:
    """Convert angle to [0, 360). Keep 360 as 360. E.g. -90 -> 270, -100 -> 260."""
    if angle == 360:
        return 360.0
    return ((angle % 360) + 360) % 360


def _restrict_to_half_circle(angle: float) -> float:
    """Restrict to [0, 180] (line rotation is symmetrical). Keep 360 as 360 (vertical)."""
    if angle == 360:
        return 360.0
    # Already in [0, 360) from _to_positive_angle. Map (180, 360) -> (0, 180): 270->90, 260->100
    if angle > 180:
        return 360.0 - angle
    return angle


def _parse_trial_angles(message: str) -> tuple[float | None, float | None]:
    """
    Parse 'trial_N: left_angle=X.Xdeg ..., right_angle=Y.Ydeg ...' from training trial line.
    Angles in [0, 360]; 360 = vertical.
    - Parallel: positive kept as-is; negative → 180 + angle (e.g. -10 → 170, -90 → 90).
    - Perpendicular: add 90 to the angle (angle + 90).
    - 360 always stays 360 (vertical). 270 is treated as vertical (360) so all vertical trials align.
    Then wrap to [0, 360) via _to_positive_angle (360 unchanged).
    Returns (left_angle, right_angle) or (None, None) if not found.
    """
    if "left_angle=" not in message or "right_angle=" not in message:
        return None, None
    left_m = re.search(r"left_angle=([-\d.]+)deg", message)
    right_m = re.search(r"right_angle=([-\d.]+)deg", message)
    if not left_m or not right_m:
        return None, None
    left_val = float(left_m.group(1))
    right_val = float(right_m.group(1))
    left_part = message[: right_m.start()] if right_m else message
    right_part = message[right_m.start() :] if right_m else ""
    left_is_parallel = "(parallel)" in left_part
    right_is_parallel = "(parallel)" in right_part

    def _to_convention(val: float, is_parallel: bool) -> float:
        if val == 360:
            return 360.0
        if is_parallel:
            return val if val >= 0 else (180.0 + val)
        return val + 90  # perpendicular: add 90

    left_par = _to_convention(left_val, left_is_parallel)
    right_par = _to_convention(right_val, right_is_parallel)
    left_par = _to_positive_angle(left_par)
    right_par = _to_positive_angle(right_par)
    # Treat 270 as vertical (same as 360): e.g. 180° perpendicular → 270, but vertical should be one bin
    if left_par == 270:
        left_par = 360.0
    if right_par == 270:
        right_par = 360.0
    return left_par, right_par


def parse_robot_manager_log(path: Path) -> tuple[list[tuple], list[tuple]]:
    """
    Parse a single robot_manager.log file.
    Trial duration = time between "door: opened" and "door: closed".
    time_to_reward = time between "reward:" and "door: closed".
    time_to_target = time between "door: opened" and "reward:".
    For training trials only: left_angle_parallel, right_angle_parallel (in [0, 360) or 360).
    Returns (pretraining_trials, training_trials). Pretraining: 5-tuple + (None, None). Training: 5-tuple + (left_par, right_par).
    """
    pretraining = []
    training = []
    # Pending "door: opened": (ts_tuple, ts_seconds, is_pretraining)
    door_opened: tuple[tuple, float, bool] | None = None
    reward_ts: float | None = None
    time_to_target: float | None = None
    # Training trial type: set when we see "trial_N: left_angle=... right_angle=..." (angles in parallel)
    current_trial_left_par: float | None = None
    current_trial_right_par: float | None = None

    with open(path, "r") as f:
        for line in f:
            if ",  robot_manager_logger, info, " not in line:
                continue
            parts = line.split(",  robot_manager_logger, info, ", 1)
            if len(parts) != 2:
                continue
            ts_str, message = parts[0].strip(), parts[1].strip()
            ts_tuple = parse_timestamp(ts_str)
            ts_sec = parse_timestamp_to_seconds(ts_str)

            # Training trial config line: "trial_N: left_angle=... right_angle=..."
            if re.match(r"trial_\d+:\s*left_angle=", message) and "pretraining" not in message:
                left_p, right_p = _parse_trial_angles(message)
                if left_p is not None and right_p is not None:
                    current_trial_left_par, current_trial_right_par = left_p, right_p
            if message.startswith("door: opened"):
                is_pretraining = "pretraining" in message
                door_opened = (ts_tuple, ts_sec, is_pretraining)
                reward_ts = None
                time_to_target = None
            elif message.startswith("reward:"):
                reward_ts = ts_sec
                if door_opened is not None:
                    _, opened_sec, _ = door_opened
                    time_to_target = reward_ts - opened_sec
            elif message.strip() == "door: closed" and door_opened is not None:
                _, opened_sec, is_pretraining = door_opened
                duration_sec = ts_sec - opened_sec
                time_to_reward = (ts_sec - reward_ts) if reward_ts is not None else None
                if is_pretraining:
                    trial_idx = len(pretraining)
                    pretraining.append((ts_tuple, trial_idx, duration_sec, time_to_reward, time_to_target, None, None))
                else:
                    trial_idx = len(training)
                    training.append((ts_tuple, trial_idx, duration_sec, time_to_reward, time_to_target, current_trial_left_par, current_trial_right_par))
                door_opened = None
                reward_ts = None
                time_to_target = None

    return pretraining, training


def collect_all_sessions(logs_dir: Path):
    """
    For each animal, find all session robot_manager.log files (sorted by path/session date),
    parse them, and aggregate counts and trial durations in chronological order.
    Trial numbers are assigned globally by session time: all trials (pretraining + training)
    are sorted chronologically, then assigned trial_id 0, 1, 2, ...
    Returns:
        animal_data: dict[animal_name] = {
            "n_pretraining": int,
            "n_training": int,
            "trials": [(trial_id, duration_sec, stage, ttr, ttt, left_angle_parallel, right_angle_parallel), ...],
        }
    """
    animal_data = defaultdict(lambda: {
        "n_pretraining": 0,
        "n_training": 0,
        "trials": [],
    })

    if not logs_dir.is_dir():
        return dict(animal_data)

    for animal_dir in sorted(logs_dir.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        # Collect all session log paths and sort by session datetime (path name)
        session_logs = []
        for session_dir in animal_dir.iterdir():
            if not session_dir.is_dir():
                continue
            log_path = session_dir / "robot_manager.log"
            if log_path.is_file():
                session_logs.append(log_path)
        session_logs.sort(key=lambda p: p.parent.name)

        # Collect all trials with (timestamp_tuple, duration, stage, ttr, ttt, left_perp, right_perp)
        all_trials_raw = []
        for log_path in session_logs:
            pre, train = parse_robot_manager_log(log_path)
            for ts, _, d, ttr, ttt, lperp, rperp in pre:
                all_trials_raw.append((ts, d, "pretraining", ttr, ttt, lperp, rperp))
            for ts, _, d, ttr, ttt, lperp, rperp in train:
                all_trials_raw.append((ts, d, "training", ttr, ttt, lperp, rperp))

        # Sort by session time (chronological order), then assign global trial_id
        all_trials_raw.sort(key=lambda x: x[0])
        trials = [(i, t[1], t[2], t[3], t[4], t[5], t[6]) for i, t in enumerate(all_trials_raw)]

        n_pre = sum(1 for t in trials if t[2] == "pretraining")
        n_train = sum(1 for t in trials if t[2] == "training")

        animal_data[animal]["n_pretraining"] = n_pre
        animal_data[animal]["n_training"] = n_train
        animal_data[animal]["trials"] = trials

    return dict(animal_data)


def plot_pretraining_vs_training(animal_data: dict, out_path: Path | None = None):
    """Bar chart: number of trials in pretraining vs training for each animal."""
    animals = sorted(animal_data.keys())
    n_pre = [animal_data[a]["n_pretraining"] for a in animals]
    n_train = [animal_data[a]["n_training"] for a in animals]

    x = np.arange(len(animals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, n_pre, width, label="Pretraining", color="steelblue")
    bars2 = ax.bar(x + width / 2, n_train, width, label="Training", color="coral")

    ax.set_ylabel("Number of trials")
    ax.set_title("Pretraining vs training trial counts per animal")
    ax.set_xticks(x)
    ax.set_xticklabels(animals)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b in bars1:
        ax.annotate(str(b.get_height()), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    for b in bars2:
        ax.annotate(str(b.get_height()), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_learning_curves(animal_data: dict, out_path: Path | None = None):
    """Plot trial duration vs trial number (learning curve). Trial number is chronological across sessions; pretraining vs training colored separately."""
    animals = sorted(animal_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    for idx, animal in enumerate(animals):
        ax = axes[idx]
        trials = animal_data[animal]["trials"]
        # Line: connect all trials in chronological order (consecutive trial ids)
        trial_ids = [t[0] for t in trials]
        durations = [t[1] for t in trials]
        ax.plot(trial_ids, durations, color="gray", linewidth=0.8, alpha=0.6, zorder=0)
        # Scatter: color by trial type
        pre = [(t[0], t[1]) for t in trials if t[2] == "pretraining"]
        train = [(t[0], t[1]) for t in trials if t[2] == "training"]
        if pre:
            nums = [t[0] for t in pre]
            durs = [t[1] for t in pre]
            ax.scatter(nums, durs, alpha=0.7, s=4, label="Pretraining", color="steelblue", zorder=1)
        if train:
            nums = [t[0] for t in train]
            durs = [t[1] for t in train]
            ax.scatter(nums, durs, alpha=0.7, s=4, label="Training", color="coral", zorder=1)

        ax.set_xlabel("Trial number (chronological)")
        ax.set_ylabel("Trial duration (s)")
        ax.set_title(animal)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot if odd number of animals
    for j in range(len(animals), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Learning curve: trial duration vs trial number", y=1.02)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


# Animals to include on the combined first-100 training plot (order and colors)
FIRST_100_TRAINING_ANIMALS = ["mickey", "rory", "wilfred"]
FIRST_100_TRAINING_COLORS = ["C0", "C1", "C2"]


def plot_learning_curves_first_100_training(animal_data: dict, out_path: Path | None = None):
    """Single plot: first 100 training trials only; mean across 3 animals with SEM error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    max_trials = 100

    # Collect first 100 training durations per animal (training only, no pretraining)
    rows = []
    for animal in FIRST_100_TRAINING_ANIMALS:
        if animal not in animal_data:
            rows.append(np.full(max_trials, np.nan))
            continue
        trials = animal_data[animal]["trials"]
        training = [t[1] for t in trials if t[2] == "training"]  # duration only
        first_100 = training[:max_trials]
        arr = np.full(max_trials, np.nan)
        arr[: len(first_100)] = first_100
        rows.append(arr)

    if not rows:
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        return

    # Stack: (n_animals, max_trials)
    durations = np.array(rows)
    n = durations.shape[1]
    trial_index = np.arange(n)

    mean = np.nanmean(durations, axis=0)
    sem = stats.sem(durations, axis=0, nan_policy="omit")
    # Where only one animal has data, SEM is NaN; use 0 for error bar
    sem = np.where(np.isnan(sem), 0.0, sem)

    # Points only (no connecting lines), with error bars
    ax.errorbar(
        trial_index,
        mean,
        yerr=sem,
        color="C0",
        fmt="o",
        markersize=4,
        capsize=2,
        capthick=1,
        label="Mean (n=3)",
        linestyle="none",
    )

    # Fit a smooth curve to the points (polynomial, degree 3)
    valid = ~np.isnan(mean)
    if np.sum(valid) >= 4:
        x_fit = trial_index[valid]
        y_fit = mean[valid]
        coeffs = np.polyfit(x_fit, y_fit, deg=3)
        x_smooth = np.linspace(0, n - 1, 200)
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_smooth, color="C1", linewidth=2, label="Fit (cubic)")

    ax.set_xlabel("Trial number (first 100 training trials)")
    ax.set_ylabel("Trial duration (s)")
    ax.set_title("Learning curve: trial duration vs trial (training only, first 100; mean ± SEM)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def _plot_first_100_training_metric(
    ax,
    animal_data: dict,
    value_index: int,
    ylabel: str,
    title: str,
    show_xlabel: bool = True,
    max_trials: int = 100,
):
    """Shared logic: first N training trials, mean ± SEM as points, cubic fit. value_index: 1=duration, 3=ttr, 4=ttt."""
    rows = []
    for animal in FIRST_100_TRAINING_ANIMALS:
        if animal not in animal_data:
            rows.append(np.full(max_trials, np.nan))
            continue
        trials = animal_data[animal]["trials"]
        training = [t for t in trials if t[2] == "training"]
        first_n = training[:max_trials]
        arr = np.full(max_trials, np.nan)
        for i, t in enumerate(first_n):
            val = t[value_index]
            arr[i] = val if val is not None else np.nan
        rows.append(arr)

    if not rows:
        return
    data = np.array(rows)
    n = data.shape[1]
    trial_index = np.arange(n)
    mean = np.nanmean(data, axis=0)
    sem = stats.sem(data, axis=0, nan_policy="omit")
    sem = np.where(np.isnan(sem), 0.0, sem)

    ax.errorbar(
        trial_index,
        mean,
        yerr=sem,
        color="C0",
        fmt="o",
        markersize=4,
        capsize=2,
        capthick=1,
        label="Mean (n=3)",
        linestyle="none",
    )
    valid = ~np.isnan(mean)
    if np.sum(valid) >= 4:
        x_fit = trial_index[valid]
        y_fit = mean[valid]
        coeffs = np.polyfit(x_fit, y_fit, deg=3)
        x_smooth = np.linspace(0, n - 1, 200)
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_smooth, color="C1", linewidth=2, label="Fit (cubic)")
    if show_xlabel:
        ax.set_xlabel(f"Trial number (first {max_trials} training trials)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def plot_learning_curves_first_100_training_combined(animal_data: dict, out_path: Path | None = None):
    """Single figure: 3 panels stacked (duration, time to target, time to reward), shared x-axis. Saves EPS (primary) and high-res PNG."""
    max_trials = 100
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    _plot_first_100_training_metric(
        axes[0],
        animal_data,
        value_index=1,  # duration
        ylabel="Trial duration (s)",
        title=f"Trial duration (training only, first {max_trials}; mean ± SEM)",
        show_xlabel=False,
        max_trials=max_trials,
    )
    _plot_first_100_training_metric(
        axes[1],
        animal_data,
        value_index=4,  # ttt
        ylabel="Time to target (s)",
        title=f"Time to target (training only, first {max_trials}; mean ± SEM)",
        show_xlabel=False,
        max_trials=max_trials,
    )
    _plot_first_100_training_metric(
        axes[2],
        animal_data,
        value_index=3,  # ttr
        ylabel="Time to reward (s)",
        title=f"Time to reward (training only, first {max_trials}; mean ± SEM)",
        show_xlabel=True,
        max_trials=max_trials,
    )
    # Larger fonts for axis labels, titles, ticks, and legend
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14
    for ax in axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
        ax.set_title(ax.get_title(), fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(legend_fontsize)
    plt.suptitle(f"Learning curves (first {max_trials} training trials)", y=1.02, fontsize=18)
    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        base = out_path.parent / out_path.stem
        # Primary: EPS (vector, high resolution)
        eps_path = base.with_suffix(".eps")
        plt.savefig(eps_path, format="eps", bbox_inches="tight")
        print(f"Saved {eps_path}")
        # Also save high-res PNG
        png_path = base.with_suffix(".png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved {png_path}")
    else:
        plt.show()


def plot_learning_curves_first_100_training_time_to_target(animal_data: dict, out_path: Path | None = None):
    """First 100 training trials: time to target (door open → reward), points + cubic fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_first_100_training_metric(
        ax,
        animal_data,
        value_index=4,  # ttt
        ylabel="Time to target (s)",
        title="Learning curve: time to target vs trial (training only, first 100; mean ± SEM)",
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_learning_curves_first_100_training_time_to_reward(animal_data: dict, out_path: Path | None = None):
    """First 100 training trials: time to reward (reward → door closed), points + cubic fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_first_100_training_metric(
        ax,
        animal_data,
        value_index=3,  # ttr
        ylabel="Time to reward (s)",
        title="Learning curve: time to reward vs trial (training only, first 100; mean ± SEM)",
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_time_to_reward(animal_data: dict, out_path: Path | None = None):
    """Plot time_to_reward (reward → door closed) vs trial number for each animal."""
    animals = sorted(animal_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    for idx, animal in enumerate(animals):
        ax = axes[idx]
        trials = animal_data[animal]["trials"]
        # Only trials with valid time_to_reward (5-tuple: trial_id, duration, stage, ttr, ttt)
        with_ttr = [(t[0], t[3], t[2]) for t in trials if t[3] is not None]
        if not with_ttr:
            ax.set_title(animal)
            ax.set_xlabel("Trial number (chronological)")
            ax.set_ylabel("Time to reward (s)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3)
            continue

        trial_ids = [x[0] for x in with_ttr]
        ttrs = [x[1] for x in with_ttr]
        ax.plot(trial_ids, ttrs, color="gray", linewidth=0.8, alpha=0.6, zorder=0)
        pre = [(x[0], x[1]) for x in with_ttr if x[2] == "pretraining"]
        train = [(x[0], x[1]) for x in with_ttr if x[2] == "training"]
        if pre:
            ax.scatter([p[0] for p in pre], [p[1] for p in pre], alpha=0.7, s=4, label="Pretraining", color="steelblue", zorder=1)
        if train:
            ax.scatter([p[0] for p in train], [p[1] for p in train], alpha=0.7, s=4, label="Training", color="coral", zorder=1)

        ax.set_xlabel("Trial number (chronological)")
        ax.set_ylabel("Time to reward (s)")
        ax.set_title(animal)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    for j in range(len(animals), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Time to reward (reward → door closed) vs trial number", y=1.02)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_time_to_target(animal_data: dict, out_path: Path | None = None):
    """Plot time_to_target (door opened → reward) vs trial number for each animal."""
    animals = sorted(animal_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    for idx, animal in enumerate(animals):
        ax = axes[idx]
        trials = animal_data[animal]["trials"]
        # Only trials with valid time_to_target (5-tuple: trial_id, duration, stage, ttr, ttt)
        with_ttt = [(t[0], t[4], t[2]) for t in trials if t[4] is not None]
        if not with_ttt:
            ax.set_title(animal)
            ax.set_xlabel("Trial number (chronological)")
            ax.set_ylabel("Time to target (s)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3)
            continue

        trial_ids = [x[0] for x in with_ttt]
        ttts = [x[1] for x in with_ttt]
        ax.plot(trial_ids, ttts, color="gray", linewidth=0.8, alpha=0.6, zorder=0)
        pre = [(x[0], x[1]) for x in with_ttt if x[2] == "pretraining"]
        train = [(x[0], x[1]) for x in with_ttt if x[2] == "training"]
        if pre:
            ax.scatter([p[0] for p in pre], [p[1] for p in pre], alpha=0.7, s=4, label="Pretraining", color="steelblue", zorder=1)
        if train:
            ax.scatter([p[0] for p in train], [p[1] for p in train], alpha=0.7, s=4, label="Training", color="coral", zorder=1)

        ax.set_xlabel("Trial number (chronological)")
        ax.set_ylabel("Time to target (s)")
        ax.set_title(animal)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    for j in range(len(animals), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Time to target (door opened → reward) vs trial number", y=1.02)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def _heatmap_shared_grid(animal_data: dict, get_cell_data_fn):
    """
    get_cell_data_fn(animal, trials) -> dict[(left_r, right_r)] -> list of values.
    Returns (animals, left_vals, right_vals, per_animal_cells, left_display, right_display, right_edges, left_edges, _tick_label).
    """
    animals = sorted(animal_data.keys())
    all_left: set[float] = set()
    all_right: set[float] = set()
    per_animal_cells: dict[str, dict[tuple[float, float], list[float]]] = {}
    for animal in animals:
        trials = animal_data[animal]["trials"]
        cell_data = get_cell_data_fn(animal, trials)
        per_animal_cells[animal] = cell_data
        for (left_r, right_r) in cell_data:
            all_left.add(left_r)
            all_right.add(right_r)
    left_vals = sorted(all_left)
    right_vals = sorted(all_right)
    if not left_vals or not right_vals:
        return animals, left_vals, right_vals, per_animal_cells, None, None, None, None, None

    def _display_positions(vals: list[float]) -> list[float]:
        others = [v for v in vals if v != 360]
        if not others:
            return list(vals)
        max_other = max(others)
        min_other = min(others)
        gap = max(15, (max_other - min_other) * 0.15) if max_other != min_other else 15
        return [v if v != 360 else max_other + gap for v in vals]

    def _tick_label(v: float) -> str:
        return "vertical" if v == 360 else (str(int(v)) if v == int(v) else f"{v:.1f}")

    def _bin_edges_centered(disp_vals: list[float]) -> list[float]:
        n = len(disp_vals)
        if n == 0:
            return []
        if n == 1:
            return [disp_vals[0] - 0.5, disp_vals[0] + 0.5]
        edges = [disp_vals[0] - (disp_vals[1] - disp_vals[0]) / 2]
        for j in range(1, n):
            edges.append(2 * disp_vals[j - 1] - edges[j - 1])
        edges.append(2 * disp_vals[-1] - edges[-1])
        return edges

    left_display = _display_positions(left_vals)
    right_display = _display_positions(right_vals)
    right_edges = _bin_edges_centered(right_display)
    left_edges = _bin_edges_centered(left_display)
    return animals, left_vals, right_vals, per_animal_cells, left_display, right_display, right_edges, left_edges, _tick_label


def plot_angle_heatmap(animal_data: dict, out_path: Path | None = None):
    """Heatmap: left_angle (y) vs right_angle (x), value = mean trial duration. Shared axes across animals; includes 360 (vertical)."""
    def get_cells(animal: str, trials: list) -> dict[tuple[float, float], list[float]]:
        training_with_angles = [(t[1], t[5], t[6]) for t in trials if t[2] == "training" and t[5] is not None and t[6] is not None]
        cell_durations: dict[tuple[float, float], list[float]] = defaultdict(list)
        for duration, left, right in training_with_angles:
            cell_durations[(round(left, 1), round(right, 1))].append(duration)
        return cell_durations

    result = _heatmap_shared_grid(animal_data, get_cells)
    animals, left_vals, right_vals, per_animal_cells = result[0], result[1], result[2], result[3]
    if result[4] is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for idx, ax in enumerate(axes.flatten()):
            ax.set_visible(idx < len(animals))
            if idx < len(animals):
                ax.set_title(animals[idx])
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        else:
            plt.show()
        return

    left_display, right_display, right_edges, left_edges, _tick_label = result[4], result[5], result[6], result[7], result[8]
    left_to_i = {v: i for i, v in enumerate(left_vals)}
    right_to_j = {v: j for j, v in enumerate(right_vals)}
    max_ticks = 10
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for idx, animal in enumerate(animals):
        ax = axes[idx]
        cell_durations = per_animal_cells[animal]
        matrix = np.full((len(left_vals), len(right_vals)), np.nan)
        for (left, right), durs in cell_durations.items():
            i = left_to_i.get(left)
            j = right_to_j.get(right)
            if i is not None and j is not None and durs:
                matrix[i, j] = np.mean(durs)
        im = ax.pcolormesh(right_edges, left_edges, matrix, cmap="coolwarm", shading="flat")
        ax.set_xlim(right_edges[0], right_edges[-1])
        ax.set_ylim(left_edges[0], left_edges[-1])
        if len(right_vals) <= max_ticks:
            ax.set_xticks(right_display)
            ax.set_xticklabels([_tick_label(v) for v in right_vals])
        else:
            step = max(1, len(right_vals) // max_ticks)
            ax.set_xticks([right_display[i] for i in range(0, len(right_vals), step)])
            ax.set_xticklabels([_tick_label(right_vals[i]) for i in range(0, len(right_vals), step)])
        if len(left_vals) <= max_ticks:
            ax.set_yticks(left_display)
            ax.set_yticklabels([_tick_label(v) for v in left_vals])
        else:
            step = max(1, len(left_vals) // max_ticks)
            ax.set_yticks([left_display[i] for i in range(0, len(left_vals), step)])
            ax.set_yticklabels([_tick_label(left_vals[i]) for i in range(0, len(left_vals), step)])
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Right angle (parallel)")
        ax.set_ylabel("Left angle (parallel)")
        ax.set_title(animal)
        plt.colorbar(im, ax=ax, label="Mean trial duration (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(len(animals), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Training: mean trial duration by left vs right angle (parallel)", y=1.02)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_angle_heatmap_time_to_target(animal_data: dict, out_path: Path | None = None):
    """Heatmap: left_angle (y) vs right_angle (x), value = mean time_to_target. Shared axes across animals; includes 360 (vertical)."""
    def get_cells(animal: str, trials: list) -> dict[tuple[float, float], list[float]]:
        training_with_angles = [(t[4], t[5], t[6]) for t in trials if t[2] == "training" and t[5] is not None and t[6] is not None and t[4] is not None]
        cell_ttt: dict[tuple[float, float], list[float]] = defaultdict(list)
        for ttt, left, right in training_with_angles:
            cell_ttt[(round(left, 1), round(right, 1))].append(ttt)
        return cell_ttt

    result = _heatmap_shared_grid(animal_data, get_cells)
    animals, left_vals, right_vals, per_animal_cells = result[0], result[1], result[2], result[3]
    if result[4] is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for idx, ax in enumerate(axes.flatten()):
            ax.set_visible(idx < len(animals))
            if idx < len(animals):
                ax.set_title(animals[idx])
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        else:
            plt.show()
        return

    left_display, right_display, right_edges, left_edges, _tick_label = result[4], result[5], result[6], result[7], result[8]
    left_to_i = {v: i for i, v in enumerate(left_vals)}
    right_to_j = {v: j for j, v in enumerate(right_vals)}
    max_ticks = 10
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for idx, animal in enumerate(animals):
        ax = axes[idx]
        cell_ttt = per_animal_cells[animal]
        matrix = np.full((len(left_vals), len(right_vals)), np.nan)
        for (left, right), ttts in cell_ttt.items():
            i = left_to_i.get(left)
            j = right_to_j.get(right)
            if i is not None and j is not None and ttts:
                matrix[i, j] = np.mean(ttts)
        im = ax.pcolormesh(right_edges, left_edges, matrix, cmap="coolwarm", shading="flat")
        ax.set_xlim(right_edges[0], right_edges[-1])
        ax.set_ylim(left_edges[0], left_edges[-1])
        if len(right_vals) <= max_ticks:
            ax.set_xticks(right_display)
            ax.set_xticklabels([_tick_label(v) for v in right_vals])
        else:
            step = max(1, len(right_vals) // max_ticks)
            ax.set_xticks([right_display[i] for i in range(0, len(right_vals), step)])
            ax.set_xticklabels([_tick_label(right_vals[i]) for i in range(0, len(right_vals), step)])
        if len(left_vals) <= max_ticks:
            ax.set_yticks(left_display)
            ax.set_yticklabels([_tick_label(v) for v in left_vals])
        else:
            step = max(1, len(left_vals) // max_ticks)
            ax.set_yticks([left_display[i] for i in range(0, len(left_vals), step)])
            ax.set_yticklabels([_tick_label(left_vals[i]) for i in range(0, len(left_vals), step)])
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Right angle (parallel)")
        ax.set_ylabel("Left angle (parallel)")
        ax.set_title(animal)
        plt.colorbar(im, ax=ax, label="Mean time to target (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(len(animals), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Training: mean time to target (door opened → reward) by left vs right angle", y=1.02)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_angle_heatmap_time_to_target_median(animal_data: dict, out_path: Path | None = None):
    """Heatmap: left_angle (y) vs right_angle (x), value = median time_to_target. Shared axes across animals; includes 360 (vertical)."""
    def get_cells(animal: str, trials: list) -> dict[tuple[float, float], list[float]]:
        training_with_angles = [(t[4], t[5], t[6]) for t in trials if t[2] == "training" and t[5] is not None and t[6] is not None and t[4] is not None]
        cell_ttt: dict[tuple[float, float], list[float]] = defaultdict(list)
        for ttt, left, right in training_with_angles:
            cell_ttt[(round(left, 1), round(right, 1))].append(ttt)
        return cell_ttt

    result = _heatmap_shared_grid(animal_data, get_cells)
    animals, left_vals, right_vals, per_animal_cells = result[0], result[1], result[2], result[3]
    if result[4] is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for idx, ax in enumerate(axes.flatten()):
            ax.set_visible(idx < len(animals))
            if idx < len(animals):
                ax.set_title(animals[idx])
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        else:
            plt.show()
        return

    left_display, right_display, right_edges, left_edges, _tick_label = result[4], result[5], result[6], result[7], result[8]
    left_to_i = {v: i for i, v in enumerate(left_vals)}
    right_to_j = {v: j for j, v in enumerate(right_vals)}
    max_ticks = 10
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for idx, animal in enumerate(animals):
        ax = axes[idx]
        cell_ttt = per_animal_cells[animal]
        matrix = np.full((len(left_vals), len(right_vals)), np.nan)
        for (left, right), ttts in cell_ttt.items():
            i = left_to_i.get(left)
            j = right_to_j.get(right)
            if i is not None and j is not None and ttts:
                matrix[i, j] = np.median(ttts)
        im = ax.pcolormesh(right_edges, left_edges, matrix, cmap="coolwarm", shading="flat")
        ax.set_xlim(right_edges[0], right_edges[-1])
        ax.set_ylim(left_edges[0], left_edges[-1])
        if len(right_vals) <= max_ticks:
            ax.set_xticks(right_display)
            ax.set_xticklabels([_tick_label(v) for v in right_vals])
        else:
            step = max(1, len(right_vals) // max_ticks)
            ax.set_xticks([right_display[i] for i in range(0, len(right_vals), step)])
            ax.set_xticklabels([_tick_label(right_vals[i]) for i in range(0, len(right_vals), step)])
        if len(left_vals) <= max_ticks:
            ax.set_yticks(left_display)
            ax.set_yticklabels([_tick_label(v) for v in left_vals])
        else:
            step = max(1, len(left_vals) // max_ticks)
            ax.set_yticks([left_display[i] for i in range(0, len(left_vals), step)])
            ax.set_yticklabels([_tick_label(left_vals[i]) for i in range(0, len(left_vals), step)])
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Right angle (parallel)")
        ax.set_ylabel("Left angle (parallel)")
        ax.set_title(animal)
        plt.colorbar(im, ax=ax, label="Median time to target (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(len(animals), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Training: median time to target (door opened → reward) by left vs right angle", y=1.02)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_angle_diff_vs_median_ttt(animal_data: dict, out_path: Path | None = None):
    """Single plot: angle_diff (right - left) vs median time to target, pooled across all 4 animals.
    Shows which angle combination is hardest overall; per-animal curves in lighter colors."""
    animals = sorted(animal_data.keys())
    # Collect (angle_diff, time_to_target) for training trials only
    # angle_diff = right - left (can be negative)
    all_pairs: list[tuple[float, float]] = []
    per_animal: dict[str, list[tuple[float, float]]] = {}

    for animal in animals:
        trials = animal_data[animal]["trials"]
        pairs = [
            (round(t[6] - t[5], 1), t[4])
            for t in trials
            if t[2] == "training" and t[5] is not None and t[6] is not None and t[4] is not None
        ]
        per_animal[animal] = pairs
        all_pairs.extend(pairs)

    if not all_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Angle diff (right − left, deg)")
        ax.set_ylabel("Median time to target (s)")
        ax.set_title("Angle diff vs median time to target (all animals) — no training data")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        else:
            plt.show()
        return

    # Bin by angle_diff and compute median TTT
    cell_ttt: dict[float, list[float]] = defaultdict(list)
    for angle_diff, ttt in all_pairs:
        cell_ttt[angle_diff].append(ttt)

    angle_diffs = sorted(cell_ttt.keys())
    medians_all = [np.median(cell_ttt[ad]) for ad in angle_diffs]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Main curve: all animals pooled
    ax.plot(angle_diffs, medians_all, "o-", color="black", linewidth=2, markersize=6, label="All animals (median TTT)")

    # Per-animal curves (lighter)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(animals), 1)))
    for idx, animal in enumerate(animals):
        pairs = per_animal[animal]
        if not pairs:
            continue
        cell_a: dict[float, list[float]] = defaultdict(list)
        for ad, ttt in pairs:
            cell_a[ad].append(ttt)
        ad_vals = sorted(cell_a.keys())
        med_a = [np.median(cell_a[ad]) for ad in ad_vals]
        ax.plot(ad_vals, med_a, "o-", color=colors[idx % len(colors)], alpha=0.6, linewidth=1, markersize=3, label=animal)

    ax.set_xlabel("Angle diff (right − left, deg)")
    ax.set_ylabel("Median time to target (s)")
    ax.set_title("Which angle combination is hardest? Angle diff vs median time to target (all 4 animals)")
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def _permutation_test_medians(sample_a: list[float], sample_b: list[float], n_perm: int = 10_000) -> float | None:
    """Two-sided permutation test: is median(sample_a) different from median(sample_b)? Returns p-value or None."""
    if not sample_a or not sample_b:
        return None
    observed = np.median(sample_a) - np.median(sample_b)
    pooled = np.array(sample_a + sample_b)
    n_a = len(sample_a)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        med_a = np.median(pooled[:n_a])
        med_b = np.median(pooled[n_a:])
        if abs(med_a - med_b) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def plot_vertical_vs_angle_median_ttt(animal_data: dict, out_path: Path | None = None):
    """Single plot: median time to target for trials with one side vertical (360°) and the other 0–360°.
    X = non-vertical angle (0–360); includes vertical vs vertical at x=360. Error bars = IQR (Q1–Q3).
    Only rory, wilfred, and mickey (jack excluded)."""
    ANGLE_MAX = 359  # one vertical + other 0–359
    animals = sorted(a for a in animal_data.keys() if a in ("rory", "wilfred", "mickey"))
    # (angle, ttt): angle = non-vertical side in [0, 359], or 360 for vertical vs vertical
    all_pairs: list[tuple[float, float]] = []

    for animal in animals:
        trials = animal_data[animal]["trials"]
        for t in trials:
            if t[2] != "training" or t[5] is None or t[6] is None or t[4] is None:
                continue
            left, right, ttt = t[5], t[6], t[4]
            if left == 360 and right == 360:
                all_pairs.append((360.0, ttt))  # vertical vs vertical at x=360
                continue
            if left == 360 and 0 <= right <= ANGLE_MAX:
                all_pairs.append((round(right, 1), ttt))
            elif right == 360 and 0 <= left <= ANGLE_MAX:
                all_pairs.append((round(left, 1), ttt))

    if not all_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Angle (deg, non-vertical side)")
        ax.set_ylabel("Median time to target (s)")
        ax.set_title("Vertical (360°) vs angle: median TTT (rory + wilfred) — no matching trials")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        else:
            plt.show()
        return

    cell_ttt: dict[float, list[float]] = defaultdict(list)
    for angle, ttt in all_pairs:
        cell_ttt[angle].append(ttt)

    angles = sorted(cell_ttt.keys())
    medians_all = [np.median(cell_ttt[a]) for a in angles]
    # Error bars: IQR (25th–75th percentile). yerr_lower = median - Q1, yerr_upper = Q3 - median
    q1 = [np.percentile(cell_ttt[a], 25) for a in angles]
    q3 = [np.percentile(cell_ttt[a], 75) for a in angles]
    yerr_lower = [medians_all[i] - q1[i] for i in range(len(angles))]
    yerr_upper = [q3[i] - medians_all[i] for i in range(len(angles))]

    # Test: (360 vs 0) vs (360 vs 90) — is *median* TTT different? (permutation test on medians)
    ttt_0 = cell_ttt.get(0.0) or cell_ttt.get(0)
    ttt_90 = cell_ttt.get(90.0) or cell_ttt.get(90)
    sig_0_vs_90 = False
    p_val_0_vs_90: float | None = None
    if ttt_0 and ttt_90 and len(ttt_0) >= 1 and len(ttt_90) >= 1:
        p_val_0_vs_90 = _permutation_test_medians(ttt_0, ttt_90)
        if p_val_0_vs_90 is not None:
            sig_0_vs_90 = p_val_0_vs_90 < 0.05
    if p_val_0_vs_90 is not None:
        print(f"Vertical vs angle (median): (360 vs 0) vs (360 vs 90) [permutation test on medians], p = {p_val_0_vs_90:.4f}" + (" (*)" if sig_0_vs_90 else ""))

    # Test: (360 vs 10) vs (360 vs 100) — is *median* TTT different? (permutation test on medians)
    ttt_10 = cell_ttt.get(10.0) or cell_ttt.get(10)
    ttt_100 = cell_ttt.get(100.0) or cell_ttt.get(100)
    sig_10_vs_100 = False
    p_val_10_vs_100: float | None = None
    if ttt_10 and ttt_100 and len(ttt_10) >= 1 and len(ttt_100) >= 1:
        p_val_10_vs_100 = _permutation_test_medians(ttt_10, ttt_100)
        if p_val_10_vs_100 is not None:
            sig_10_vs_100 = p_val_10_vs_100 < 0.05
    if p_val_10_vs_100 is not None:
        print(f"Vertical vs angle (median): (360 vs 10) vs (360 vs 100) [permutation test on medians], p = {p_val_10_vs_100:.4f}" + (" (*)" if sig_10_vs_100 else ""))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        angles,
        medians_all,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color="black",
        linewidth=2,
        markersize=6,
        capsize=3,
        capthick=1.5,
    )

    y_range = max(medians_all) - min(m - yl for m, yl in zip(medians_all, yerr_lower))
    if sig_0_vs_90 and (0 in angles or 0.0 in angles) and (90 in angles or 90.0 in angles):
        idx_0 = angles.index(0.0) if 0.0 in angles else angles.index(0)
        idx_90 = angles.index(90.0) if 90.0 in angles else angles.index(90)
        y_0 = medians_all[idx_0] + yerr_upper[idx_0]
        y_90 = medians_all[idx_90] + yerr_upper[idx_90]
        y_bracket = max(y_0, y_90) + 0.06 * y_range
        ax.plot([0, 0], [y_0, y_bracket], color="black", linewidth=1)
        ax.plot([90, 90], [y_90, y_bracket], color="black", linewidth=1)
        ax.plot([0, 90], [y_bracket, y_bracket], color="black", linewidth=1)
        ax.annotate("*", xy=(45, y_bracket), ha="center", va="bottom", fontsize=16)
    if sig_10_vs_100 and (10 in angles or 10.0 in angles) and (100 in angles or 100.0 in angles):
        idx_10 = angles.index(10.0) if 10.0 in angles else angles.index(10)
        idx_100 = angles.index(100.0) if 100.0 in angles else angles.index(100)
        y_10 = medians_all[idx_10] + yerr_upper[idx_10]
        y_100 = medians_all[idx_100] + yerr_upper[idx_100]
        y_bracket = max(y_10, y_100) + 0.06 * y_range
        if sig_0_vs_90 and (0 in angles or 0.0 in angles):
            y_bracket += 0.08 * y_range  # stack above 0–90 bracket
        ax.plot([10, 10], [y_10, y_bracket], color="black", linewidth=1)
        ax.plot([100, 100], [y_100, y_bracket], color="black", linewidth=1)
        ax.plot([10, 100], [y_bracket, y_bracket], color="black", linewidth=1)
        ax.annotate("*", xy=(55, y_bracket), ha="center", va="bottom", fontsize=16)

    ax.set_xlabel("Angle (deg, non-vertical side; vertical = 360°)")
    ax.set_ylabel("Median time to target (s)")
    ax.set_title("Vertical (360°) vs angle 0–360° (incl. vertical vs vertical at 360); median TTT; error bars = IQR")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def plot_vertical_vs_angle_mean_ttt(animal_data: dict, out_path: Path | None = None):
    """Single plot: mean time to target for trials with one side vertical (360°) and the other 0–360°.
    X = non-vertical angle (0–360); includes vertical vs vertical at x=360. Error bars = SEM.
    Only rory, wilfred, and mickey (jack excluded)."""
    ANGLE_MAX = 359  # one vertical + other 0–359
    animals = sorted(a for a in animal_data.keys() if a in ("rory", "wilfred", "mickey"))
    all_pairs: list[tuple[float, float]] = []

    for animal in animals:
        trials = animal_data[animal]["trials"]
        for t in trials:
            if t[2] != "training" or t[5] is None or t[6] is None or t[4] is None:
                continue
            left, right, ttt = t[5], t[6], t[4]
            if left == 360 and right == 360:
                all_pairs.append((360.0, ttt))  # vertical vs vertical at x=360
                continue
            if left == 360 and 0 <= right <= ANGLE_MAX and right != 360:
                all_pairs.append((round(right, 1), ttt))
            elif right == 360 and 0 <= left <= ANGLE_MAX and left != 360:
                all_pairs.append((round(left, 1), ttt))

    if not all_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Angle (deg, non-vertical side)")
        ax.set_ylabel("Mean time to target (s)")
        ax.set_title("Vertical (360°) vs angle: mean TTT (rory + wilfred + mickey) — no matching trials")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved {out_path}")
        else:
            plt.show()
        return

    cell_ttt: dict[float, list[float]] = defaultdict(list)
    for angle, ttt in all_pairs:
        cell_ttt[angle].append(ttt)

    angles = sorted(cell_ttt.keys())
    means = [np.mean(cell_ttt[a]) for a in angles]
    # Error bars: SEM = std / sqrt(n)
    sems = [np.std(cell_ttt[a], ddof=1) / np.sqrt(len(cell_ttt[a])) if len(cell_ttt[a]) > 1 else 0.0 for a in angles]

    # Test: (360 vs 0) vs (360 vs 90) — is *mean* TTT different? (Welch t-test)
    ttt_0 = cell_ttt.get(0.0) or cell_ttt.get(0)
    ttt_90 = cell_ttt.get(90.0) or cell_ttt.get(90)
    sig_0_vs_90 = False
    p_val_0_vs_90: float | None = None
    if ttt_0 and ttt_90 and len(ttt_0) >= 2 and len(ttt_90) >= 2:
        try:
            _, p_val_0_vs_90 = stats.ttest_ind(ttt_0, ttt_90, equal_var=False)  # Welch
            sig_0_vs_90 = p_val_0_vs_90 < 0.05
        except Exception:
            pass
    if p_val_0_vs_90 is not None:
        print(f"Vertical vs angle (mean): (360 vs 0) vs (360 vs 90) [Welch t-test on means], p = {p_val_0_vs_90:.4f}" + (" (*)" if sig_0_vs_90 else ""))

    # Test: (360 vs 10) vs (360 vs 100) — is *mean* TTT different? (Welch t-test)
    ttt_10 = cell_ttt.get(10.0) or cell_ttt.get(10)
    ttt_100 = cell_ttt.get(100.0) or cell_ttt.get(100)
    sig_10_vs_100 = False
    p_val_10_vs_100: float | None = None
    if ttt_10 and ttt_100 and len(ttt_10) >= 2 and len(ttt_100) >= 2:
        try:
            _, p_val_10_vs_100 = stats.ttest_ind(ttt_10, ttt_100, equal_var=False)  # Welch
            sig_10_vs_100 = p_val_10_vs_100 < 0.05
        except Exception:
            pass
    if p_val_10_vs_100 is not None:
        print(f"Vertical vs angle (mean): (360 vs 10) vs (360 vs 100) [Welch t-test on means], p = {p_val_10_vs_100:.4f}" + (" (*)" if sig_10_vs_100 else ""))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        angles,
        means,
        yerr=sems,
        fmt="o-",
        color="black",
        linewidth=2,
        markersize=6,
        capsize=3,
        capthick=1.5,
    )

    y_range = max(means) - min(m - s for m, s in zip(means, sems))
    if sig_0_vs_90 and (0 in angles or 0.0 in angles) and (90 in angles or 90.0 in angles):
        idx_0 = angles.index(0.0) if 0.0 in angles else angles.index(0)
        idx_90 = angles.index(90.0) if 90.0 in angles else angles.index(90)
        y_0 = means[idx_0] + sems[idx_0]
        y_90 = means[idx_90] + sems[idx_90]
        y_bracket = max(y_0, y_90) + 0.06 * y_range
        ax.plot([0, 0], [y_0, y_bracket], color="black", linewidth=1)
        ax.plot([90, 90], [y_90, y_bracket], color="black", linewidth=1)
        ax.plot([0, 90], [y_bracket, y_bracket], color="black", linewidth=1)
        ax.annotate("*", xy=(45, y_bracket), ha="center", va="bottom", fontsize=16)
    if sig_10_vs_100 and (10 in angles or 10.0 in angles) and (100 in angles or 100.0 in angles):
        idx_10 = angles.index(10.0) if 10.0 in angles else angles.index(10)
        idx_100 = angles.index(100.0) if 100.0 in angles else angles.index(100)
        y_10 = means[idx_10] + sems[idx_10]
        y_100 = means[idx_100] + sems[idx_100]
        y_bracket = max(y_10, y_100) + 0.06 * y_range
        if sig_0_vs_90 and (0 in angles or 0.0 in angles):
            y_bracket += 0.08 * y_range  # stack above 0–90 bracket
        ax.plot([10, 10], [y_10, y_bracket], color="black", linewidth=1)
        ax.plot([100, 100], [y_100, y_bracket], color="black", linewidth=1)
        ax.plot([10, 100], [y_bracket, y_bracket], color="black", linewidth=1)
        ax.annotate("*", xy=(55, y_bracket), ha="center", va="bottom", fontsize=16)

    ax.set_xlabel("Angle (deg, non-vertical side; vertical = 360°)")
    ax.set_ylabel("Mean time to target (s)")
    ax.set_title("Vertical (360°) vs angle 0–360° (incl. vertical vs vertical at 360); mean TTT; error bars = SEM")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


def main():
    logs_dir = LOGS_DIR
    print(f"Reading logs from {logs_dir}")
    animal_data = collect_all_sessions(logs_dir)

    if not animal_data:
        print("No animal/session data found.")
        return

    out_dir = Path(__file__).resolve().parent
    plot_pretraining_vs_training(
        animal_data,
        out_path=out_dir / "plots_pretraining_vs_training.png",
    )
    plot_learning_curves(
        animal_data,
        out_path=out_dir / "plots_learning_curves.png",
    )
    plot_learning_curves_first_100_training_combined(
        animal_data,
        out_path=out_dir / "plots_learning_curves_first100_training_combined.eps",
    )
    plot_time_to_reward(
        animal_data,
        out_path=out_dir / "plots_time_to_reward.png",
    )
    plot_time_to_target(
        animal_data,
        out_path=out_dir / "plots_time_to_target.png",
    )
    plot_angle_heatmap(
        animal_data,
        out_path=out_dir / "plots_angle_heatmap.png",
    )
    plot_angle_heatmap_time_to_target(
        animal_data,
        out_path=out_dir / "plots_angle_heatmap_time_to_target.png",
    )
    plot_angle_heatmap_time_to_target_median(
        animal_data,
        out_path=out_dir / "plots_angle_heatmap_time_to_target_median.png",
    )
    plot_angle_diff_vs_median_ttt(
        animal_data,
        out_path=out_dir / "plots_angle_diff_vs_median_ttt.png",
    )
    plot_vertical_vs_angle_median_ttt(
        animal_data,
        out_path=out_dir / "plots_vertical_vs_angle_median_ttt.png",
    )
    plot_vertical_vs_angle_mean_ttt(
        animal_data,
        out_path=out_dir / "plots_vertical_vs_angle_mean_ttt.png",
    )


if __name__ == "__main__":
    main()
