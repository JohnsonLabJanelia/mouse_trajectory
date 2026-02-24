"""
Run 3D prediction per trial from a trials CSV (frame_id_start, frame_id_end).
Saves one prediction folder per trial under predictions/predictions3D/.
"""
import argparse
import os
import sys
import pandas as pd
from jarvis.config.project_manager import ProjectManager
from jarvis.utils.paramClasses import Predict3DParams
from jarvis.prediction.predict3D import predict3D


def main():
    p = argparse.ArgumentParser(
        description=(
            "Run 3D prediction for each trial in trial_frames.csv; "
            "save one folder per trial under predictions/predictions3D/."
        )
    )
    p.add_argument("--project", required=True, help="JARVIS project name (e.g. mouseClimb4)")
    p.add_argument("--recording-path", required=True, help="Path to folder containing camera videos")
    p.add_argument("--dataset-name", required=True, help="Calibration params folder path")
    p.add_argument(
        "--trials-csv",
        required=True,
        help="CSV with columns frame_id_start, frame_id_end, and optionally trial_index",
    )
    p.add_argument("--weights-center", default="latest")
    p.add_argument("--weights-hybridnet", default="latest")
    p.add_argument("--trt", choices=["off", "new", "previous"], default="off")
    p.add_argument(
        "--trial-index-col",
        default="trial_index",
        help="Column name for trial index (used in output folder name)",
    )
    p.add_argument("--start-col", default="frame_id_start")
    p.add_argument("--end-col", default="frame_id_end")
    p.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Optional subdirectory under predictions/predictions3D/ to store this "
            "session's trial folders (e.g. animal_YYYY_MM_DD_HH_MM_SS)."
        ),
    )
    args = p.parse_args()

    if not os.path.exists(args.recording_path):
        print(f"[error] Recording path not found: {args.recording_path}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.trials_csv):
        print(f"[error] Trials CSV not found: {args.trials_csv}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.trials_csv)
    for col in (args.start_col, args.end_col):
        if col not in df.columns:
            print(f"[error] Trials CSV must have column '{col}'", file=sys.stderr)
            sys.exit(2)
    if args.trial_index_col not in df.columns:
        df[args.trial_index_col] = range(len(df))

    project = ProjectManager()
    if not project.load(args.project):
        print(f"[error] Could not load project: {args.project}", file=sys.stderr)
        sys.exit(2)
    cfg = project.cfg

    # Base directory for predictions. Optionally place all trials for a session
    # into a dedicated subdirectory (e.g. animal_YYYY_MM_DD_HH_MM_SS) so that
    # different sessions do not overwrite each other's predictions.
    base_parts = [
        project.parent_dir,
        cfg.PROJECTS_ROOT_PATH,
        args.project,
        "predictions",
        "predictions3D",
    ]
    if args.output_subdir:
        base_parts.append(args.output_subdir)
    base_dir = os.path.join(*base_parts)
    os.makedirs(base_dir, exist_ok=True)

    n_trials = len(df)
    for i, row in df.iterrows():
        frame_start = int(row[args.start_col])
        frame_end = int(row[args.end_col])
        num_frames = frame_end - frame_start + 1
        trial_idx = int(row[args.trial_index_col])
        out_name = f"Predictions_3D_trial_{trial_idx:04d}_{frame_start}-{frame_end}"
        output_dir = os.path.join(base_dir, out_name)

        print(f"[{i+1}/{n_trials}] Trial {trial_idx}: frames {frame_start}-{frame_end} ({num_frames} frames) -> {out_name}")

        params = Predict3DParams(
            project_name=args.project,
            recording_path=args.recording_path,
            weights_center_detect=args.weights_center,
            weights_hybridnet=args.weights_hybridnet,
            frame_start=frame_start,
            number_frames=num_frames,
            trt_mode=args.trt,
            output_dir=output_dir,
        )
        params.dataset_name = args.dataset_name
        predict3D(params)

    print(f"Done. {n_trials} trial predictions saved under {base_dir}")


if __name__ == "__main__":
    main()

