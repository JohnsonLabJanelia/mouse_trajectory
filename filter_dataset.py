#!/usr/bin/env python3
"""
Filter raw predictions into a clean 24-kp dataset.

Reads every trial's data3D.csv from dataset/predictions_raw/, applies
arena-mask gating on the Snout position, and writes one CSV per animal:

    dataset/rory.csv
    dataset/wilfred.csv

Each row:
    animal, video_folder, trial_index, frame_id,
    Snout_x, Snout_y, Snout_z, Snout_conf,
    EarL_x, ..., TailTip_conf

Frame_id is the absolute video frame number (matches frame_id in trials CSV).

Arena mask:
    predictions3D/arena_mask.npz  — binary (H, W) mask in Cam2005325 image space.
    A frame is kept only if:
      - Snout_x is not NaN  (mouse was detected)
      - Snout projects inside the arena mask in Cam2005325

Usage:
    python3 filter_dataset.py
    python3 filter_dataset.py --min-confidence 0.1 --no-arena-mask
    python3 filter_dataset.py --animals rory   (partial update)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_DIR  = SCRIPT_DIR / 'dataset'
RAW_DIR      = DATASET_DIR / 'predictions_raw'
ARENA_MASK_PATH  = SCRIPT_DIR / 'predictions3D' / 'arena_mask.npz'
REF_CAMERA       = 'Cam2005325'   # arena mask is in this camera's image space


# ---------------------------------------------------------------------------
# Calibration / projection
# ---------------------------------------------------------------------------

def load_calib_P(calib_dir: Path, cam_name: str) -> np.ndarray | None:
    """Load projection matrix P (3×4) for cam_name from calib_dir."""
    yaml_path = calib_dir / f'{cam_name}.yaml'
    if not yaml_path.exists():
        return None
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    K  = fs.getNode('intrinsicMatrix').mat()   # 3×3 (JARVIS: stored as K^T)
    R  = fs.getNode('R').mat()                 # 3×3
    T  = fs.getNode('T').mat()                 # 3×1
    fs.release()
    RT = np.vstack([R, T.reshape(1, 3)])       # 4×3
    return (RT @ K).T                          # 3×4


def project_point(X3d: np.ndarray, P: np.ndarray):
    """Project Nx3 world points through P (3×4) -> Nx2 pixel coords (u, v)."""
    X4 = np.hstack([X3d, np.ones((len(X3d), 1))])   # (N, 4)
    p  = X4 @ P.T                                     # (N, 3)
    u  = p[:, 0] / p[:, 2]
    v  = p[:, 1] / p[:, 2]
    return u, v


# ---------------------------------------------------------------------------
# Per-trial CSV reader
# ---------------------------------------------------------------------------

def read_data3d(csv_path: Path):
    """
    Read a data3D.csv file.
    Returns (kp_names, coords_array) where coords_array is (N_frames, J*4).
    Rows with all-NaN are kept as NaN rows.
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        kp_row    = next(reader)
        coord_row = next(reader)
        rows = [r for r in reader]

    kp_names = list(dict.fromkeys(kp_row))   # unique, preserving order
    n_joints = len(kp_names)
    n_frames = len(rows)

    data = np.full((n_frames, n_joints * 4), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if val != 'NaN':
                try:
                    data[i, j] = float(val)
                except ValueError:
                    pass
    return kp_names, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--animals',         nargs='+', default=['rory', 'wilfred'])
    ap.add_argument('--dataset-dir',     type=Path, default=DATASET_DIR)
    ap.add_argument('--min-confidence',  type=float, default=0.05,
                    help='Drop frames where mean Snout confidence < this')
    ap.add_argument('--no-arena-mask',   action='store_true',
                    help='Skip arena mask filter (keep all detected frames)')
    ap.add_argument('--calib-root',      type=Path,
                    default=SCRIPT_DIR / 'calib_params')
    args = ap.parse_args()

    raw_dir = args.dataset_dir / 'predictions_raw'
    if not raw_dir.is_dir():
        sys.exit(f'predictions_raw not found: {raw_dir}\nRun make_dataset.py first.')

    # Load arena mask
    arena_mask = None
    if not args.no_arena_mask:
        if ARENA_MASK_PATH.exists():
            arena_mask = np.load(ARENA_MASK_PATH)['mask']
            print(f'Arena mask loaded: {arena_mask.shape}')
        else:
            print(f'Warning: arena mask not found at {ARENA_MASK_PATH}, skipping spatial filter')

    # Load sessions metadata
    sessions_csv = args.dataset_dir / 'sessions.csv'
    if not sessions_csv.exists():
        sys.exit(f'sessions.csv not found: {sessions_csv}')
    sessions = {(r['animal'], r['video_folder']): r
                for r in csv.DictReader(open(sessions_csv))}

    # Cache projection matrices for reference camera per calib date
    P_cache = {}

    for animal in args.animals:
        out_csv = args.dataset_dir / f'{animal}.csv'
        print(f'\n=== {animal} -> {out_csv} ===')

        # Collect all session dirs for this animal
        session_dirs = sorted(d for d in raw_dir.iterdir()
                              if d.is_dir() and d.name.startswith(f'{animal}_'))

        # Build column names
        kp_names_known = None
        total_kept = 0
        total_frames = 0

        with open(out_csv, 'w', newline='') as out_fh:
            writer = None   # created after we know kp_names

            for session_dir in session_dirs:
                session_name  = session_dir.name              # e.g. rory_2026_01_07_...
                video_folder  = session_name[len(animal)+1:]  # strip 'rory_'
                meta_key      = (animal, video_folder)

                if meta_key not in sessions:
                    print(f'  No metadata for {session_name}, skipping')
                    continue

                meta      = sessions[meta_key]
                calib_dir = Path(meta['calib_dir'])
                try:
                    fps = float(meta.get('fps', 180))
                except ValueError:
                    fps = 180.0

                # Load projection matrix for reference camera (cached by calib date)
                calib_key = meta['calib_date']
                if calib_key not in P_cache:
                    P = load_calib_P(calib_dir, REF_CAMERA)
                    P_cache[calib_key] = P
                P_ref = P_cache[calib_key]

                trial_dirs = sorted(d for d in session_dir.iterdir()
                                    if d.is_dir() and d.name.startswith('trial_'))

                for trial_dir in trial_dirs:
                    data3d_path = trial_dir / 'data3D.csv'
                    if not data3d_path.exists():
                        continue

                    # Parse trial metadata from folder name:
                    # trial_{idx:04d}_{frame_start}-{frame_end}
                    parts = trial_dir.name.split('_')
                    try:
                        trial_idx   = int(parts[1])
                        frame_start = int(parts[2].split('-')[0])
                        frame_end   = int(parts[2].split('-')[1])
                    except (IndexError, ValueError):
                        print(f'  Cannot parse trial dir name: {trial_dir.name}')
                        continue

                    kp_names, data = read_data3d(data3d_path)
                    n_frames = data.shape[0]
                    n_joints = len(kp_names)

                    # Infer stride from frame count vs predicted rows
                    recorded_frames = frame_end - frame_start + 1
                    stride = max(1, round(recorded_frames / n_frames))

                    # Build absolute frame ids
                    frame_ids = np.arange(frame_start, frame_start + n_frames * stride, stride)
                    frame_ids = frame_ids[:n_frames]   # guard against rounding

                    # Set up writer on first trial
                    if writer is None:
                        kp_names_known = kp_names
                        cols = ['animal', 'video_folder', 'trial_index', 'frame_id']
                        for kp in kp_names:
                            cols += [f'{kp}_x', f'{kp}_y', f'{kp}_z', f'{kp}_conf']
                        writer = csv.DictWriter(out_fh, fieldnames=cols)
                        writer.writeheader()

                    # Snout index = 0 (first keypoint)
                    snout_x = data[:, 0]
                    snout_y = data[:, 1]
                    snout_z = data[:, 2]
                    snout_c = data[:, 3]

                    # Basic filter: snout was detected
                    valid = ~np.isnan(snout_x)

                    # Confidence filter
                    valid &= (snout_c >= args.min_confidence)

                    # Arena mask filter
                    if arena_mask is not None and P_ref is not None:
                        snout_pts = np.stack([snout_x, snout_y, snout_z], axis=1)
                        # Only project valid rows to avoid NaN in projection
                        u_all = np.full(n_frames, -1.0)
                        v_all = np.full(n_frames, -1.0)
                        if valid.any():
                            u_v, v_v = project_point(snout_pts[valid], P_ref)
                            u_all[valid] = u_v
                            v_all[valid] = v_v

                        mask_h, mask_w = arena_mask.shape
                        ui = np.clip(u_all.astype(np.int32), 0, mask_w - 1)
                        vi = np.clip(v_all.astype(np.int32), 0, mask_h - 1)
                        in_bounds = ((u_all >= 0) & (u_all < mask_w) &
                                     (v_all >= 0) & (v_all < mask_h))
                        in_mask   = (arena_mask[vi, ui] != 0) & in_bounds
                        valid    &= in_mask

                    kept_idx = np.where(valid)[0]
                    total_frames += n_frames
                    total_kept   += len(kept_idx)

                    for i in kept_idx:
                        row = {
                            'animal': animal,
                            'video_folder': video_folder,
                            'trial_index': trial_idx,
                            'frame_id': int(frame_ids[i]),
                        }
                        for j, kp in enumerate(kp_names):
                            row[f'{kp}_x']    = round(float(data[i, j*4+0]), 4)
                            row[f'{kp}_y']    = round(float(data[i, j*4+1]), 4)
                            row[f'{kp}_z']    = round(float(data[i, j*4+2]), 4)
                            row[f'{kp}_conf'] = round(float(data[i, j*4+3]), 4)
                        writer.writerow(row)

        pct = 100 * total_kept / total_frames if total_frames else 0
        print(f'  Kept {total_kept:,} / {total_frames:,} frames ({pct:.1f}%) -> {out_csv}')

    print('\nDone. Dataset files:')
    for f in sorted(args.dataset_dir.glob('*.csv')):
        mb = f.stat().st_size / 1e6
        print(f'  {f.name}  ({mb:.1f} MB)')


if __name__ == '__main__':
    main()
