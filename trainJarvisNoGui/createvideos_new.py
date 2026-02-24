import os
import argparse
import time
import cv2
import json
import numpy as np
import pandas as pd

from ruamel.yaml import YAML
from tqdm import tqdm

# Jarvis utilities
from jarvis.utils.reprojection import get_repro_tool
from jarvis.config.project_manager import ProjectManager
from jarvis.utils.skeleton import get_skeleton
import jarvis.visualization.visualization_utils as utils


def _load_info(pred_dir):
    info_path = os.path.join(pred_dir, "info.yaml")
    if not os.path.isfile(info_path):
        raise SystemExit(f"Missing info.yaml in {pred_dir}")
    yaml = YAML()
    with open(info_path, "r") as f:
        return yaml.load(f)


def _parse_cams(cams_csv, default=None):
    if cams_csv:
        cams = [c.strip() for c in cams_csv.split(",") if c.strip()]
        if len(cams) == 0:
            return default
        return cams
    return default


def _load_predictions(data_csv):
    """Load data3D.csv robustly."""
    try:
        df = pd.read_csv(data_csv, header=None)
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
        while df_numeric.iloc[0].isna().all():
            df_numeric = df_numeric.iloc[1:]
        data = df_numeric.to_numpy()
    except Exception:
        data = np.genfromtxt(data_csv, delimiter=",")
        if np.isnan(data[0, 0]):
            data = data[2:]

    if data.ndim != 2:
        raise SystemExit(f"Unexpected CSV shape from {data_csv}: {data.shape}")

    ncols = data.shape[1]
    has_frame_col = ((ncols - 1) % 4 == 0)
    if has_frame_col:
        frame_idx = data[:, 0].astype(np.int64)
        payload = data[:, 1:]
    else:
        frame_idx = None
        payload = data

    if payload.shape[1] % 4 != 0:
        raise SystemExit("CSV format error: columns not divisible into (x,y,z,conf) tuples")

    J = payload.shape[1] // 4
    conf = payload[:, 3::4]
    xyz_flat = np.delete(payload, list(range(3, payload.shape[1], 4)), axis=1)
    points3D = xyz_flat.reshape(-1, J, 3).astype(np.float32)
    conf = conf.astype(np.float32)
    return frame_idx, points3D, conf


def _get_video_paths_and_flags(recording_path, reproTool, wanted_cams):
    videos = os.listdir(recording_path)
    video_paths, write_flag = [], []
    for i, cam in enumerate(reproTool.cameras):
        found = False
        for v in videos:
            if cam == v.split(".")[0]:
                video_paths.append(os.path.join(recording_path, v))
                write_flag.append(cam in wanted_cams)
                found = True
                break
        if not found:
            raise SystemExit(f"Missing recording for camera {cam}")
    return video_paths, write_flag


def _open_caps_and_writers(video_paths, write_flag, out_dir, frame_start):
    caps, outs = [], []
    img_size = [0, 0]
    fps_ref = None
    os.makedirs(out_dir, exist_ok=True)

    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if img_size == [0, 0]:
            img_size = [w, h]
            fps_ref = fps
        else:
            assert img_size == [w, h], "All videos must share same resolution"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_start >= total_frames:
            raise SystemExit("frame_start exceeds total video length")

        caps.append(cap)
        if write_flag[i]:
            basename = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{basename}.mp4")
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            outs.append(cv2.VideoWriter(out_path, fourcc, fps_ref, (w, h)))
        else:
            outs.append(None)
    return caps, outs, img_size


def _seek_all(caps, frame_num):
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


def _read_all(caps, img_buf):
    for i, cap in enumerate(caps):
        ok, frame = cap.read()
        if not ok:
            h, w = img_buf.shape[1], img_buf.shape[2]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        img_buf[i] = frame.astype(np.uint8)


def _draw_pose_for_cam(img, pts2D_cam, skeleton_lines, colors, img_size):
    good = np.isfinite(pts2D_cam).all(axis=1)
    for (a, b) in skeleton_lines:
        if a < len(pts2D_cam) and b < len(pts2D_cam):
            if good[a] and good[b]:
                utils.draw_line(img, (a, b), pts2D_cam, img_size, colors[b])
    for j, p in enumerate(pts2D_cam):
        if j < len(pts2D_cam) and good[j]:
            utils.draw_point(img, p, img_size, colors[j])


def render_all_trials(prediction_dir,
                      trials_csv,
                      start_col="trial_start",
                      end_col="trial_end",
                      cams_csv=None,
                      min_conf=0.0,
                      clip_prefix="clip",
                      project_name=None):
    """Render annotated videos for all trial ranges."""
    info = _load_info(prediction_dir)
    recording_path = info.get("recording_path")
    dataset_name = info.get("dataset_name")
    if not (project_name and recording_path and dataset_name):
        raise SystemExit("Please provide --project-name; info.yaml must contain recording_path and dataset_name")

    project = ProjectManager()
    if not project.load(project_name):
        raise SystemExit(f"Could not load project {project_name}")
    cfg = project.cfg
    reproTool = get_repro_tool(cfg, dataset_name, "cpu")

    all_cams = list(reproTool.cameras)
    video_cam_list = _parse_cams(cams_csv, default=all_cams[:2] if len(all_cams) >= 2 else all_cams)

    csvs = sorted([fn for fn in os.listdir(prediction_dir) if fn.lower().endswith(".csv")])
    if not csvs:
        raise SystemExit(f"No CSV prediction file in {prediction_dir}")
    data_csv = os.path.join(prediction_dir, csvs[0])

    frame_idx, points3D, confidences = _load_predictions(data_csv)
    num_frames, num_joints = points3D.shape[0], points3D.shape[1]
    if frame_idx is None:
        frame_idx = np.arange(num_frames, dtype=np.int64)
    frame_to_row = {int(f): i for i, f in enumerate(frame_idx.tolist())}

    colors, line_idxs = get_skeleton(cfg)

    tr = pd.read_csv(trials_csv)
    if start_col not in tr.columns or end_col not in tr.columns:
        raise SystemExit(f"Trials CSV must contain '{start_col}' and '{end_col}' columns")
    trials = tr[[start_col, end_col]].dropna().astype(int).to_numpy()

    out_root = os.path.join(prediction_dir, "visualization",
                            f"Videos_3D_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_root, exist_ok=True)
    print(f"Writing videos under: {out_root}")

    for t_idx, (start_f, end_f) in enumerate(trials):
        if end_f < start_f:
            continue
        clip_len = end_f - start_f + 1
        clip_dir = os.path.join(out_root, f"{clip_prefix}_{t_idx:04d}_{start_f}-{end_f}")
        os.makedirs(clip_dir, exist_ok=True)

        video_paths, write_flag = _get_video_paths_and_flags(recording_path, reproTool, video_cam_list)
        caps, outs, img_size = _open_caps_and_writers(video_paths, write_flag, clip_dir, start_f)
        _seek_all(caps, start_f)
        imgs = np.zeros((len(caps), img_size[1], img_size[0], 3), dtype=np.uint8)

        pbar = tqdm(total=clip_len, desc=f"Clip {t_idx} [{start_f}-{end_f}]")

        for f in range(start_f, end_f + 1):
            _read_all(caps, imgs)
            row = frame_to_row.get(int(f), None)
            if row is not None:
                pts3D = points3D[row]
                conf = confidences[row]
                if np.isfinite(pts3D).any():
                    import torch
                    pts3D_t = torch.from_numpy(np.asarray(pts3D, dtype=np.float32)).float()
                    pts2D = reproTool.reprojectPoint(pts3D_t).numpy()

                    for cam_i, out in enumerate(outs):
                        if out is None:
                            continue
                        pts2D_cam = pts2D[:, cam_i, :]
                        valid = np.isfinite(pts2D_cam).all(axis=1) & np.isfinite(pts3D).all(axis=1) & (conf >= min_conf)
                        pts2D_cam_masked = pts2D_cam.copy()
                        pts2D_cam_masked[~valid] = np.nan
                        _draw_pose_for_cam(imgs[cam_i], pts2D_cam_masked, line_idxs, colors, img_size)

            for cam_i, out in enumerate(outs):
                if out is not None:
                    out.write(imgs[cam_i])
            pbar.update(1)

        pbar.close()
        for out in outs:
            if out is not None:
                out.release()
        for cap in caps:
            cap.release()
    print("Done.")


def main():
    ap = argparse.ArgumentParser("Render annotated 3D videos for ALL trial ranges")
    ap.add_argument("--prediction-dir", required=True,
                    help="Path to a Predictions_3D_* folder produced by jarvis_predict3D.py")
    ap.add_argument("--trials-csv", required=True,
                    help="CSV with trial ranges (same used for prediction)")
    ap.add_argument("--project-name", required=True,
                    help="Name of the Jarvis project (same used for prediction)")
    ap.add_argument("--start-col", default="trial_start")
    ap.add_argument("--end-col", default="trial_end")
    ap.add_argument("--cams", default=None,
                    help="Comma-separated camera names, e.g. 'Cam710031,Cam710037'")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Per-joint minimum confidence; joints below are hidden")
    ap.add_argument("--clip-prefix", default="clip")
    args = ap.parse_args()

    render_all_trials(
        prediction_dir=args.prediction_dir,
        trials_csv=args.trials_csv,
        start_col=args.start_col,
        end_col=args.end_col,
        cams_csv=args.cams,
        min_conf=args.min_conf,
        clip_prefix=args.clip_prefix,
        project_name=args.project_name
    )


if __name__ == "__main__":
    main()



'''
python scripts/createvideos_new.py \
--project-name fin5 \
--prediction-dir /groups/johnson/johnsonlab/jinyao_share/2025_06_04_14_04_41/trial_wise_predict3d/predictions/jarvis/fin5/Predictions_3D_20251119-182738/ \
--trials-csv /groups/johnson/johnsonlab/jinyao_share/2025_06_04_14_04_41/trial_wise_predict3d/trial_sorted.csv
'''

