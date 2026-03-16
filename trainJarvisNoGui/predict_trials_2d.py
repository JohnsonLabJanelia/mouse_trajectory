"""
Run 2D+DLT prediction per trial from a trials CSV.

Drop-in replacement for predict_trials.py that uses predict2D_triangulate
(mouseHybrid24 CenterDetect + KeypointDetect + DLT) instead of HybridNet 3D.

Models are loaded once and reused across all trials in the session.
Output directory structure matches predict_trials.py so downstream scripts work.

Usage:
    conda run -n jarvis python3 predict_trials_2d.py \\
        --project mouseHybrid24 \\
        --recording-path /mnt/mouse2/rory/2026_01_07_17_31_36 \\
        --calib-dir /home/user/src/analyzeMiceTrajectory/calib_params/2026_01_07 \\
        --trials-csv /tmp/rory_2026_01_07.csv \\
        --trt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, '/home/user/src/JARVIS-HybridNet')
from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack

# Reuse helpers from predict2D_triangulate
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))
from predict2D_triangulate import (
    load_all_calibrations,
    triangulate_dlt,
)


def _read_frame_parallel(caps, imgs_buf):
    def _read(cap, idx):
        ret, img = cap.read()
        if ret and img is not None:
            imgs_buf[idx] = img
        return ret
    results = Parallel(n_jobs=len(caps), require='sharedmem')(
        delayed(_read)(cap, i) for i, cap in enumerate(caps))
    return all(results)


def _find_video(recording_path, cam_name):
    for fname in os.listdir(recording_path):
        if fname.split('.')[0] == cam_name and fname.endswith('.mp4'):
            return os.path.join(recording_path, fname)
    return None


def run_session(
    project_name,
    recording_path,
    calib_dir,
    trials_csv,
    output_subdir,
    use_trt=False,
    center_threshold=40.0,
    kp_conf_threshold=0.0,
    max_reproj_err=200.0,
    start_col='frame_id_start',
    end_col='frame_id_end',
    trial_index_col='trial_index',
):
    # ---- Load project / models (once per session) ----
    project = ProjectManager()
    if not project.load(project_name):
        print(f"[error] Cannot load project {project_name}", file=sys.stderr)
        return False
    cfg = project.cfg

    trt_dir = os.path.join(
        '/home/user/src/JARVIS-HybridNet/projects', project_name,
        'trt-models', 'predict2D')

    if use_trt and os.path.isdir(trt_dir):
        import torch_tensorrt  # noqa
        print("[2D+tri] Loading TRT models ...")
        center_model   = torch.jit.load(os.path.join(trt_dir, 'centerDetect.pt')).cuda()
        keypoint_model = torch.jit.load(os.path.join(trt_dir, 'keypointDetect.pt')).cuda()
    else:
        if use_trt:
            print(f"[2D+tri] WARNING: TRT models not found at {trt_dir}, falling back to PyTorch")
        center_model   = EfficientTrack('CenterDetectInference',  cfg, 'latest').model.eval().cuda()
        keypoint_model = EfficientTrack('KeypointDetectInference', cfg, 'latest').model.eval().cuda()

    img_size_cd = cfg.CENTERDETECT.IMAGE_SIZE
    bbox_size   = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    bbox_hw     = bbox_size // 2
    num_joints  = cfg.KEYPOINTDETECT.NUM_JOINTS
    mean_t = torch.tensor(cfg.DATASET.MEAN, device='cuda').view(3,1,1)
    std_t  = torch.tensor(cfg.DATASET.STD,  device='cuda').view(3,1,1)

    # ---- Calibration ----
    calib_files  = sorted(f for f in os.listdir(calib_dir) if f.endswith('.yaml'))
    camera_names = [f.replace('.yaml','') for f in calib_files
                    if 'Cam710040' not in f]
    num_cameras  = len(camera_names)
    _, _, _, _, Ps_np = load_all_calibrations(calib_dir, camera_names)
    Ps_np = Ps_np.astype(np.float32)

    # ---- Video file map ----
    video_map = {}
    for cn in camera_names:
        vp = _find_video(recording_path, cn)
        if vp:
            video_map[cn] = vp
    camera_names = [cn for cn in camera_names if cn in video_map]
    cam_idx_map  = {cn: i for i, cn in enumerate(camera_names)}
    Ps_np = Ps_np[[cam_idx_map[cn] for cn in camera_names]]
    num_cameras  = len(camera_names)
    print(f"[2D+tri] {num_cameras} cameras with video + calib")

    # ---- Get video dimensions from first camera ----
    sample_cap = cv2.VideoCapture(video_map[camera_names[0]])
    img_h = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_cap.release()

    # ---- Output base dir (same structure as predict_trials.py) ----
    base_dir = os.path.join(
        '/home/user/src/JARVIS-HybridNet/projects', project_name,
        'predictions', 'predictions3D',
        output_subdir)
    os.makedirs(base_dir, exist_ok=True)

    # ---- Read trials CSV ----
    import pandas as pd
    df = pd.read_csv(trials_csv)
    for col in (start_col, end_col):
        if col not in df.columns:
            print(f"[error] Missing column '{col}' in {trials_csv}", file=sys.stderr)
            return False
    if trial_index_col not in df.columns:
        df[trial_index_col] = range(len(df))

    n_trials = len(df)
    imgs_buf = np.zeros((num_cameras, img_h, img_w, 3), dtype=np.uint8)

    for row_i, row in df.iterrows():
        frame_start = int(row[start_col])
        frame_end   = int(row[end_col])
        num_frames  = frame_end - frame_start + 1
        trial_idx   = int(row[trial_index_col])
        out_name    = f"Predictions_3D_trial_{trial_idx:04d}_{frame_start}-{frame_end}"
        output_dir  = os.path.join(base_dir, out_name)

        if os.path.isdir(output_dir) and os.path.isfile(os.path.join(output_dir, 'data3D.csv')):
            print(f"[{row_i+1}/{n_trials}] Trial {trial_idx}: already done, skipping")
            continue

        print(f"[{row_i+1}/{n_trials}] Trial {trial_idx}: frames {frame_start}-{frame_end} "
              f"({num_frames} frames) -> {out_name}")
        t0 = time.perf_counter()

        # Open all cameras and seek to start
        caps = [cv2.VideoCapture(video_map[cn]) for cn in camera_names]
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        os.makedirs(output_dir, exist_ok=True)
        csvfile = open(os.path.join(output_dir, 'data3D.csv'), 'w', newline='')
        writer  = csv.writer(csvfile)
        if len(cfg.KEYPOINT_NAMES) == num_joints:
            import itertools
            joints = list(itertools.chain.from_iterable(
                itertools.repeat(kp, 4) for kp in cfg.KEYPOINT_NAMES))
            writer.writerow(joints)
            writer.writerow(['x','y','z','confidence'] * num_joints)

        for _ in tqdm(range(num_frames), leave=False):
            ok = _read_frame_parallel(caps, imgs_buf)
            if not ok:
                writer.writerow(['NaN'] * (num_joints * 4))
                continue

            imgs_t = (torch.from_numpy(imgs_buf).cuda().float()
                      .permute(0, 3, 1, 2)[:, [2,1,0]] / 255.0)

            with torch.no_grad():
                imgs_cd = F.interpolate(imgs_t, (img_size_cd,)*2,
                                        mode='bilinear', align_corners=False)
                cd_out  = center_model((imgs_cd - mean_t) / std_t)
                hm      = cd_out[1].view(num_cameras, -1)
                m       = hm.argmax(1)
                maxvals = hm[range(num_cameras), m]
                cx = (m % cd_out[1].shape[3]).float() * (img_w / cd_out[1].shape[3])
                cy = (m // cd_out[1].shape[3]).float() * (img_h / cd_out[1].shape[2])
                cx = cx.long().clamp(bbox_hw, img_w - bbox_hw)
                cy = cy.long().clamp(bbox_hw, img_h - bbox_hw)

                detected = (maxvals > center_threshold).cpu().numpy()
                det_idx  = np.where(detected)[0]

                if len(det_idx) < 2:
                    writer.writerow(['NaN'] * (num_joints * 4))
                    continue

                crops = torch.stack([
                    imgs_t[i, :, cy[i]-bbox_hw:cy[i]+bbox_hw, cx[i]-bbox_hw:cx[i]+bbox_hw]
                    for i in det_idx
                ])
                kp_out  = keypoint_model((crops - mean_t) / std_t)
                hm_kp   = kp_out[1].view(len(det_idx), num_joints, -1)
                m_kp    = hm_kp.argmax(2)
                conf_kp = (hm_kp.gather(2, m_kp.unsqueeze(-1)).squeeze(-1).clamp(max=255.) / 255.).cpu().numpy()
                kp_x    = (m_kp % kp_out[1].shape[3]).float() * 2
                kp_y    = (m_kp // kp_out[1].shape[3]).float() * 2
                kp_x   += (cx[det_idx] - bbox_hw).float().unsqueeze(1)
                kp_y   += (cy[det_idx] - bbox_hw).float().unsqueeze(1)
                kp_x    = kp_x.cpu().numpy()
                kp_y    = kp_y.cpu().numpy()

            row_out = []
            for j in range(num_joints):
                pts2d  = np.stack([kp_x[:,j], kp_y[:,j]], axis=1)
                confs  = conf_kp[:,j]
                pt3d   = triangulate_dlt(pts2d, Ps_np[det_idx],
                                         confs, min_conf=kp_conf_threshold,
                                         max_reproj_err=max_reproj_err)
                if pt3d is not None:
                    row_out += [float(pt3d[0]), float(pt3d[1]), float(pt3d[2]),
                                float(confs.mean())]
                else:
                    row_out += ['NaN', 'NaN', 'NaN', 'NaN']
            writer.writerow(row_out)

        for cap in caps:
            cap.release()
        csvfile.close()
        elapsed = time.perf_counter() - t0
        fps = num_frames / elapsed
        print(f"         {fps:.2f} fps  ({elapsed:.1f}s)")

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project',        required=True)
    p.add_argument('--recording-path', required=True)
    p.add_argument('--calib-dir',      required=True,
                   help='Path to JARVIS-format calib YAML directory (calib_params/YYYY_MM_DD)')
    p.add_argument('--trials-csv',     required=True)
    p.add_argument('--output-subdir',  required=True,
                   help='Subdirectory under predictions/predictions3D/ (e.g. rory_2026_01_07_…)')
    p.add_argument('--trt',            action='store_true')
    p.add_argument('--center-threshold', type=float, default=40.0)
    p.add_argument('--max-reproj-err',   type=float, default=200.0)
    p.add_argument('--start-col',        default='frame_id_start')
    p.add_argument('--end-col',          default='frame_id_end')
    p.add_argument('--trial-index-col',  default='trial_index')
    args = p.parse_args()

    ok = run_session(
        project_name=args.project,
        recording_path=args.recording_path,
        calib_dir=args.calib_dir,
        trials_csv=args.trials_csv,
        output_subdir=args.output_subdir,
        use_trt=args.trt,
        center_threshold=args.center_threshold,
        max_reproj_err=args.max_reproj_err,
        start_col=args.start_col,
        end_col=args.end_col,
        trial_index_col=args.trial_index_col,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
