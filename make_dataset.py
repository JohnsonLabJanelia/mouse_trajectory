#!/usr/bin/env python3
"""
Generate the 24-keypoint 3D dataset for rory and wilfred.

Loads mouseHybrid24 TRT models ONCE, then iterates all sessions and trials
from the trial CSVs, running 2D+DLT prediction for every trial window.

Raw predictions land in:
    dataset/predictions_raw/{animal}_{video_folder}/
        trial_{idx:04d}_{start}-{end}/
            data3D.csv

After predictions, filter_dataset.py collapses those into per-animal CSVs
with arena-mask gating applied.

Usage:
    conda run -n jarvis python3 make_dataset.py
    conda run -n jarvis python3 make_dataset.py --animals rory --stride 2
    conda run -n jarvis python3 make_dataset.py --skip-done       # resume

Arguments:
    --animals       which animals to process (default: rory wilfred)
    --trials-dir    directory containing {animal}_trials.csv files (default: /tmp)
    --video-root    root directory of video recordings (default: /mnt/mouse2)
    --calib-root    root of per-date calibration dirs (default: calib_params/)
    --output-dir    where to save predictions_raw/ (default: dataset/)
    --stride        only predict every Nth frame (default: 1)
    --skip-done     skip trials that already have a data3D.csv
    --no-trt        use PyTorch models instead of TRT
"""
import argparse
import csv
import itertools
import os
import sys
import time
import logging
from datetime import datetime, date
from pathlib import Path

import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

# decord: fast video I/O with optional NVDEC hardware decode.
# Falls back to cv2 if not installed.
try:
    import decord as _decord
    _decord.bridge.set_bridge('torch')
    HAVE_DECORD = True
except ImportError:
    HAVE_DECORD = False

SCRIPT_DIR = Path(__file__).resolve().parent

# Support running from any machine — find JARVIS relative to home or script
_JARVIS_CANDIDATES = [
    Path.home() / 'JARVIS-HybridNet',
    SCRIPT_DIR.parent / 'JARVIS-HybridNet',
    Path('/home/user/src/JARVIS-HybridNet'),
]
JARVIS_DIR = next((p for p in _JARVIS_CANDIDATES if p.is_dir()), _JARVIS_CANDIDATES[0])
sys.path.insert(0, str(JARVIS_DIR))

from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from predict2D_triangulate import load_all_calibrations, triangulate_dlt

PROJECT      = 'mouseHybrid24'
TRT_DIR      = str(JARVIS_DIR / 'projects' / PROJECT / 'trt-models' / 'predict2D')
CALIB_IGNORE = {'2025_12_19', '2025_12_21'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def list_calib_dates(calib_root: Path):
    """Return sorted list of (date, path) for valid calibration directories."""
    out = []
    for p in calib_root.iterdir():
        if not p.is_dir() or p.name in CALIB_IGNORE:
            continue
        try:
            d = datetime.strptime(p.name, '%Y_%m_%d').date()
            if any(p.glob('*.yaml')):
                out.append((d, p))
        except ValueError:
            continue
    return sorted(out, key=lambda t: t[0])


def closest_calib(video_folder: str, calib_dates):
    try:
        vd = datetime.strptime(video_folder[:19], '%Y_%m_%d_%H_%M_%S').date()
    except ValueError:
        return None
    if not calib_dates:
        return None
    return min(calib_dates, key=lambda t: abs((t[0] - vd).days))[1]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(use_trt: bool):
    p = ProjectManager()
    if not p.load(PROJECT):
        sys.exit(f'Cannot load project {PROJECT}')
    cfg = p.cfg

    if use_trt and os.path.isdir(TRT_DIR):
        import torch_tensorrt  # noqa
        log.info('Loading TRT models ...')
        cd = torch.jit.load(os.path.join(TRT_DIR, 'centerDetect.pt')).cuda()
        kd = torch.jit.load(os.path.join(TRT_DIR, 'keypointDetect.pt')).cuda()
    else:
        if use_trt:
            log.warning('TRT models not found, falling back to PyTorch')
        cd = EfficientTrack('CenterDetectInference',  cfg, 'latest').model.eval().cuda()
        kd = EfficientTrack('KeypointDetectInference', cfg, 'latest').model.eval().cuda()

    mean = torch.tensor(cfg.DATASET.MEAN, device='cuda').view(3, 1, 1)
    std  = torch.tensor(cfg.DATASET.STD,  device='cuda').view(3, 1, 1)
    log.info('Models ready.')
    return cfg, cd, kd, mean, std


# ---------------------------------------------------------------------------
# Per-trial prediction
# ---------------------------------------------------------------------------

def _read_frame_parallel(caps, buf):
    """cv2 fallback: read one frame from all cameras using a thread pool."""
    def _r(args):
        cap, i = args
        ret, img = cap.read()
        if ret and img is not None:
            buf[i] = img
        return ret
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(caps)) as ex:
        return all(ex.map(_r, [(cap, i) for i, cap in enumerate(caps)]))


def _iter_frames_prefetch(video_paths, img_h, img_w, frame_start, frame_end,
                           stride, prefetch=4):
    """
    Yield (ok, imgs_numpy) for each strided frame in [frame_start, frame_end].
    imgs_numpy: (N_cams, H, W, 3) uint8 numpy array.

    A background thread reads ahead `prefetch` frames using a per-camera
    ThreadPoolExecutor, so GPU inference and I/O run in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    num_cams = len(video_paths)
    caps = [cv2.VideoCapture(p) for p in video_paths]
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    pf_queue = queue.Queue(maxsize=prefetch)

    def _loader():
        with ThreadPoolExecutor(max_workers=num_cams) as ex:
            for abs_frame in range(frame_start, frame_end + 1, stride):
                buf = np.zeros((num_cams, img_h, img_w, 3), dtype=np.uint8)

                def read_one(i, _buf=buf):
                    ret, img = caps[i].read()
                    if ret and img is not None:
                        _buf[i] = img
                    return ret

                rets = list(ex.map(read_one, range(num_cams)))
                pf_queue.put((all(rets), buf))
        pf_queue.put(None)  # sentinel

    t = threading.Thread(target=_loader, daemon=True)
    t.start()

    while True:
        item = pf_queue.get()
        if item is None:
            break
        yield item

    t.join()
    for cap in caps:
        cap.release()


def _infer_one_frame(imgs_t, cd_model, kd_model, mean, std,
                     img_size_cd, bbox_size, img_h, img_w,
                     num_cameras, num_joints, Ps_np,
                     center_threshold, max_reproj_err):
    """
    Run CenterDetect + KeypointDetect + DLT on a single (N_cams, 3, H, W)
    float CUDA tensor.  Returns a flat CSV row or None on no detection.
    """
    bbox_hw = bbox_size // 2
    with torch.no_grad():
        imgs_cd = F.interpolate(imgs_t, (img_size_cd,) * 2,
                                mode='bilinear', align_corners=False)
        cd_out  = cd_model((imgs_cd - mean) / std)
        hm      = cd_out[1].view(num_cameras, -1)
        m       = hm.argmax(1)
        maxvals = hm[range(num_cameras), m]
        cx = (m % cd_out[1].shape[3]).float() * (img_w / cd_out[1].shape[3])
        cy = (m // cd_out[1].shape[3]).float() * (img_h / cd_out[1].shape[2])
        cx = cx.long().clamp(bbox_hw, img_w - bbox_hw)
        cy = cy.long().clamp(bbox_hw, img_h - bbox_hw)

        det_idx = torch.where(maxvals > center_threshold)[0].cpu().numpy()
        if len(det_idx) < 2:
            return ['NaN'] * (num_joints * 4)

        crops = torch.stack([
            imgs_t[i, :, cy[i] - bbox_hw: cy[i] + bbox_hw,
                         cx[i] - bbox_hw: cx[i] + bbox_hw]
            for i in det_idx])
        kp_out  = kd_model((crops - mean) / std)
        hm_kp   = kp_out[1].view(len(det_idx), num_joints, -1)
        m_kp    = hm_kp.argmax(2)
        conf_kp = (hm_kp.gather(2, m_kp.unsqueeze(-1)).squeeze(-1)
                   .clamp(max=255.) / 255.).cpu().numpy()
        kp_x    = (m_kp % kp_out[1].shape[3]).float() * 2
        kp_y    = (m_kp // kp_out[1].shape[3]).float() * 2
        kp_x   += (cx[det_idx] - bbox_hw).float().unsqueeze(1)
        kp_y   += (cy[det_idx] - bbox_hw).float().unsqueeze(1)
        kp_x    = kp_x.cpu().numpy()
        kp_y    = kp_y.cpu().numpy()

    row = []
    for j in range(num_joints):
        pts2d = np.stack([kp_x[:, j], kp_y[:, j]], axis=1)
        confs = conf_kp[:, j]
        pt3d  = triangulate_dlt(pts2d, Ps_np[det_idx], confs,
                                max_reproj_err=max_reproj_err)
        if pt3d is not None:
            row += [float(pt3d[0]), float(pt3d[1]), float(pt3d[2]),
                    float(confs.mean())]
        else:
            row += ['NaN', 'NaN', 'NaN', 'NaN']
    return row


def predict_trial(
    cfg, cd_model, kd_model, mean, std,
    recording_path, calib_dir, camera_names,
    Ps_np, frame_start, frame_end, output_dir,
    center_threshold=40.0, max_reproj_err=200.0, stride=1,
):
    """Predict one trial and write data3D.csv to output_dir."""
    img_size_cd = cfg.CENTERDETECT.IMAGE_SIZE
    bbox_size   = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    num_joints  = cfg.KEYPOINTDETECT.NUM_JOINTS
    num_cameras = len(camera_names)

    # Resolve video paths (camera_name.mp4 inside recording_path)
    video_paths = []
    for cn in camera_names:
        for fname in os.listdir(recording_path):
            if fname.split('.')[0] == cn and fname.endswith('.mp4'):
                video_paths.append(os.path.join(recording_path, fname))
                break
    if len(video_paths) != num_cameras:
        log.error(f'Expected {num_cameras} cameras, found {len(video_paths)}')
        return False

    # Get frame dimensions from the first video
    _cap = cv2.VideoCapture(video_paths[0])
    img_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _cap.release()

    os.makedirs(output_dir, exist_ok=True)
    csvfile = open(os.path.join(output_dir, 'data3D.csv'), 'w', newline='')
    writer  = csv.writer(csvfile)

    kp_names = cfg.KEYPOINT_NAMES
    writer.writerow(list(itertools.chain.from_iterable(
        itertools.repeat(kp, 4) for kp in kp_names)))
    writer.writerow(['x', 'y', 'z', 'confidence'] * num_joints)

    frames_written = 0

    for ok, imgs_buf in _iter_frames_prefetch(
            video_paths, img_h, img_w, frame_start, frame_end, stride):
        if not ok:
            writer.writerow(['NaN'] * (num_joints * 4))
            frames_written += 1
            continue
        imgs_t = (torch.from_numpy(imgs_buf).to('cuda', non_blocking=True)
                  .float().permute(0, 3, 1, 2)[:, [2, 1, 0]] / 255.0)
        row = _infer_one_frame(
            imgs_t, cd_model, kd_model, mean, std,
            img_size_cd, bbox_size, img_h, img_w,
            num_cameras, num_joints, Ps_np,
            center_threshold, max_reproj_err)
        writer.writerow(row)
        frames_written += 1

    csvfile.close()
    return True


# ---------------------------------------------------------------------------
# Session CSV helpers
# ---------------------------------------------------------------------------

def load_trials_csv(csv_path: Path):
    """Return list of trial dicts sorted by (video_folder, trial_index)."""
    rows = list(csv.DictReader(open(csv_path)))
    rows.sort(key=lambda r: (r['video_folder'], int(r.get('trial_index', 0))))
    return rows


def group_by_session(trials):
    """Group trial rows by video_folder."""
    by_session = {}
    for t in trials:
        by_session.setdefault(t['video_folder'], []).append(t)
    return by_session


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--animals',     nargs='+', default=['rory', 'wilfred'])
    ap.add_argument('--session',     type=str,  default=None,
                    help='Process only this one session, format: animal/video_folder '
                         '(e.g. rory/2026_01_07_17_31_36). Used by cluster array jobs.')
    ap.add_argument('--trials-dir',  type=Path, default=SCRIPT_DIR / 'dataset')
    ap.add_argument('--video-root',  type=Path, default=Path('/mnt/mouse2'),
                    help='Root dir containing animal/session subdirs with .mp4 files')
    ap.add_argument('--calib-root',  type=Path, default=SCRIPT_DIR / 'calib_params')
    ap.add_argument('--output-dir',  type=Path, default=SCRIPT_DIR / 'dataset')
    ap.add_argument('--stride',      type=int,  default=1,
                    help='Predict every Nth frame (1=all, 3=60fps at 180fps camera)')
    ap.add_argument('--skip-done',   action='store_true',
                    help='Skip trials that already have data3D.csv')
    ap.add_argument('--no-trt',      action='store_true')
    ap.add_argument('--center-threshold', type=float, default=40.0)
    ap.add_argument('--max-reproj-err',   type=float, default=200.0)
    args = ap.parse_args()

    # --session animal/video_folder restricts to a single session (cluster jobs)
    session_filter = None
    if args.session:
        parts = args.session.strip('/').split('/', 1)
        if len(parts) != 2:
            sys.exit('--session must be animal/video_folder, e.g. rory/2026_01_07_17_31_36')
        session_filter = (parts[0], parts[1])
        args.animals = [parts[0]]

    raw_dir     = args.output_dir / 'predictions_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Log file alongside raw predictions
    log_path = args.output_dir / 'make_dataset.log'
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(fh)

    log.info(f'Animals: {args.animals}  stride={args.stride}  skip_done={args.skip_done}')

    calib_dates = list_calib_dates(args.calib_root)
    log.info(f'Calibration dates available: {[str(d) for d, _ in calib_dates]}')

    # Load models ONCE
    cfg, cd_model, kd_model, mean, std = load_models(not args.no_trt)

    # Write sessions metadata CSV
    sessions_csv_path = args.output_dir / 'sessions.csv'
    sessions_written = set()
    if sessions_csv_path.exists():
        for row in csv.DictReader(open(sessions_csv_path)):
            sessions_written.add((row['animal'], row['video_folder']))
    sessions_meta_fh = open(sessions_csv_path, 'a', newline='')
    sessions_writer = csv.DictWriter(sessions_meta_fh,
        fieldnames=['animal', 'video_folder', 'calib_date', 'video_path',
                    'calib_dir', 'n_trials', 'fps'])
    if not sessions_written:
        sessions_writer.writeheader()

    t_pipeline_start = time.perf_counter()
    total_frames_done = 0

    for animal in args.animals:
        trials_csv = args.trials_dir / f'{animal}_trials.csv'
        if not trials_csv.exists():
            log.warning(f'No trials CSV for {animal} at {trials_csv}, skipping')
            continue

        trials = load_trials_csv(trials_csv)
        by_session = group_by_session(trials)
        log.info(f'=== {animal}: {len(trials)} trials across {len(by_session)} sessions ===')

        for session_idx, (video_folder, session_trials) in enumerate(sorted(by_session.items())):
            if session_filter and (animal, video_folder) != session_filter:
                continue
            rec_path = args.video_root / animal / video_folder
            if not rec_path.is_dir():
                log.warning(f'  Skip {video_folder}: recording dir not found')
                continue

            calib_dir = closest_calib(video_folder, calib_dates)
            if calib_dir is None:
                log.warning(f'  Skip {video_folder}: no calibration found')
                continue

            # Calibration: cameras present in calib_dir
            cam_files = sorted([f for f in os.listdir(rec_path)
                                 if f.endswith('.mp4') and 'Cam710040' not in f])
            camera_names = [f.split('.')[0] for f in cam_files]
            # Only keep cameras that have a calib yaml
            camera_names = [cn for cn in camera_names
                            if (calib_dir / f'{cn}.yaml').exists()]
            if len(camera_names) < 4:
                log.warning(f'  Skip {video_folder}: only {len(camera_names)} calibrated cameras')
                continue

            _, _, _, _, Ps_np = load_all_calibrations(str(calib_dir), camera_names)
            Ps_np = Ps_np.astype(np.float32)

            # Get FPS from first camera
            cap0 = cv2.VideoCapture(str(rec_path / cam_files[0]))
            fps  = cap0.get(cv2.CAP_PROP_FPS)
            cap0.release()

            session_key = f'{animal}_{video_folder}'
            session_out = raw_dir / session_key
            session_out.mkdir(parents=True, exist_ok=True)

            if (animal, video_folder) not in sessions_written:
                sessions_writer.writerow({
                    'animal': animal, 'video_folder': video_folder,
                    'calib_date': calib_dir.name,
                    'video_path': str(rec_path),
                    'calib_dir': str(calib_dir),
                    'n_trials': len(session_trials),
                    'fps': fps,
                })
                sessions_meta_fh.flush()
                sessions_written.add((animal, video_folder))

            n_trials = len(session_trials)
            log.info(f'  [{session_idx+1}/{len(by_session)}] {video_folder}  '
                     f'calib={calib_dir.name}  cameras={len(camera_names)}  '
                     f'trials={n_trials}')

            for trial_i, trial in enumerate(session_trials):
                try:
                    frame_start = int(trial['frame_id_start'])
                    frame_end   = int(trial['frame_id_end'])
                    trial_idx   = int(trial.get('trial_index', trial_i))
                except (ValueError, KeyError):
                    continue

                num_frames = frame_end - frame_start + 1
                if num_frames < 2:
                    continue

                trial_name = f'trial_{trial_idx:04d}_{frame_start}-{frame_end}'
                trial_out  = session_out / trial_name
                done_file  = trial_out / 'data3D.csv'

                if args.skip_done and done_file.exists():
                    log.info(f'    [{trial_i+1}/{n_trials}] {trial_name}: already done, skipping')
                    continue

                frames_to_predict = (num_frames + args.stride - 1) // args.stride
                log.info(f'    [{trial_i+1}/{n_trials}] {trial_name}  '
                         f'{num_frames} frames -> {frames_to_predict} predictions  '
                         f'(stride={args.stride})')

                t0 = time.perf_counter()
                ok = predict_trial(
                    cfg, cd_model, kd_model, mean, std,
                    str(rec_path), str(calib_dir), camera_names, Ps_np,
                    frame_start, frame_end, str(trial_out),
                    center_threshold=args.center_threshold,
                    max_reproj_err=args.max_reproj_err,
                    stride=args.stride,
                )
                elapsed = time.perf_counter() - t0
                total_frames_done += num_frames

                if ok:
                    pred_fps = frames_to_predict / elapsed
                    elapsed_total = time.perf_counter() - t_pipeline_start
                    log.info(f'      done: {pred_fps:.1f} fps  ({elapsed:.0f}s)  '
                             f'pipeline running {elapsed_total/3600:.2f}h')
                else:
                    log.error(f'      FAILED: {trial_name}')

    sessions_meta_fh.close()
    total_elapsed = time.perf_counter() - t_pipeline_start
    log.info(f'Predictions complete. Total time: {total_elapsed/3600:.2f}h')
    log.info(f'Raw predictions at: {raw_dir}')
    log.info(f'Next step: python3 filter_dataset.py')


if __name__ == '__main__':
    main()
