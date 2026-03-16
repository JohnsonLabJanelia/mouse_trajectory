"""
Validate mouseHybrid24 24-keypoint detections across recordings and calibrations.

For each (recording, calib) pair, samples frames at 10%, 50%, 90% of the recording
and generates a camera-grid image with 24 keypoints drawn.

Usage:
    conda run -n jarvis python3 validate_24kp.py
    conda run -n jarvis python3 validate_24kp.py --out-dir /tmp/kp_validation
"""
import argparse
import math
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/user/src/JARVIS-HybridNet')
from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack

# -----------------------------------------------------------------------
# Skeleton connections for the 24-kp mouseJan30 layout
# (pairs of keypoint names that should be connected with a line)
# -----------------------------------------------------------------------
SKELETON = [
    ('Snout',    'Neck'),
    ('Neck',     'EarL'),
    ('Neck',     'EarR'),
    ('Neck',     'SpineL'),
    ('SpineL',   'TailBase'),
    ('TailBase', 'Tail1Q'),
    ('Tail1Q',   'TailMid'),
    ('TailMid',  'Tail3Q'),
    ('Tail3Q',   'TailTip'),
    # Left forelimb
    ('Neck',      'ShoulderL'),
    ('ShoulderL', 'ElbowL'),
    ('ElbowL',    'WristL'),
    ('WristL',    'HandL'),
    # Right forelimb
    ('Neck',      'ShoulderR'),
    ('ShoulderR', 'ElbowR'),
    ('ElbowR',    'WristR'),
    ('WristR',    'HandR'),
    # Left hindlimb
    ('TailBase',  'KneeL'),
    ('KneeL',     'AnkleL'),
    ('AnkleL',    'FootL'),
    # Right hindlimb
    ('TailBase',  'KneeR'),
    ('KneeR',     'AnkleR'),
    ('AnkleR',    'FootR'),
]

# -----------------------------------------------------------------------
# Recordings to test: (recording_path, calib_dir, label)
# -----------------------------------------------------------------------
CALIB_BASE = '/home/user/src/analyzeMiceTrajectory/calib_params'
VIDEO_BASE  = '/mnt/mouse2/rory'

TEST_CASES = [
    (f'{VIDEO_BASE}/2025_12_23_16_57_09', f'{CALIB_BASE}/2025_12_22', 'Dec23_calib22'),
    (f'{VIDEO_BASE}/2025_12_24_13_29_06', f'{CALIB_BASE}/2025_12_24', 'Dec24_calib24'),
    (f'{VIDEO_BASE}/2026_01_07_17_31_36', f'{CALIB_BASE}/2026_01_07', 'Jan07_calib07'),
    (f'{VIDEO_BASE}/2026_02_06_16_59_10', f'{CALIB_BASE}/2026_01_08', 'Feb06_calib08'),
]

# Sample positions within the recording (fraction of total frames)
SAMPLE_FRACS = [0.10, 0.50, 0.90]

TRT_DIR = '/home/user/src/JARVIS-HybridNet/projects/mouseHybrid24/trt-models/predict2D'


def conf_color(c):
    """Map confidence [0,1] to BGR color: red=low, yellow=mid, green=high."""
    if c >= 0.5:
        g = 255
        r = int(255 * (1 - (c - 0.5) / 0.5))
        return (0, g, r)
    else:
        r = 255
        g = int(255 * (c / 0.5))
        return (0, g, r)


def detect_frame(img, cfg, cd_model, kd_model, mean, std, img_size_cd, bbox_size, bbox_hw):
    """Run full 2-stage detection on one image. Returns (kp_x, kp_y, confs, cx, cy, maxval)."""
    img_h, img_w = img.shape[:2]
    img_t = torch.from_numpy(img).cuda().float().permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0) / 255.0

    with torch.no_grad():
        img_cd = F.interpolate(img_t, size=(img_size_cd,)*2, mode='bilinear', align_corners=False)
        img_cd = (img_cd - mean) / std
        out = cd_model(img_cd)
        hm = out[1].view(-1)
        maxval = hm.max().item()
        m = hm.argmax().item()
        cx = int((m % out[1].shape[3]) * (img_w / out[1].shape[3]))
        cy = int((m // out[1].shape[3]) * (img_h / out[1].shape[2]))

        if maxval <= 40:
            return None, None, None, cx, cy, maxval

        cx_c = max(bbox_hw, min(img_w - bbox_hw, cx))
        cy_c = max(bbox_hw, min(img_h - bbox_hw, cy))
        crop = img_t[:, :, cy_c-bbox_hw:cy_c+bbox_hw, cx_c-bbox_hw:cx_c+bbox_hw]
        crop = (crop - mean) / std
        ko = kd_model(crop)

        J = ko[1].shape[1]
        hm_kp = ko[1].view(1, J, -1)
        m_kp  = hm_kp.argmax(2)
        confs = (hm_kp.gather(2, m_kp.unsqueeze(-1)).squeeze() / 255.).cpu().numpy()
        kp_x  = ((m_kp % ko[1].shape[3]).float() * 2 + (cx_c - bbox_hw)).squeeze().cpu().numpy()
        kp_y  = ((m_kp // ko[1].shape[3]).float() * 2 + (cy_c - bbox_hw)).squeeze().cpu().numpy()

    return kp_x, kp_y, confs, cx, cy, maxval


def draw_kp(img_vis, kp_x, kp_y, confs, kp_names, cx, cy, bbox_size, scale=1.0):
    """Draw keypoints + skeleton + center box on img_vis (in-place)."""
    bbox_hw = bbox_size // 2
    cx_c = max(bbox_hw, min(img_vis.shape[1] - bbox_hw, cx))
    cy_c = max(bbox_hw, min(img_vis.shape[0] - bbox_hw, cy))
    cv2.rectangle(img_vis, (cx_c-bbox_hw, cy_c-bbox_hw), (cx_c+bbox_hw, cy_c+bbox_hw), (200,200,0), 2)
    cv2.circle(img_vis, (cx, cy), 12, (255, 255, 0), 3)

    # Index lookup for skeleton connections
    name2idx = {n: i for i, n in enumerate(kp_names)}

    # Skeleton lines (drawn first so kp circles appear on top)
    for (na, nb) in SKELETON:
        ia, ib = name2idx.get(na), name2idx.get(nb)
        if ia is None or ib is None:
            continue
        ca, cb = confs[ia], confs[ib]
        if ca < 0.05 or cb < 0.05:
            continue
        pt_a = (int(kp_x[ia]), int(kp_y[ia]))
        pt_b = (int(kp_x[ib]), int(kp_y[ib]))
        col = conf_color((ca + cb) / 2)
        cv2.line(img_vis, pt_a, pt_b, col, max(1, int(2*scale)))

    # Keypoint circles + labels
    for i, name in enumerate(kp_names):
        x, y, c = int(kp_x[i]), int(kp_y[i]), confs[i]
        col = conf_color(c)
        r = max(4, int(8 * scale))
        cv2.circle(img_vis, (x, y), r, col, -1)
        cv2.putText(img_vis, name[:4], (x + r + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, col, max(1, int(scale)))


def make_grid(frames_kp, cam_names, grid_cols=4, thumb_w=640, thumb_h=400):
    """
    frames_kp: list of (cam_name, img_vis) tuples
    Returns a grid image.
    """
    n = len(frames_kp)
    cols = min(grid_cols, n)
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, (cam, img) in enumerate(frames_kp):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        # Annotate camera name
        cv2.putText(thumb, cam, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        y0, x0 = r * thumb_h, c * thumb_w
        grid[y0:y0+thumb_h, x0:x0+thumb_w] = thumb
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp/kp_validation')
    ap.add_argument('--no-trt', action='store_true', help='Use PyTorch models instead of TRT')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load model ----
    p = ProjectManager(); p.load('mouseHybrid24'); cfg = p.cfg
    kp_names   = cfg.KEYPOINT_NAMES
    img_size_cd = cfg.CENTERDETECT.IMAGE_SIZE
    bbox_size   = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    bbox_hw     = bbox_size // 2
    mean = torch.tensor(cfg.DATASET.MEAN, device='cuda').view(3,1,1)
    std  = torch.tensor(cfg.DATASET.STD,  device='cuda').view(3,1,1)

    use_trt = (not args.no_trt) and os.path.isdir(TRT_DIR)
    if use_trt:
        import torch_tensorrt  # noqa
        print("Loading TRT models ...")
        cd_model = torch.jit.load(os.path.join(TRT_DIR, 'centerDetect.pt')).cuda()
        kd_model = torch.jit.load(os.path.join(TRT_DIR, 'keypointDetect.pt')).cuda()
    else:
        print("Loading PyTorch models ...")
        cd_model = EfficientTrack('CenterDetectInference',   cfg, 'latest').model.eval().cuda()
        kd_model = EfficientTrack('KeypointDetectInference', cfg, 'latest').model.eval().cuda()

    # ---- Warmup ----
    with torch.no_grad():
        _ = cd_model(torch.zeros(1, 3, img_size_cd, img_size_cd, device='cuda'))
        _ = kd_model(torch.zeros(1, 3, bbox_size, bbox_size, device='cuda'))

    summary_rows = []

    for rec_path, calib_dir, label in TEST_CASES:
        if not os.path.isdir(rec_path):
            print(f"  SKIP {label}: recording not found at {rec_path}")
            continue
        if not os.path.isdir(calib_dir):
            print(f"  SKIP {label}: calibration not found at {calib_dir}")
            continue

        # Find camera videos (skip Cam710040 — different resolution)
        cam_files = sorted([f for f in os.listdir(rec_path)
                            if f.endswith('.mp4') and 'Cam710040' not in f])
        cam_names = [f.split('.')[0] for f in cam_files]
        if not cam_names:
            print(f"  SKIP {label}: no .mp4 files")
            continue

        # Get total frame count from first camera
        cap0 = cv2.VideoCapture(os.path.join(rec_path, cam_files[0]))
        total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap0.get(cv2.CAP_PROP_FPS)
        cap0.release()

        print(f"\n{'='*60}")
        print(f"Recording: {label}")
        print(f"  Path: {rec_path}")
        print(f"  Calib: {calib_dir}")
        print(f"  Cameras: {len(cam_names)}  Frames: {total_frames}  ({total_frames/fps/60:.1f} min)")

        for frac in SAMPLE_FRACS:
            frame_idx = int(total_frames * frac)
            print(f"\n  -- Frame {frame_idx} ({frac*100:.0f}% of recording) --")

            panels = []
            det_count = 0
            conf_vals = []

            for cam_name, cam_file in zip(cam_names, cam_files):
                cap = cv2.VideoCapture(os.path.join(rec_path, cam_file))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, img = cap.read()
                cap.release()
                if not ret or img is None:
                    continue

                kp_x, kp_y, confs, cx, cy, maxval = detect_frame(
                    img, cfg, cd_model, kd_model, mean, std,
                    img_size_cd, bbox_size, bbox_hw)

                img_vis = img.copy()
                if kp_x is not None:
                    draw_kp(img_vis, kp_x, kp_y, confs, kp_names, cx, cy, bbox_size)
                    det_count += 1
                    conf_vals.extend(confs.tolist())
                    status = f"det conf={confs.mean():.2f}"
                else:
                    # No detection — draw center guess location
                    cv2.circle(img_vis, (cx, cy), 20, (0, 0, 255), 3)
                    cv2.putText(img_vis, f"NO DET maxval={maxval:.0f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    status = f"NO DET maxval={maxval:.0f}"

                print(f"    {cam_name}: {status}")
                panels.append((cam_name, img_vis))

            if conf_vals:
                mean_conf = np.mean(conf_vals)
                high_conf = np.mean(np.array(conf_vals) >= 0.5)
            else:
                mean_conf = 0.0
                high_conf = 0.0

            summary_rows.append({
                'label': label, 'frame': frame_idx, 'frac': frac,
                'detected': det_count, 'total_cams': len(cam_names),
                'mean_conf': mean_conf, 'pct_high_conf': high_conf * 100,
            })
            print(f"    Detected: {det_count}/{len(cam_names)} cameras  "
                  f"mean_conf={mean_conf:.3f}  pct_conf≥0.5={high_conf*100:.0f}%")

            # Save camera grid
            grid = make_grid(panels, cam_names, grid_cols=4, thumb_w=600, thumb_h=375)
            fname = f"{label}_frame{frame_idx:07d}_{frac*100:.0f}pct.jpg"
            out_path = os.path.join(args.out_dir, fname)
            cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"    Saved: {out_path}")

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'Recording':<22} {'Frame':>8} {'%':>4} | {'Det':>5} {'Cams':>5} | {'MeanConf':>9} {'%Conf≥0.5':>10}")
    print("-"*70)
    for r in summary_rows:
        print(f"{r['label']:<22} {r['frame']:>8} {r['frac']*100:>3.0f}% | "
              f"{r['detected']:>5}/{r['total_cams']:<4} | "
              f"{r['mean_conf']:>9.3f} {r['pct_high_conf']:>9.0f}%")

    print(f"\nAll visualizations saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
