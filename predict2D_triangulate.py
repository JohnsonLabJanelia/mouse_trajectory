"""
Batched 2D prediction + DLT triangulation pipeline.

Uses the 24-keypoint model's CenterDetect + KeypointDetect (no HybridNet),
then triangulates 3D positions from all camera views using the same DLT
approach as JARVIS's ReprojectionTool.

Key speed improvements over standard predict3D:
- CenterDetect runs on ALL cameras in a single batched forward pass.
- KeypointDetect runs on all detected cameras in one batched forward pass.
- No volumetric HybridNet step.

Output: data3D.csv with the same header/column format as predict3D output.

Usage:
    python predict2D_triangulate.py \\
        --project mouseJan30 \\
        --recording-path /mnt/mouse2/rory/2025_12_23_16_57_09 \\
        --calib-dir /home/user/src/analyzeMiceTrajectory/calib_params/2025_12_22 \\
        --frame-start 15186 --num-frames 100 \\
        --output-dir /path/to/output_dir

    # Benchmark mode (--tag writes timing.txt):
    python predict2D_triangulate.py --project mouseJan30 ... --tag jan30_2d
"""
import argparse
import csv
import itertools
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, '/home/user/src/JARVIS-HybridNet')
from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack

# -------------------------------------------------------------------
# Calibration helpers
# -------------------------------------------------------------------

def load_camera_calib(calib_yaml):
    """Load intrinsicMatrix, R, T, distortionCoefficients from OpenCV YAML."""
    fs = cv2.FileStorage(calib_yaml, cv2.FILE_STORAGE_READ)
    K = fs.getNode('intrinsicMatrix').mat()        # 3x3 (transposed K in JARVIS convention)
    R = fs.getNode('R').mat()                      # 3x3 rotation
    T = fs.getNode('T').mat()                      # 3x1 translation
    dist = fs.getNode('distortionCoefficients').mat()  # 1x5
    fs.release()
    # Projection matrix in JARVIS convention: P = (vstack([R, T.T]) @ K).T  -> 3x4
    RT = np.vstack([R, T.reshape(1, 3)])           # 4x3
    P_jarvis = (RT @ K).T                          # 3x4  (JARVIS transposed convention)
    return K, R, T.reshape(3), dist.reshape(-1), P_jarvis


def load_all_calibrations(calib_dir, camera_names):
    """Return lists of K, R, T, dist, P (numpy) in camera_names order."""
    Ks, Rs, Ts, dists, Ps = [], [], [], [], []
    for cam in camera_names:
        yaml_path = os.path.join(calib_dir, cam + '.yaml')
        K, R, T, dist, P = load_camera_calib(yaml_path)
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
        dists.append(dist)
        Ps.append(P)
    return (np.array(Ks), np.array(Rs), np.array(Ts),
            np.array(dists), np.array(Ps))


# -------------------------------------------------------------------
# DLT triangulation  (same algebra as JARVIS ReprojectionTool.reconstructPoint)
# -------------------------------------------------------------------

def _dlt_core(pts, Ps, ws):
    """Inner DLT solver: pts (M,2), Ps (M,3,4), ws (M,) confidence weights."""
    rows = []
    for i in range(len(pts)):
        u, v = pts[i]
        rows.append((u * Ps[i, 2] - Ps[i, 0]) * ws[i])
        rows.append((v * Ps[i, 2] - Ps[i, 1]) * ws[i])
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def triangulate_dlt(points2d, proj_matrices, confidences, min_conf=0.0,
                    max_reproj_err=200.0, max_iters=5):
    """
    Robust DLT triangulation with iterative outlier rejection.

    points2d:    (N, 2) float  - 2D pixel coordinates in each camera
    proj_matrices: (N, 3, 4) float - projection matrices in JARVIS convention
    confidences: (N,) float  - per-camera confidence weights
    min_conf:    float       - cameras below this confidence are excluded
    max_reproj_err: float    - cameras with reprojection error above this (px)
                               are dropped and triangulation is re-run
    max_iters:   int         - max outlier-rejection iterations

    Returns 3D point (3,) in world coordinates, or None if < 2 good cameras.
    """
    mask = confidences >= min_conf
    if mask.sum() < 2:
        return None

    for _ in range(max_iters):
        pts = points2d[mask]
        Ps  = proj_matrices[mask]
        ws  = confidences[mask]
        if len(pts) < 2:
            return None

        X = _dlt_core(pts, Ps, ws)

        # Compute reprojection errors for all currently-included cameras
        X4 = np.append(X, 1.0)
        reproj = Ps @ X4          # (M, 3)
        reproj /= reproj[:, 2:3]  # homogeneous divide
        errs = np.sqrt(((pts - reproj[:, :2]) ** 2).sum(axis=1))

        bad_local = errs > max_reproj_err
        if not bad_local.any():
            break
        # Map local bad indices back to global mask
        global_idx = np.where(mask)[0]
        mask[global_idx[bad_local]] = False
        if mask.sum() < 2:
            return None

    return X


# -------------------------------------------------------------------
# Video helpers
# -------------------------------------------------------------------

def open_videos(recording_path, camera_names):
    caps = []
    for cam in camera_names:
        for fname in os.listdir(recording_path):
            if fname.split('.')[0] == cam:
                cap = cv2.VideoCapture(os.path.join(recording_path, fname))
                caps.append(cap)
                break
    assert len(caps) == len(camera_names), "Missing videos for some cameras"
    return caps


def read_frame_parallel(caps, imgs_buf):
    """Read one frame from each cap into imgs_buf (N, H, W, 3) in-place."""
    def _read(cap, idx):
        ret, img = cap.read()
        if ret and img is not None:
            imgs_buf[idx] = img
        return ret
    results = Parallel(n_jobs=len(caps), require='sharedmem')(
        delayed(_read)(cap, i) for i, cap in enumerate(caps))
    return all(results)


# -------------------------------------------------------------------
# Main prediction function
# -------------------------------------------------------------------

def predict2d_triangulate(
    project_name,
    recording_path,
    calib_dir,
    frame_start=0,
    num_frames=100,
    output_dir=None,
    center_threshold=40,
    kp_conf_threshold=0.0,
    max_reproj_err=200.0,
    weights_center='latest',
    weights_keypoint='latest',
    use_trt=False,
):
    # ---- Load project config + models ----
    project = ProjectManager()
    if not project.load(project_name):
        sys.exit(f"Cannot load project {project_name}")
    cfg = project.cfg

    trt_dir = os.path.join(
        '/home/user/src/JARVIS-HybridNet/projects', project_name,
        'trt-models', 'predict2D')

    if use_trt and os.path.isdir(trt_dir):
        import torch_tensorrt  # noqa: registers TRT backend
        print("[2D+tri] Loading TRT models ...")
        center_model   = torch.jit.load(os.path.join(trt_dir, 'centerDetect.pt')).cuda()
        keypoint_model = torch.jit.load(os.path.join(trt_dir, 'keypointDetect.pt')).cuda()
    else:
        if use_trt:
            print(f"[2D+tri] WARNING: TRT models not found at {trt_dir}, falling back to PyTorch")
        center_model  = EfficientTrack('CenterDetectInference',  cfg, weights_center).model.eval().cuda()
        keypoint_model = EfficientTrack('KeypointDetectInference', cfg, weights_keypoint).model.eval().cuda()

    # Pre-compute constants
    img_size_cd  = cfg.CENTERDETECT.IMAGE_SIZE
    bbox_size    = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    bbox_hw      = bbox_size // 2
    num_joints   = cfg.KEYPOINTDETECT.NUM_JOINTS
    mean = torch.tensor(cfg.DATASET.MEAN, device='cuda').view(3,1,1)
    std  = torch.tensor(cfg.DATASET.STD,  device='cuda').view(3,1,1)

    # ---- Find calibration files ----
    calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.yaml')])
    camera_names = [f.replace('.yaml', '') for f in calib_files]
    # Skip Cam710040 (different resolution, matches existing pipeline behaviour)
    camera_names = [c for c in camera_names if 'Cam710040' not in c]
    num_cameras = len(camera_names)
    print(f"[2D+tri] Using {num_cameras} cameras: {camera_names}")

    # ---- Load calibrations ----
    Ks, Rs, Ts, dists, Ps = load_all_calibrations(calib_dir, camera_names)
    Ps_np = Ps.astype(np.float32)   # (N, 3, 4) numpy — for triangulation

    # ---- Open videos ----
    caps = open_videos(recording_path, camera_names)
    # Seek to start frame
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    img_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    if output_dir is None:
        output_dir = os.path.join(
            '/home/user/src/JARVIS-HybridNet/projects', project_name,
            'predictions/predictions3D',
            f'Predictions_2Dtri_{time.strftime("%Y%m%d-%H%M%S")}'
        )
    os.makedirs(output_dir, exist_ok=True)

    # ---- Output CSV ----
    csvfile = open(os.path.join(output_dir, 'data3D.csv'), 'w', newline='')
    writer  = csv.writer(csvfile)
    if len(cfg.KEYPOINT_NAMES) == num_joints:
        joints = list(itertools.chain.from_iterable(
            itertools.repeat(kp, 4) for kp in cfg.KEYPOINT_NAMES))
        coords = ['x','y','z','confidence'] * num_joints
        writer.writerow(joints)
        writer.writerow(coords)

    # ---- Frame buffer ----
    imgs_buf = np.zeros((num_cameras, img_h, img_w, 3), dtype=np.uint8)

    # ---- Process frames ----
    for _ in tqdm(range(num_frames)):
        ok = read_frame_parallel(caps, imgs_buf)
        if not ok:
            writer.writerow(['NaN'] * (num_joints * 4))
            continue

        # (N, 3, H, W) float32 on GPU, BGR→RGB, normalized to [0,1]
        imgs_t = (torch.from_numpy(imgs_buf).cuda().float()
                  .permute(0, 3, 1, 2)[:, [2, 1, 0]] / 255.0)

        with torch.no_grad():
            # ---- CenterDetect: all cameras at once ----
            imgs_cd = F.interpolate(imgs_t,
                        size=(img_size_cd, img_size_cd), mode='bilinear',
                        align_corners=False)
            imgs_cd = (imgs_cd - mean) / std
            cd_out  = center_model(imgs_cd)                        # (N, 1, H', W')
            hm      = cd_out[1].view(num_cameras, -1)              # (N, H'*W')
            m       = hm.argmax(1)                                  # (N,)
            maxvals = hm[range(num_cameras), m]                    # (N,)
            cx = (m % cd_out[1].shape[3]).float() * (img_w / cd_out[1].shape[3])
            cy = (m // cd_out[1].shape[3]).float() * (img_h / cd_out[1].shape[2])
            cx = cx.long().clamp(bbox_hw, img_w - bbox_hw)
            cy = cy.long().clamp(bbox_hw, img_h - bbox_hw)

            detected = (maxvals > center_threshold).cpu().numpy()   # (N,) bool
            det_idx  = np.where(detected)[0]

            if len(det_idx) < 2:
                writer.writerow(['NaN'] * (num_joints * 4))
                continue

            # ---- Crop + KeypointDetect: detected cameras at once ----
            crops = torch.stack([
                imgs_t[i, :,
                       cy[i]-bbox_hw : cy[i]+bbox_hw,
                       cx[i]-bbox_hw : cx[i]+bbox_hw]
                for i in det_idx
            ])                                                      # (M, 3, bbox, bbox)
            crops = (crops - mean) / std
            kp_out = keypoint_model(crops)                          # (M, J, H'', W'')

            hm_kp  = kp_out[1].view(len(det_idx), num_joints, -1)  # (M, J, H''*W'')
            m_kp   = hm_kp.argmax(2)                               # (M, J)
            conf_kp = hm_kp.gather(2, m_kp.unsqueeze(-1)).squeeze(-1)  # (M, J)
            conf_kp = (conf_kp.clamp(max=255.) / 255.).cpu().numpy()    # (M, J)

            kp_x = (m_kp % kp_out[1].shape[3]).float() * 2        # (M, J) in crop coords
            kp_y = (m_kp // kp_out[1].shape[3]).float() * 2       # (M, J) in crop coords
            # Map back to full image coords
            kp_x = kp_x + (cx[det_idx] - bbox_hw).float().unsqueeze(1)
            kp_y = kp_y + (cy[det_idx] - bbox_hw).float().unsqueeze(1)
            kp_x = kp_x.cpu().numpy()   # (M, J)
            kp_y = kp_y.cpu().numpy()   # (M, J)

        # ---- Triangulate each keypoint ----
        row = []
        for j in range(num_joints):
            pts2d  = np.stack([kp_x[:, j], kp_y[:, j]], axis=1)   # (M, 2)
            confs  = conf_kp[:, j]                                  # (M,)
            Ps_det = Ps_np[det_idx]                                 # (M, 3, 4)
            pt3d   = triangulate_dlt(pts2d, Ps_det, confs,
                                     min_conf=kp_conf_threshold,
                                     max_reproj_err=max_reproj_err)
            if pt3d is not None:
                avg_conf = float(confs.mean())
                row += [float(pt3d[0]), float(pt3d[1]), float(pt3d[2]), avg_conf]
            else:
                row += ['NaN', 'NaN', 'NaN', 'NaN']

        writer.writerow(row)

    for cap in caps:
        cap.release()
    csvfile.close()
    return output_dir


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project',        default='mouseJan30')
    p.add_argument('--recording-path', default='/mnt/mouse2/rory/2025_12_23_16_57_09')
    p.add_argument('--calib-dir',      default='/home/user/src/analyzeMiceTrajectory/calib_params/2025_12_22')
    p.add_argument('--frame-start',    type=int, default=15186)
    p.add_argument('--num-frames',     type=int, default=100)
    p.add_argument('--output-dir',     default=None)
    p.add_argument('--tag',            default='', help='adds timing.txt to output dir')
    p.add_argument('--center-threshold', type=float, default=40.0)
    p.add_argument('--max-reproj-err',  type=float, default=200.0,
                   help='per-camera reprojection error threshold for outlier rejection (px)')
    p.add_argument('--trt', action='store_true',
                   help='use TRT-compiled models (compile first with compile_trt_hybrid24.py)')
    args = p.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(
            '/home/user/src/JARVIS-HybridNet/projects', args.project,
            'predictions/predictions3D',
            f'benchmark_{args.tag}_frames{args.frame_start}-{args.frame_start+args.num_frames-1}'
        )

    print(f"[2D+tri] project={args.project} frames={args.num_frames} -> {out_dir}")
    t0 = time.perf_counter()
    predict2d_triangulate(
        args.project, args.recording_path, args.calib_dir,
        args.frame_start, args.num_frames, out_dir,
        center_threshold=args.center_threshold,
        max_reproj_err=args.max_reproj_err,
        use_trt=args.trt,
    )
    elapsed = time.perf_counter() - t0
    fps = args.num_frames / elapsed
    print(f"[2D+tri] DONE  elapsed={elapsed:.1f}s  fps={fps:.2f}  ({elapsed/args.num_frames*1000:.0f} ms/frame)")
    if args.tag:
        with open(os.path.join(out_dir, 'timing.txt'), 'w') as f:
            f.write(f"project: {args.project}\nmethod: 2D+triangulation\n"
                    f"frames: {args.num_frames}\nelapsed_s: {elapsed:.2f}\n"
                    f"fps: {fps:.2f}\nms_per_frame: {elapsed/args.num_frames*1000:.1f}\n")

if __name__ == '__main__':
    main()
