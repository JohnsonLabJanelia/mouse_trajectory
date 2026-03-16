"""
Speed benchmark: run JARVIS 3D prediction on a small frame window,
time it, and save results to a non-overlapping output dir.

Usage:
    python test_speed_benchmark.py --project mouseClimb4 --trt off --tag baseline
    python test_speed_benchmark.py --project mouseClimb4 --trt new  --tag trt
    python test_speed_benchmark.py --project mouseJan30  --trt off --tag jan30
"""
import argparse, os, sys, time
sys.path.insert(0, '/home/user/src/JARVIS-HybridNet')
from jarvis.utils.paramClasses import Predict3DParams
from jarvis.prediction.predict3D import predict3D

RECORDING_PATH = '/mnt/mouse2/rory/2025_12_23_16_57_09'
DATASET_NAME   = '/home/user/src/analyzeMiceTrajectory/calib_params/2025_12_22'
FRAME_START    = 15186
NUM_FRAMES     = 100   # small window for benchmarking

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='mouseClimb4')
    p.add_argument('--trt', choices=['off', 'new', 'previous'], default='off')
    p.add_argument('--tag', default='test')
    p.add_argument('--num-frames', type=int, default=NUM_FRAMES)
    p.add_argument('--frame-start', type=int, default=FRAME_START)
    args = p.parse_args()

    out_dir = os.path.join(
        '/home/user/src/JARVIS-HybridNet/projects', args.project,
        'predictions/predictions3D',
        f'benchmark_{args.tag}_frames{args.frame_start}-{args.frame_start+args.num_frames-1}'
    )
    os.makedirs(out_dir, exist_ok=True)

    params = Predict3DParams(
        project_name=args.project,
        recording_path=RECORDING_PATH,
        weights_center_detect='latest',
        weights_hybridnet='latest',
        frame_start=args.frame_start,
        number_frames=args.num_frames,
    )
    params.dataset_name = DATASET_NAME
    params.trt_mode = args.trt
    params.output_dir = out_dir

    print(f"[bench] project={args.project} trt={args.trt} frames={args.num_frames} -> {out_dir}")
    t0 = time.perf_counter()
    predict3D(params)
    elapsed = time.perf_counter() - t0

    fps = args.num_frames / elapsed
    print(f"[bench] DONE  elapsed={elapsed:.1f}s  fps={fps:.2f} frames/s  ({elapsed/args.num_frames*1000:.0f} ms/frame)")
    # Write timing summary
    with open(os.path.join(out_dir, 'timing.txt'), 'w') as f:
        f.write(f"project: {args.project}\ntrt: {args.trt}\nframes: {args.num_frames}\n"
                f"elapsed_s: {elapsed:.2f}\nfps: {fps:.2f}\nms_per_frame: {elapsed/args.num_frames*1000:.1f}\n")

if __name__ == '__main__':
    main()
