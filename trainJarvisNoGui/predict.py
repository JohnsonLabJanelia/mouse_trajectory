import argparse
import os
import sys
from jarvis.utils.paramClasses import Predict3DParams
from jarvis.prediction.predict3D import predict3D

'''
python scripts/predict.py --project jakob --recording-path /groups/johnson/johnsonlab/rob_share/voigts/2025_10_07/ --dataset-name /groups/johnson/johnsonlab/rob_share/voigts/2025_10_07/jarvis_calib/
'''
def main():
    p = argparse.ArgumentParser("Run 3D prediction")
    p.add_argument("--project", default="none")
    p.add_argument("--recording-path",
                   default='/groups/johnson/johnsonlab/jinyao_share/2025_04_04_14_26_47'
)
    p.add_argument("--dataset-name", default='/groups/johnson/johnsonlab/jinyao_share/2025_04_04_14_26_47/jarvis_rat24/calib_params/2025_08_06_22_36_35/', 
                   help="calib folder path")
    p.add_argument("--weights-center", default="latest")
    p.add_argument("--weights-hybridnet", default="latest")
    p.add_argument("--frame-start", type=int, default=3800)
    p.add_argument("--number-frames", type=int, default=-1, help="-1 for all available frames")
    p.add_argument("--trt", choices=["off", "new", "previous"], default="off")
    args = p.parse_args()

    if not os.path.exists(args.recording_path):
        print(f"[error] Recording path not found: {args.recording_path}", file=sys.stderr)
        sys.exit(2)

    params = Predict3DParams(
        project_name=args.project,
        recording_path=args.recording_path,
        weights_center_detect=args.weights_center,
        weights_hybridnet=args.weights_hybridnet,
        frame_start=args.frame_start,
        number_frames=-1 if args.number_frames < 0 else args.number_frames
    )
    if args.dataset_name:
        params.dataset_name = args.dataset_name
    params.trt_mode = args.trt

    predict3D(params)

if __name__ == "__main__":
    main()

