import os
import argparse
from ruamel.yaml import YAML
from jarvis.visualization.create_videos3D import create_videos3D
from jarvis.utils.paramClasses import CreateVideos3DParams
from os.path import dirname, abspath


'''
python scripts/createvideos.py --project fin5 --prediction-dir Predictions_3D_20250929-172602

python scripts/createvideos.py --project jakob --prediction-dir Predictions_3D_20251007-164836 --cams Cam710031,Cam710037
'''

def parse_cams(s):
    if s is None or s.strip() == "":
        return None
    # return camera names as strings (e.g., "Cam700031,Cam700032")
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser("Create annotated 3D videos from a prediction run")
    p.add_argument("--prediction-dir", default=None)
    p.add_argument("--project", default="fin2")
    p.add_argument("--cams", default="Cam700031,Cam710038")
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--number-frames", type=int, default=None)
    args = p.parse_args()

    jarvis_dir = dirname(dirname(abspath(__file__)))
    projects_dir = os.path.join(jarvis_dir, "projects", args.project)
    full_prediction_dir = os.path.join(projects_dir, "predictions", "predictions3D", args.prediction_dir)

    if not os.path.isdir(full_prediction_dir):
        # Try under JARVIS-HybridNet (where predict.py saves when PYTHONPATH points there)
        jarvis_alt = os.path.join(jarvis_dir, "JARVIS-HybridNet")
        projects_dir_alt = os.path.join(jarvis_alt, "projects", args.project)
        full_prediction_dir_alt = os.path.join(projects_dir_alt, "predictions", "predictions3D", args.prediction_dir)
        if os.path.isdir(full_prediction_dir_alt):
            full_prediction_dir = full_prediction_dir_alt
            projects_dir = projects_dir_alt
        else:
            raise SystemExit(f"Prediction folder not found: {full_prediction_dir}")

    yaml = YAML()
    info_path = os.path.join(full_prediction_dir, "info.yaml")
    info = {}
    if os.path.isfile(info_path):
        with open(info_path, "r") as f:
            info = yaml.load(f)

    project = args.project or info.get("project_name") or info.get("project")
    if not project:
        raise SystemExit("Project name not found. Pass --project.")

 
    csvs = sorted([fn for fn in os.listdir(full_prediction_dir) if fn.lower().endswith(".csv")])
    if not csvs:
        raise SystemExit("No CSV found in prediction-dir. Pass --csv.")
    # use absolute path
    data_csv = os.path.join(full_prediction_dir, csvs[0])


    recording_path = info.get("recording_path")
    if not recording_path:
        raise SystemExit("Recording path not found in info.yaml; run predict3D first.")

    params = CreateVideos3DParams(
        project_name=project,
        recording_path=recording_path,
        data_csv=data_csv
    )
    params.dataset_name = info.get("dataset_name")
    params.video_cam_list = parse_cams(args.cams) if args.cams else info.get("cameras")
    params.frame_start = args.frame_start if args.frame_start is not None else info.get("frame_start", 0)
    params.number_frames = args.number_frames if args.number_frames is not None else info.get("number_frames", -1)
    params.skeleton_color = (0, 255, 255)  # Yellow in BGR for OpenCV

    create_videos3D(params)

if __name__ == "__main__":
    main()

