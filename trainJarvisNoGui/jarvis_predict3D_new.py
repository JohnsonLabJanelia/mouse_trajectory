# jarvis_predict3D.py  (updated)
import os
import csv
import itertools
import numpy as np
import torch
import cv2
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from ruamel.yaml import YAML

from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D
from jarvis.utils.paramClasses import Predict3DParams

import pandas as pd
import argparse
import glob
import sys

# -------- reprojection helper (unchanged, but kept here) --------
def get_repro_tool(cfg, dataset_name, cameras_to_use=None, device="cuda"):
    reproTools = load_reprojection_tools(
        cfg, cameras_to_use=cameras_to_use, device=device
    )
    if dataset_name is not None and dataset_name not in reproTools:
        if os.path.isdir(dataset_name):
            dataset_dir = os.path.join(
                cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_3D
            )
            dataset_json = open(os.path.join(dataset_dir, "annotations", "instances_val.json"))
            data = json.load(dataset_json)
            calibPaths = {}
            calibParams = list(data["calibrations"].keys())[0]
            for cam in data["calibrations"][calibParams]:
                if cameras_to_use is None or cam in cameras_to_use:
                    calibPaths[cam] = data["calibrations"][calibParams][cam].split("/")[-1]
            if cfg["PROJECT_NAME"] in ["rat_pose", "rat24_2"]:
                # keep your remapping for those projects
                ordered_serial = [
                    "2002496","2002483","2002488","2002480","2002489","2002485","2002490",
                    "2002492","2002479","2002494","2002495","2002482","2002481","2002491",
                    "2002493","2002484","710038",
                ]
                calibPaths_new = {}
                for cam_name, _ in calibPaths.items():
                    cam_order = int(cam_name[3:])
                    cam_new_name = "Cam" + ordered_serial[cam_order]
                    calibPaths_new[cam_new_name] = cam_new_name + ".yaml"
                calibPaths = calibPaths_new
            reproTool = ReprojectionTool(dataset_name, calibPaths, device)
        else:
            print("Could not load reprojection Tool for specified project...")
            return None
    elif len(reproTools) == 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    elif len(reproTools) > 1:
        reproTool = reproTools[list(reproTools.keys())[0]] if dataset_name is None else reproTools[dataset_name]
    else:
        print("Could not load reprojection Tool for specified project...")
        return None
    return reproTool

# -------- core runner (kept) --------
def mypredict3d(params, trials_frames, output_root, cameras_to_use=None):
    # Load project and config
    project = ProjectManager()
    if not project.load(params.project_name):
        print(f"Could not load project: {params.project_name}! Aborting....")
        return
    cfg = project.cfg

    if cameras_to_use is not None:
        cfg.HYBRIDNET.NUM_CAMERAS = len(cameras_to_use)

    jarvisPredictor = JarvisPredictor3D(
        cfg,
        params.weights_center_detect,
        params.weights_hybridnet,
        params.trt_mode,
    )

    reproTool = get_repro_tool(cfg, params.dataset_name, cameras_to_use=cameras_to_use)

    params.output_dir = os.path.join(
        output_root,
        "predictions",
        "jarvis",
        params.project_name,
        f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}',
    )
    print(params.output_dir)
    os.makedirs(params.output_dir, exist_ok=True)
    create_info_file(params)

    # openCV readers
    video_paths = get_video_paths(params.recording_path, reproTool)
    caps, img_size = create_video_reader(params, reproTool, video_paths)

    csvfile = open(os.path.join(params.output_dir, "data3D.csv"), "w", newline="")
    writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # header if names are available
    if len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS:
        create_header(writer, cfg)

    imgs_orig = np.zeros((len(caps), img_size[1], img_size[0], 3)).astype(np.uint8)

    for trial_idx in tqdm(range(trials_frames.shape[0])):
        frame_start, frame_end = trials_frames[trial_idx]

        # seek to the start of the trial in all streams
        Parallel(n_jobs=17, require="sharedmem")(delayed(seek)(cap, frame_start) for cap in caps)

        trial_rows = []
        trial_failed = False

        for frame_num in range(frame_start, frame_end + 1):
            # read all cameras for this frame
            results = Parallel(n_jobs=17, require="sharedmem")(
                delayed(read_images)(cap, slice, imgs_orig) for slice, cap in enumerate(caps)
            )
            if any(r is False for r in results):
                print(f"[warn] Read failure in trial {trial_idx}, frame {frame_num}. Marking entire trial as NaN.")
                trial_failed = True
                break

            imgs = (
                torch.from_numpy(imgs_orig).cuda().float().permute(0, 3, 1, 2)[:, [2, 1, 0]] / 255.0
            )
            points3D_net, confidences = jarvisPredictor(
                imgs,
                reproTool.cameraMatrices.cuda(),
                reproTool.intrinsicMatrices.cuda(),
                reproTool.distortionCoefficients.cuda(),
            )

            if points3D_net is not None:
                row = [frame_num]
                for point, conf in zip(points3D_net.squeeze(), confidences.squeeze().cpu().numpy()):
                    row = row + point.tolist() + [conf]
                trial_rows.append(row)
            else:
                row = [frame_num] + (["NaN"] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 4))
                trial_rows.append(row)

        if trial_failed:
            # write NaNs for every frame in this trial to keep frame coverage
            for frame_num in range(frame_start, frame_end + 1):
                row = [frame_num] + (["NaN"] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 4))
                writer.writerow(row)
            print(f"[info] Trial {trial_idx} ({frame_start}-{frame_end}) written as NaN due to read error.")
        else:
            # trial succeeded: write buffered rows
            for row in trial_rows:
                writer.writerow(row)
            print(f"[info] Trial {trial_idx} ({frame_start}-{frame_end}) processed successfully.")

    for cap in caps:
        cap.release()
    csvfile.close()
    return params.output_dir

# -------- helpers (kept) --------
def create_video_reader(params, reproTool, video_paths):
    caps = []
    img_size = [0, 0]
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        img_size_new = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        assert img_size == [0, 0] or img_size == img_size_new, "All videos need to have the same resolution"
        img_size = img_size_new
        caps.append(cap)
    return caps, img_size

def get_video_paths(recording_path, reproTool):
    videos = os.listdir(recording_path)
    video_paths = []
    for i, camera in enumerate(reproTool.cameras):
        for video in videos:
            if camera == video.split(".")[0]:
                video_paths.append(os.path.join(recording_path, video))
        assert len(video_paths) == i + 1, ("Missing Recording for camera " + camera)
    return video_paths

def seek(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

def read_images(cap, slice, imgs):
    ret, img = cap.read()
    if not ret or img is None:
        # signal failure to caller
        return False
    imgs[slice] = img.astype(np.uint8)
    return True

def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 4) for x in cfg.KEYPOINT_NAMES))
    coords = ["x", "y", "z", "confidence"] * len(cfg.KEYPOINT_NAMES)
    joints.insert(0, "frame")
    coords.insert(0, "frame")
    writer.writerow(joints)
    writer.writerow(coords)

def create_info_file(params):
    with open(os.path.join(params.output_dir, "info.yaml"), "w") as file:
        yaml = YAML()
        yaml.dump(
            {
                "recording_path": params.recording_path,
                "dataset_name": params.dataset_name,
            },
            file,
        )

def convert2jarviscalib(input_folder, output_folder):
    cam_names = []
    for file in glob.glob(input_folder + "/*.yaml"):
        file_name = file.split("/")
        cam_names.append(file_name[-1].split(".")[0])
    cam_names.sort()

    for idx in range(len(cam_names)):
        input_file_name = input_folder + "/{}.yaml".format(cam_names[idx])
        print(input_file_name)
        fs = cv2.FileStorage(input_file_name, cv2.FILE_STORAGE_READ)
        intrinsicMatrix = fs.getNode("camera_matrix").mat().T
        distortionCoefficients = fs.getNode("distortion_coefficients").mat().T
        R = fs.getNode("rc_ext").mat().T
        T = fs.getNode("tc_ext").mat()

        output_filename = output_folder + "/{}.yaml".format(cam_names[idx])
        s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
        s.write("intrinsicMatrix", intrinsicMatrix)
        s.write("distortionCoefficients", distortionCoefficients)
        s.write("R", R)
        s.write("T", T)
        s.release()

# -------- NEW: CLI that mirrors predict.py and allows overrides --------
def _parse_trials(trials_csv, start_col, end_col):
    df = pd.read_csv(trials_csv)
    arr = df.loc[:, [start_col, end_col]].to_numpy()
    mask = ~np.isnan(arr).any(axis=1)
    return arr[mask].astype(int)
'''
python scripts/jarvis_predict3D_new.py \
  -i  /groups/johnson/johnsonlab/jinyao_share/2025_06_04_14_04_41/trial_wise_predict3d \
  --project fin5 \
  --recording-path /groups/johnson/johnsonlab/jinyao_share/2025_06_04_14_04_41 \
  --dataset-name /groups/johnson/johnsonlab/jinyao_share/2025_06_04_14_04_41/jarvis_rat24/calib_params/2025_08_28_14_55_31/ 
'''



def main():
    p = argparse.ArgumentParser("Run 3D prediction on trial windows (CSV)")
    p.add_argument("-i", "--input-folder", required=True,
                   help="Folder containing config.json and (by default) the trials CSV")
    p.add_argument("-m", "--mode", default="none", help="Kept for compatibility; not strictly required")

    # New: parity with predict.py
    p.add_argument("--project", default="none")
    p.add_argument("--recording-path", default=None,
                   help="Override: folder with camera videos (e.g., /path/to/videos)")
    p.add_argument("--dataset-name", default=None,
                   help="Override: JARVIS-style calib folder OR dataset key (skips conversion)")

    p.add_argument("--weights-center", default="latest")
    p.add_argument("--weights-hybridnet", default="latest")
    p.add_argument("--trt", choices=["off", "new", "previous"], default="off")

    # Trials CSV controls
    p.add_argument("--trials-csv", default=None,
                   help="CSV with trial start/end (default: INPUT_FOLDER/trial_sorted.csv)")
    p.add_argument("--trial-start-col", default="trial_start")
    p.add_argument("--trial-end-col", default="trial_end")

    # Optional camera subset & output location
    p.add_argument("--cameras", nargs="+", default=None,
                   help='Subset of cameras, e.g. --cameras Cam0 Cam4 Cam6')
    p.add_argument("--output-root", default=None,
                   help="Where to create predictions/... (default: INPUT_FOLDER)")

    args = p.parse_args()

    input_folder = args.input_folder
    output_root = args.output_root or input_folder

    # Load config for defaults if present
    config_path = os.path.join(input_folder, "config.json")
    config = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    # Resolve dataset/calibration folder
    dataset_name = args.dataset_name
    if dataset_name is None:
        if not config or "calibration_folder" not in config:
            print("[error] No --dataset-name provided and config.json lacks 'calibration_folder'.", file=sys.stderr)
            sys.exit(2)
        save_calib_folder = os.path.join(input_folder, "calibration_jarvis")
        if not os.path.isdir(save_calib_folder):
            os.makedirs(save_calib_folder, exist_ok=True)
            convert2jarviscalib(config["calibration_folder"], save_calib_folder)
        dataset_name = save_calib_folder

    # Resolve recording path
    recording_path = args.recording_path or (config.get("media_folder") if config else None)
    if not recording_path or not os.path.isdir(recording_path):
        print(f"[error] Recording path not found: {recording_path}", file=sys.stderr)
        sys.exit(2)

    # Trials CSV
    trials_csv = args.trials_csv or os.path.join(input_folder, "trial_sorted.csv")
    if not os.path.exists(trials_csv):
        print(f"[error] Trials CSV not found: {trials_csv}", file=sys.stderr)
        sys.exit(2)
    trials_frames = _parse_trials(trials_csv, args.trial_start_col, args.trial_end_col)

    # Build params (named args like predict.py)
    params = Predict3DParams(
        project_name=args.project,
        recording_path=recording_path,
        weights_center_detect=args.weights_center,
        weights_hybridnet=args.weights_hybridnet,
    )
    params.dataset_name = dataset_name
    params.trt_mode = args.trt

    # Run
    mypredict3d(params, trials_frames, output_root, cameras_to_use=args.cameras)

if __name__ == "__main__":
    main()

