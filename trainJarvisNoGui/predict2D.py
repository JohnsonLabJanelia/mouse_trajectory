import argparse
import os
import sys
import csv
import itertools
import numpy as np
import torch
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from ruamel.yaml import YAML

from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis2D import JarvisPredictor2D

def create_video_reader(recording_path, video_paths):
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

def get_video_paths(recording_path, camera_names):
    videos = os.listdir(recording_path)
    video_paths = []
    for i, camera in enumerate(camera_names):
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
        return False
    imgs[slice] = img.astype(np.uint8)
    return True

def create_header(writer, cfg, num_cameras):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in cfg.KEYPOINT_NAMES))
    coords = ["x", "y", "confidence"] * len(cfg.KEYPOINT_NAMES)
    joints.insert(0, "frame")
    coords.insert(0, "frame")
    joints.insert(0, "camera")
    coords.insert(0, "camera")
    writer.writerow(joints)
    writer.writerow(coords)

def create_info_file(output_dir, recording_path, project_name):
    with open(os.path.join(output_dir, "info.yaml"), "w") as file:
        yaml = YAML()
        yaml.dump(
            {
                "recording_path": recording_path,
                "project_name": project_name,
            },
            file,
        )

def predict2D(project_name, recording_path, weights_center="latest", weights_keypoint="latest", 
              frame_start=0, number_frames=-1, trt_mode="off", cameras=None):
    """
    Run 2D keypoint prediction on video files.
    
    Args:
        project_name: Name of the JARVIS project
        recording_path: Path to folder containing video files
        weights_center: CenterDetect weights to use (default: "latest")
        weights_keypoint: KeypointDetect weights to use (default: "latest")
        frame_start: Starting frame number (default: 0)
        number_frames: Number of frames to process (-1 for all, default: -1)
        trt_mode: TensorRT mode ("off", "new", "previous", default: "off")
        cameras: List of camera names to use (None for all, default: None)
    """
    # Load project and config
    project = ProjectManager()
    if not project.load(project_name):
        print(f"Could not load project: {project_name}! Aborting....", file=sys.stderr)
        sys.exit(2)
    cfg = project.cfg

    # Initialize 2D predictor
    jarvisPredictor = JarvisPredictor2D(
        cfg,
        weights_center,
        weights_keypoint,
        trt_mode,
    )

    # Get camera names
    if cameras is None:
        # Try to get cameras from config or list videos
        videos = os.listdir(recording_path)
        cameras = [v.split(".")[0] for v in videos if v.endswith(('.mp4', '.avi', '.mov'))]
        cameras.sort()
    
    # Create output directory
    output_dir = os.path.join(
        recording_path,
        "predictions",
        "jarvis",
        project_name,
        f'Predictions_2D_{time.strftime("%Y%m%d-%H%M%S")}',
    )
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    create_info_file(output_dir, recording_path, project_name)

    # Get video paths
    video_paths = get_video_paths(recording_path, cameras)
    caps, img_size = create_video_reader(recording_path, video_paths)

    # Create CSV file
    csvfile = open(os.path.join(output_dir, "data2D.csv"), "w", newline="")
    writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header if keypoint names are available
    if len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS:
        create_header(writer, cfg, len(cameras))

    # Determine frame range
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    if number_frames < 0:
        frame_end = total_frames - 1
    else:
        frame_end = min(frame_start + number_frames - 1, total_frames - 1)

    imgs_orig = np.zeros((len(caps), img_size[1], img_size[0], 3)).astype(np.uint8)

    # Process frames
    for frame_num in tqdm(range(frame_start, frame_end + 1)):
        # Seek to frame in all streams
        Parallel(n_jobs=len(caps), require="sharedmem")(delayed(seek)(cap, frame_num) for cap in caps)

        # Read all cameras for this frame
        results = Parallel(n_jobs=len(caps), require="sharedmem")(
            delayed(read_images)(cap, slice, imgs_orig) for slice, cap in enumerate(caps)
        )
        
        if any(r is False for r in results):
            print(f"[warn] Read failure at frame {frame_num}. Writing NaN.")
            for cam_idx, camera in enumerate(cameras):
                row = [camera, frame_num] + (["NaN"] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 3))
                writer.writerow(row)
            continue

        # Process each camera
        imgs = torch.from_numpy(imgs_orig).cuda().float().permute(0, 3, 1, 2)[:, [2, 1, 0]] / 255.0

        for cam_idx, camera in enumerate(cameras):
            img = imgs[cam_idx:cam_idx+1]  # Single camera image
            
            # Run 2D prediction
            points2D, confidences = jarvisPredictor(img)

            if points2D is not None and confidences is not None:
                row = [camera, frame_num]
                for point, conf in zip(points2D.squeeze().cpu().numpy(), confidences.squeeze().cpu().numpy()):
                    row = row + [point[0], point[1], conf]
                writer.writerow(row)
            else:
                row = [camera, frame_num] + (["NaN"] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 3))
                writer.writerow(row)

    # Cleanup
    for cap in caps:
        cap.release()
    csvfile.close()
    
    print(f"2D predictions saved to: {output_dir}")
    return output_dir

def main():
    p = argparse.ArgumentParser("Run 2D keypoint prediction")
    p.add_argument("--project", default="none", required=True,
                   help="JARVIS project name")
    p.add_argument("--recording-path", required=True,
                   help="Path to folder containing camera videos")
    p.add_argument("--weights-center", default="latest",
                   help="CenterDetect weights (default: latest)")
    p.add_argument("--weights-keypoint", default="latest",
                   help="KeypointDetect weights (default: latest)")
    p.add_argument("--frame-start", type=int, default=0,
                   help="Starting frame number (default: 0)")
    p.add_argument("--number-frames", type=int, default=-1,
                   help="Number of frames to process (-1 for all, default: -1)")
    p.add_argument("--trt", choices=["off", "new", "previous"], default="off",
                   help="TensorRT mode (default: off)")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="List of camera names to use (default: all cameras in recording path)")
    
    args = p.parse_args()

    if not os.path.exists(args.recording_path):
        print(f"[error] Recording path not found: {args.recording_path}", file=sys.stderr)
        sys.exit(2)

    predict2D(
        args.project,
        args.recording_path,
        args.weights_center,
        args.weights_keypoint,
        args.frame_start,
        args.number_frames,
        args.trt,
        args.cameras,
    )

if __name__ == "__main__":
    main()



