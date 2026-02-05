#!/usr/bin/env python3
"""
Extract a frame from a video.

Usage:
  # Extract frame 0 from a video (original behavior):
  python extract_first_frame.py <video.mp4> [output.png]
  # Extract the trial start frame from a prediction folder (e.g. frame 11572 for
  # Predictions_3D_trial_0000_11572-20491):
  python extract_first_frame.py <prediction_folder> --camera Cam2005325 [output.png]
"""

from pathlib import Path
import argparse
import re
import subprocess
import yaml


def parse_trial_frame_and_recording(prediction_dir: Path) -> tuple[int, Path | None]:
    """
    Parse frame_start from the prediction folder name (e.g. ..._11572-20491 -> 11572)
    and read recording_path from info.yaml if present.
    """
    # Extract frame_start from folder name pattern: ..._11572-20491
    match = re.search(r'_(\d+)-(\d+)$', prediction_dir.name)
    if not match:
        raise ValueError(f"Could not parse frame_start from folder name: {prediction_dir.name}")
    frame_start = int(match.group(1))
    
    # Try to read recording_path from info.yaml
    info_path = prediction_dir / "info.yaml"
    recording_path = None
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = yaml.safe_load(f)
                if "recording_path" in info:
                    recording_path = Path(info["recording_path"])
        except Exception:
            pass
    
    return frame_start, recording_path


def extract_frame(video_path: Path, frame_index: int, out_path: Path) -> None:
    """Extract a single frame from a video with ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_index})",
        "-vframes", "1",
        "-y",  # Overwrite output file
        str(out_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Extract a frame from a video")
    parser.add_argument(
        "input",
        type=Path,
        help="Video file path, or prediction folder path",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output PNG path (default: frame.png in same directory)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera name (e.g. Cam2005325) when input is a prediction folder",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to extract (default: 0, or frame_start from folder name)",
    )
    args = parser.parse_args()
    
    input_path = args.input.resolve()
    
    # Determine if input is a prediction folder or video file
    if input_path.is_dir():
        # Prediction folder mode
        if not args.camera:
            raise ValueError("--camera required when input is a prediction folder")
        
        frame_start, recording_path = parse_trial_frame_and_recording(input_path)
        frame_index = frame_start if args.frame_index == 0 else args.frame_index
        
        if not recording_path:
            raise ValueError(f"Could not find recording_path in {input_path / 'info.yaml'}")
        
        # Find video file for this camera
        video_path = recording_path / f"{args.camera}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = input_path / f"{args.camera}_frame{frame_index}.png"
    else:
        # Direct video mode
        video_path = input_path
        frame_index = args.frame_index
        
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = video_path.parent / f"{video_path.stem}_frame{frame_index}.png"
    
    extract_frame(video_path, frame_index, out_path)
    print(f"Extracted frame {frame_index} from {video_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
