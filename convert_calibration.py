#!/usr/bin/env python3
"""
Convert calibration files from OpenCV format to JARVIS format.

OpenCV format expects:
- camera_matrix
- distortion_coefficients
- rc_ext (rotation matrix)
- tc_ext (translation vector)

JARVIS format expects:
- intrinsicMatrix
- distortionCoefficients
- R (rotation matrix)
- T (translation vector)

Usage:
    python convert_calibration.py --input-folder /path/to/opencv/calibration --output-folder /path/to/jarvis/calibration
"""

import os
import argparse
import glob
import cv2


def convert2jarviscalib(input_folder, output_folder):
    """
    Convert calibration files from OpenCV format to JARVIS format.
    
    Args:
        input_folder: Path to folder containing OpenCV format YAML files
        output_folder: Path to folder where JARVIS format YAML files will be saved
    """
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all YAML files in input folder
    yaml_files = glob.glob(os.path.join(input_folder, "*.yaml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in input folder: {input_folder}")
    
    # Extract camera names from filenames
    cam_names = []
    for file_path in yaml_files:
        filename = os.path.basename(file_path)
        cam_name = os.path.splitext(filename)[0]
        cam_names.append(cam_name)
    
    cam_names.sort()
    print(f"Found {len(cam_names)} calibration files to convert:")
    
    converted_count = 0
    failed_count = 0
    
    for cam_name in cam_names:
        input_file = os.path.join(input_folder, f"{cam_name}.yaml")
        output_file = os.path.join(output_folder, f"{cam_name}.yaml")
        
        try:
            # Read OpenCV format calibration file
            fs = cv2.FileStorage(input_file, cv2.FILE_STORAGE_READ)
            
            if fs.isOpened():
                # Read calibration parameters
                camera_matrix_node = fs.getNode("camera_matrix")
                distortion_node = fs.getNode("distortion_coefficients")
                rc_ext_node = fs.getNode("rc_ext")
                tc_ext_node = fs.getNode("tc_ext")
                
                if camera_matrix_node.empty() or distortion_node.empty() or rc_ext_node.empty() or tc_ext_node.empty():
                    print(f"  ⚠️  Skipping {cam_name}.yaml: Missing required calibration parameters")
                    failed_count += 1
                    fs.release()
                    continue
                
                intrinsicMatrix = camera_matrix_node.mat().T
                distortionCoefficients = distortion_node.mat().T
                R = rc_ext_node.mat().T
                T = tc_ext_node.mat()
                
                fs.release()
                
                # Write JARVIS format calibration file
                s = cv2.FileStorage(output_file, cv2.FileStorage_WRITE)
                s.write("intrinsicMatrix", intrinsicMatrix)
                s.write("distortionCoefficients", distortionCoefficients)
                s.write("R", R)
                s.write("T", T)
                s.release()
                
                print(f"  ✓ Converted {cam_name}.yaml")
                converted_count += 1
            else:
                print(f"  ✗ Failed to open {cam_name}.yaml")
                failed_count += 1
                
        except Exception as e:
            print(f"  ✗ Error converting {cam_name}.yaml: {str(e)}")
            failed_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Failed: {failed_count}")
    print(f"\nOutput saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert calibration files from OpenCV format to JARVIS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_calibration.py --input-folder /path/to/opencv/calibration --output-folder /path/to/jarvis/calibration
  
  python convert_calibration.py -i /run/user/1000/gvfs/smb-share:server=prfs,share=johnsonlab/quanshare/hurdles/calibration \\
                                 -o /tmp/jarvis_calibration
        """
    )
    parser.add_argument(
        "--input-folder", "-i",
        required=True,
        help="Path to folder containing OpenCV format YAML calibration files"
    )
    parser.add_argument(
        "--output-folder", "-o",
        required=True,
        help="Path to folder where JARVIS format YAML calibration files will be saved"
    )
    
    args = parser.parse_args()
    
    try:
        convert2jarviscalib(args.input_folder, args.output_folder)
    except Exception as e:
        print(f"Error: {str(e)}", file=__import__('sys').stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
