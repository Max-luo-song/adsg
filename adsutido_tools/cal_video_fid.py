import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
import tempfile
import shutil
import subprocess
import sys
import time

# Add potential paths to cleanfid if not installed in site-packages
# Try to find drivestudio root and add third_party/clean-fid
current_dir = os.path.dirname(os.path.abspath(__file__))
# Structure: project/adstudio/adsutido_tools/cal_video_fid.py
# We want to reach: project/drivestudio/third_party/clean-fid
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
cleanfid_path = os.path.join(project_root, "drivestudio", "third_party", "clean-fid")

if os.path.exists(cleanfid_path):
    sys.path.append(cleanfid_path)
    # print(f"Added {cleanfid_path} to sys.path")

try:
    from cleanfid import fid
except ImportError:
    # Fallback: try relative path if running from drivestudio root
    sys.path.append("third_party/clean-fid")
    try:
        from cleanfid import fid
    except ImportError:
        print("Error: Could not import cleanfid. Please ensure it is installed or in PYTHONPATH.")
        print(f"Checked path: {cleanfid_path}")
        raise

def extract_frames_from_video(video_path, output_dir):
    """Extract all frames from video to a temporary directory using ffmpeg (faster)"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ffmpeg for faster extraction
    # -i: input
    # -vf: video filter (fps passthrough)
    # %06d.png: output pattern
    # -v error: quiet output
    cmd = [
        "ffmpeg", 
        "-y", 
        "-i", video_path, 
        "-vsync", "0", 
        "-q:v", "2",  # High quality for jpg, ignored for png but good habit
        os.path.join(output_dir, "frame_%06d.png")
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames with ffmpeg: {e}")
        # Fallback to OpenCV if ffmpeg fails
        return extract_frames_from_video_cv2(video_path, output_dir)
    except FileNotFoundError:
        print("ffmpeg not found, falling back to cv2")
        return extract_frames_from_video_cv2(video_path, output_dir)
    
    # Count frames
    num_frames = len([name for name in os.listdir(output_dir) if name.endswith('.png')])
    print(f"Extracted {num_frames} frames from {video_path}")
    return num_frames

def extract_frames_from_video_cv2(video_path, output_dir):
    """Fallback OpenCV extraction"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        Image.fromarray(frame_rgb).save(frame_path)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} (cv2)")
    return frame_count

def calculate_video_fid(pred_video_path, gt_folder, device="cuda", batch_size=32, num_workers=4):
    """
    Calculate FID score between a generated video and ground truth images.
    Updated to match Drivestudio logic: Filters GT images to use only cam0 (_0.jpg)
    """
    # Create temporary directory for extracted frames
    with tempfile.TemporaryDirectory() as temp_dir:
        pred_folder = os.path.join(temp_dir, "pred_frames")
        os.makedirs(pred_folder, exist_ok=True)
        
        # Extract frames from video
        print(f"Extracting frames from video: {pred_video_path}")
        num_frames = extract_frames_from_video(pred_video_path, pred_folder)
        
        if num_frames == 0:
            print("Warning: No frames extracted from video.")
            return -1.0

        # Create temporary GT folder with only cam0 images
        temp_gt_folder = os.path.join(temp_dir, "gt_frames")
        os.makedirs(temp_gt_folder, exist_ok=True)
        
        print(f"[{time.strftime('%H:%M:%S')}] Filtering GT images (using only cam0)...")
        gt_count = 0
        
        if not os.path.exists(gt_folder):
             print(f"Error: GT folder not found: {gt_folder}")
             return -1.0
             
        for filename in os.listdir(gt_folder):
            # Waymo convention: cam0 is typically suffix _0.jpg (FRONT)
            if filename.endswith("_0.jpg") or filename.endswith("_0.png"):
                src_path = os.path.join(gt_folder, filename)
                dst_path = os.path.join(temp_gt_folder, filename)
                try:
                    # Use symlink to save space/time, fallback to copy if fails
                    os.symlink(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)
                gt_count += 1
                
        print(f"[{time.strftime('%H:%M:%S')}] Prepared {gt_count} GT images (cam0 only) in temp folder")
        
        if gt_count == 0:
             print(f"Warning: No cam0 images found in GT folder {gt_folder}. Looking for *_0.jpg")
             return -1.0

        print(f"Calculating FID between filtered GT and pred_folder")
        print(f"Using device: {device}, Batch size: {batch_size}, Workers: {num_workers}")
        
        try:
            # Use fid.compute_fid directly (no manual stats caching, matching Drivestudio)
            fid_score = fid.compute_fid(
                temp_gt_folder, # Use filtered GT
                pred_folder, 
                mode="clean", 
                device=torch.device(device),
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=True
            )
            
        except Exception as e:
            print(f"Error computing FID: {e}")
            import traceback
            traceback.print_exc()
            return -1.0
            
    return fid_score

def main():
    parser = argparse.ArgumentParser(description="Calculate FID score for generated trajectory video")
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to the generated video"
    )
    parser.add_argument(
        "--gt_folder", 
        type=str, 
        required=True,
        help="Path to the folder containing ground truth images"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8,
        help="Number of workers for data loading"
    )
    
    args = parser.parse_args()
    
    # Calculate FID
    fid_score = calculate_video_fid(
        args.video, 
        args.gt_folder, 
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\n{'='*60}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"{'='*60}\n")
    
    return fid_score

if __name__ == "__main__":
    main()