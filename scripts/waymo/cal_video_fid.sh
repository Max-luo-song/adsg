#!/bin/bash

# Check if scene ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <scene_id>"
    echo "Example: $0 23"
    exit 1
fi

INPUT_ID="$1"

# Handle Scene ID formatting
# 1. Remove leading zeros for Output directory (e.g., 023 -> 23)
#    (Using sed to remove leading zeros. If result is empty, default to 0)
SCENE_ID_CLEAN=$(echo "$INPUT_ID" | sed 's/^0*//')
if [ -z "$SCENE_ID_CLEAN" ]; then
    SCENE_ID_CLEAN="0"
fi

# 2. Pad to 3 digits for GT directory (e.g., 23 -> 023)
#    (Using printf)
SCENE_ID_PADDED=$(printf "%03d" "$SCENE_ID_CLEAN")

echo "Input ID: $INPUT_ID"
echo "Clean ID (for Output): $SCENE_ID_CLEAN"
echo "Padded ID (for GT):    $SCENE_ID_PADDED"

# Define base paths
BASE_OUTPUT_DIR="/inspire/hdd/global_user/guoluosong-253108120129/project/adstudio/output"
# Construct Scene Directory using CLEAN ID (unpadded)
SCENE_DIR="${BASE_OUTPUT_DIR}/waymo_${SCENE_ID_CLEAN}/20260317_lidar+cam0_1_2adstudiov1"
NOVEL_TRAJ_DIR="${SCENE_DIR}/novel_traj"

# GT Folder path using PADDED ID (3 digits)
GT_BASE_DIR="/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed/training"
GT_FOLDER="${GT_BASE_DIR}/${SCENE_ID_PADDED}/images"

# Validate directories
if [ ! -d "$SCENE_DIR" ]; then
    echo "Error: Scene directory not found: $SCENE_DIR"
    echo "Checked path: $SCENE_DIR"
    echo "Note: Expected format is 'origin_<unpadded_id>'"
    exit 1
fi

if [ ! -d "$NOVEL_TRAJ_DIR" ]; then
    echo "Error: Novel trajectory directory not found: $NOVEL_TRAJ_DIR"
    exit 1
fi

if [ ! -d "$GT_FOLDER" ]; then
    echo "Error: GT images directory not found: $GT_FOLDER"
    echo "Checked path: $GT_FOLDER"
    echo "Note: Expected format is '<padded_3_digit_id>/images'"
    exit 1
fi

echo "Processing Scene: $SCENE_ID_CLEAN (Padded: $SCENE_ID_PADDED)"
echo "Searching in: $NOVEL_TRAJ_DIR"
echo "Using GT: $GT_FOLDER"

# Determine script directory to find python script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="/inspire/hdd/global_user/guoluosong-253108120129/project/adstudio/adsutido_tools/cal_video_fid.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Iterate through each trajectory folder in novel_traj
# Enable nullglob so loop doesn't run if no matches
shopt -s nullglob
FOUND_ANY=false

for TRAJ_PATH in "$NOVEL_TRAJ_DIR"/*; do
    if [ -d "$TRAJ_PATH" ]; then
        FOUND_ANY=true
        TRAJ_NAME=$(basename "$TRAJ_PATH")
        VIDEOS_DIR="${TRAJ_PATH}/videos"
        
        # Check if videos directory exists
        if [ ! -d "$VIDEOS_DIR" ]; then
             # echo "Skipping $TRAJ_NAME (No 'videos' subdirectory)"
             continue
        fi

        echo "------------------------------------------------"
        echo "Processing trajectory: $TRAJ_NAME"
        
        # Find the 3 video files
        # Logic: Left(1), Middle(0), Right(2)
        VID0=$(find "$VIDEOS_DIR" -maxdepth 1 -name "*cam0*.mp4" | head -n 1)
        VID1=$(find "$VIDEOS_DIR" -maxdepth 1 -name "*cam1*.mp4" | head -n 1)
        VID2=$(find "$VIDEOS_DIR" -maxdepth 1 -name "*cam2*.mp4" | head -n 1)
        
        if [ -n "$VID0" ] && [ -n "$VID1" ] && [ -n "$VID2" ]; then
            COMBINED_VIDEO="${VIDEOS_DIR}/combined_layout.mp4"
            OUTPUT_TXT="${TRAJ_PATH}/fid_score.txt"
            
            echo "Stitching videos..."
            # Stitch: Left(1) | Middle(0) | Right(2)
            # Use -y to overwrite output
            # Use -v error to suppress noise, -stats to show progress
            ffmpeg -y -v error -stats \
                -i "$VID1" -i "$VID0" -i "$VID2" \
                -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
                -map "[v]" \
                "$COMBINED_VIDEO"
            
    if [ $? -eq 0 ]; then
                echo "Stitching successful: $COMBINED_VIDEO"
                
                # FID Calculation - Use cam0 video (single view) instead of combined
                if [ -n "$VID0" ] && [ -f "$VID0" ]; then
                    echo "Calculating FID for $(basename "$VID0")..."
                    
                    # Ensure we use the correct environment where cleanfid is installed
                    # Assuming drivestudio env which was used in the pipeline
                    source /opt/conda/etc/profile.d/conda.sh
                    conda activate drivestudio
                    
                    # Execute python script
                    # Redirect output to fid_score.txt in the trajectory folder
                    python "$PYTHON_SCRIPT" \
                        --video "$VID0" \
                        --gt_folder "$GT_FOLDER" > "$OUTPUT_TXT" 2>&1
                        
                    echo "FID calculation done. Result saved to: $OUTPUT_TXT"
                else
                    echo "Error: cam0 video not found for FID calculation."
                fi
            else
                echo "Error: ffmpeg stitching failed."
            fi
        else
            echo "Warning: Incomplete video set in $VIDEOS_DIR"
            echo "Needed: *cam0*.mp4, *cam1*.mp4, *cam2*.mp4"
            echo "Found: 0=${VID0} 1=${VID1} 2=${VID2}"
        fi
    fi
done

if [ "$FOUND_ANY" = false ]; then
    echo "Warning: No trajectory folders found in $NOVEL_TRAJ_DIR"
fi

echo "Done."
