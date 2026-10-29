#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_videos_directory>"
    exit 1
fi

INPUT_DIR="$1"

# Remove trailing slash if present
INPUT_DIR="${INPUT_DIR%/}"

# Check if directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Find the video files
# We look for patterns containing cam0, cam1, and cam2
# The user specified layout: Left=1, Middle=0, Right=2
VID0=$(find "$INPUT_DIR" -maxdepth 1 -name "*cam0*.mp4" | head -n 1)
VID1=$(find "$INPUT_DIR" -maxdepth 1 -name "*cam1*.mp4" | head -n 1)
VID2=$(find "$INPUT_DIR" -maxdepth 1 -name "*cam2*.mp4" | head -n 1)

# Check if all files were found
if [ -z "$VID0" ] || [ -z "$VID1" ] || [ -z "$VID2" ]; then
    echo "Error: Could not find all three video files (*cam0*.mp4, *cam1*.mp4, *cam2*.mp4) in $INPUT_DIR"
    echo "Found:"
    echo "  Middle (0): ${VID0:-Not Found}"
    echo "  Left   (1): ${VID1:-Not Found}"
    echo "  Right  (2): ${VID2:-Not Found}"
    exit 1
fi

# Define output filename
OUTPUT_FILE="${INPUT_DIR}/combined_layout.mp4"

echo "Stitching videos..."
echo "  Left:   $(basename "$VID1")"
echo "  Middle: $(basename "$VID0")"
echo "  Right:  $(basename "$VID2")"
echo "  Output: $OUTPUT_FILE"

# Run ffmpeg
# -y : Overwrite output files without asking
# -i : Input files
# -filter_complex hstack=inputs=3 : Stack 3 input videos horizontally
# The inputs are supplied in order 1, 0, 2, so hstack will arrange them 1|0|2
ffmpeg -y \
    -i "$VID1" \
    -i "$VID0" \
    -i "$VID2" \
    -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
    -map "[v]" \
    "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully created $OUTPUT_FILE"
else
    echo "Error: ffmpeg failed."
    exit 1
fi
