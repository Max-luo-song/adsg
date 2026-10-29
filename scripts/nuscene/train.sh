#!/bin/bash
###############################################################################
# NuScenes Training Script for AdStudio
# Usage: ./scripts/nuscene/train.sh <scene_name> [processed_dir] [start_timestep] [end_timestep] [gpu_id] [output_root] [dataset_config] [camera_ids_csv]
# Example: ./scripts/nuscene/train.sh scene-0061 /path/to/processed/mini 0 -1 0 output6cams nuscenes/6cams 0,1,2,3,4,5
###############################################################################

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate drivestudio

# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_name> [processed_dir] [start_timestep] [end_timestep] [gpu_id] [output_root] [dataset_config] [camera_ids_csv]"
    echo "ERROR: Example: $0 scene-0061 /path/to/processed/mini 0 -1 0 output6cams nuscenes/6cams 0,1,2,3,4,5"
    exit 1
fi

scene_name="$1"
if [[ "$scene_name" == scene-* ]]; then
    scene_digits="${scene_name#scene-}"
elif [[ "$scene_name" =~ ^[0-9]+$ ]]; then
    scene_digits="$scene_name"
else
    echo "[ERROR] Invalid scene_name: $scene_name (expected scene-XXXX or numeric index)"
    exit 1
fi

if ! [[ "$scene_digits" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid numeric scene index parsed from: $scene_name"
    exit 1
fi
scene_idx=$(printf "%03d" "$((10#$scene_digits))")

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    # Default to project/data/nuscenes/processedads
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
    processed_dir="${PARENT_ROOT}/data/nuscenes/processedads/mini"
fi

# Start and end timestep
start_timestep="${3:-0}"
end_timestep="${4:--1}"
gpu="${5:--1}"
output_root="${6:-output6cams}"
dataset_config="${7:-nuscenes/6cams}"
camera_ids_csv="${8:-0,1,2,3,4,5}"

IFS=',' read -r -a camera_ids <<< "$camera_ids_csv"
if [ "${#camera_ids[@]}" -eq 0 ]; then
    echo "[ERROR] camera_ids_csv is empty: $camera_ids_csv"
    exit 1
fi

# Normalize path
processed_dir="${processed_dir%/}"

echo "[INFO] Scene Name: $scene_name"
echo "[INFO] Scene Index: $scene_idx"
echo "[INFO] Processed Dir: $processed_dir"
echo "[INFO] Start Timestep: $start_timestep"
echo "[INFO] End Timestep: $end_timestep"
echo "[INFO] GPU ID: $gpu"
echo "[INFO] Output Root: $output_root"
echo "[INFO] Dataset Config: $dataset_config"
echo "[INFO] Camera IDs: $camera_ids_csv"

# Parameters
################################################################################
lidar_type="lidar"
config_file="configs/omnire_extended_cam_lidar.yaml"
extra_config_info="adstudiov1_nuscene6cam_pbr"
################################################################################

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Pick GPU
if [ -f "${PROJECT_ROOT}/scripts/utils.sh" ]; then
    source "${PROJECT_ROOT}/scripts/utils.sh"
fi

if [ "${gpu}" = "-1" ]; then
    if command -v pick_gpu &> /dev/null; then
        gpu=$(pick_gpu)
    else
        echo "pick_gpu function not found, defaulting to GPU 0"
        gpu=0
    fi
fi

if [ -z "${gpu}" ]; then
    echo "No GPU found or selected"
    exit 1
fi
echo "Using GPU: ${gpu}"

project_name="nuscene"
run_name="$scene_name"

run_dir="${output_root}/${project_name}/${run_name}"

# Check output directory
if [ -d "$run_dir" ]; then
    echo "Warning: Directory $run_dir already exists!"
fi
mkdir -p "$run_dir"

# Backup script
script_name=$(basename "$0")
backup_path="${run_dir}/${script_name}"
cp "$0" "$backup_path"

# Start training
cd "${PROJECT_ROOT}" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py \
    --config_file $config_file \
    --output_root $output_root \
    --project $project_name \
    --run_name $run_name \
    dataset=$dataset_config \
    data.data_root=$processed_dir \
    data.scene_idx=$scene_idx \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
