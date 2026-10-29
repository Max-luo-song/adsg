#!/bin/bash
###############################################################################
# KITTI Training Script
# Usage: ./scripts/kitti/train.sh <scene_id> [processed_dir]
###############################################################################
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate drivestudio
# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_id> [processed_dir]"
    echo "ERROR: Example: $0 2011_09_26_drive_0001_sync /path/to/processed"
    exit 1
fi

scene_idx="$1"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    # Default to project/data/kitti/processed
    # Get project root from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
    processed_dir="${PARENT_ROOT}/data/kitti/processedads"
fi

# Normalize path
processed_dir="${processed_dir%/}"

echo "[INFO] Scene ID: $scene_idx"
echo "[INFO] Processed Dir: $processed_dir"

# Parameters
################################################################################
gpu=-1

camera_ids=(0 1)  # KITTI uses left (0) and right (1) cameras
lidar_type="lidar"
#/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio/configs/omnire.yaml
config_file="configs/omnire_extended_cam_lidar_kitti.yaml"
#config_file="/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio/configs/omnire.yaml"
dataset_config="kitti/2cams"
extra_config_info="adstudiov1_401_usedrive "

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
################################################################################

# Pick GPU
# Check if scripts/utils.sh exists relative to current dir or script dir
if [ -f "scripts/utils.sh" ]; then
    source scripts/utils.sh
elif [ -f "${PROJECT_ROOT}/scripts/utils.sh" ]; then
    source "${PROJECT_ROOT}/scripts/utils.sh"
else
    echo "Warning: scripts/utils.sh not found. GPU selection might fail if not specified."
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


output_root="output"
project_name="kitti_${scene_idx}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"

run_dir="${output_root}/${project_name}/${run_name}"

# Check output directory
if [ -d "$run_dir" ]; then
    echo "Warning: Directory $run_dir already exists!"
    # rm -r "$run_dir"
fi
mkdir -p "$run_dir"

# Backup script
script_name=$(basename "$0")
backup_path="${run_dir}/${script_name}"
cp "$0" "$backup_path"

# Start training
# Ensure we are in project root for python execution
if [ -d "${PROJECT_ROOT}" ]; then
    cd "${PROJECT_ROOT}"
fi

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
