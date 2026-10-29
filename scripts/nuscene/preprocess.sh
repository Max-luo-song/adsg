#!/bin/bash
###############################################################################
# NuScenes Preprocessing Script for AdStudio
# Usage: ./scripts/nuscene/preprocess.sh [start_idx] [num_scenes] [input_dir] [output_dir] [split] [interpolate_N] [workers] [gpu_id]
# Example: ./scripts/nuscene/preprocess.sh 0 10 \
#   /inspire/hdd/global_user/guoluosong-253108120129/project/data/sets/nuscenes \
#   /inspire/hdd/global_user/guoluosong-253108120129/project/data/nuscenes/processedads \
#   v1.0-mini 4 32 0
###################################################################

set -euo pipefail

# Ensure script execution directory context
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")" # This is ADSTUDIO_ROOT
ADSTUDIO_ROOT="$PROJECT_ROOT"
DRIVESTUDIO_ROOT="${DRIVESTUDIO_ROOT:-/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio}"

# Parse arguments
start_idx="${1:-0}"
num_scenes="${2:-10}"
input_dir="${3:-/inspire/hdd/global_user/guoluosong-253108120129/project/data/sets/nuscenes}"
output_dir="${4:-/inspire/hdd/global_user/guoluosong-253108120129/project/data/nuscenes/processedads}"
split="${5:-v1.0-mini}"
interpolate_N="${6:-4}"
workers="${7:-32}"
gpu="${8:-0}"

# Normalize paths
input_dir="${input_dir%/}"
output_dir="${output_dir%/}"
split_short="${split#v1.0-}"

# Validate inputs
if ! [[ "$start_idx" =~ ^[0-9]+$ && "$num_scenes" =~ ^[0-9]+$ && "$workers" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] start_idx / num_scenes / workers must be non-negative integers."
    exit 1
fi

if ! [[ "$interpolate_N" =~ ^[0-9]+$ ]] || [ "$interpolate_N" -gt 4 ]; then
    echo "[ERROR] interpolate_N must be an integer in [0, 4]."
    exit 1
fi

if [ ! -d "$input_dir" ]; then
    echo "[ERROR] Input dir not found: $input_dir"
    exit 1
fi

if [ ! -d "$input_dir/$split" ]; then
    echo "[ERROR] Split metadata dir not found: $input_dir/$split"
    exit 1
fi

if [ ! -d "$input_dir/samples" ] || [ ! -d "$input_dir/sweeps" ]; then
    echo "[ERROR] Invalid NuScenes raw layout under: $input_dir"
    echo "[ERROR] Expected at least: $input_dir/{samples,sweeps,$split}"
    exit 1
fi

# Calculate output paths
if [ "$interpolate_N" -gt 0 ]; then
    hz=$(( (interpolate_N + 1) * 2 ))
    converted_root="${output_dir/processed/processed_${hz}Hz}"
else
    converted_root="$output_dir"
fi
scene_root="${converted_root}/${split_short}"

echo "[INFO] AdStudio Root    : $ADSTUDIO_ROOT"
echo "[INFO] DriveStudio Root : $DRIVESTUDIO_ROOT"
echo "[INFO] Input Dir        : $input_dir"
echo "[INFO] Output Dir       : $output_dir"
echo "[INFO] Split            : $split (short=${split_short})"
echo "[INFO] Start Idx        : $start_idx"
echo "[INFO] Num Scenes       : $num_scenes"
echo "[INFO] Interpolate N    : $interpolate_N"
echo "[INFO] Workers          : $workers"
echo "[INFO] GPU              : $gpu"
echo "[INFO] Processed Root   : $scene_root"

cd "$ADSTUDIO_ROOT" || { echo "[ERROR] Cannot cd to $ADSTUDIO_ROOT"; exit 1; }
export PYTHONPATH="$ADSTUDIO_ROOT:${PYTHONPATH:-}"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ============================================================================
# STAGE 1: Raw Data Conversion
# ============================================================================
echo "[INFO] ========== STAGE 1: Convert NuScenes =========="
conda activate drivestudio

python datasets/preprocess.py \
    --data_root "$input_dir" \
    --target_dir "$output_dir" \
    --dataset nuscenes \
    --split "$split" \
    --start_idx "$start_idx" \
    --num_scenes "$num_scenes" \
    --interpolate_N "$interpolate_N" \
    --workers "$workers" \
    --process_keys images lidar calib dynamic_masks objects

if [ $? -ne 0 ]; then
    echo "[ERROR] Data conversion failed"
    exit 1
fi

echo "[SUCCESS] Stage 1 completed."

# ============================================================================
# STAGE 2: Mask Extraction (SegFormer)
# ============================================================================
echo "[INFO] ========== STAGE 2: Extract Masks =========="
conda activate segformer
export PYTHONPATH="$ADSTUDIO_ROOT:${PYTHONPATH:-}"

# Pick GPU using adstudio utils.sh
if [ -f "${ADSTUDIO_ROOT}/scripts/utils.sh" ]; then
    source "${ADSTUDIO_ROOT}/scripts/utils.sh"
    if [ "${gpu}" = "-1" ] || [ "${gpu}" = "auto" ]; then
        gpu=$(pick_gpu)
    fi
else
    echo "[WARN] scripts/utils.sh not found"
    if [ "${gpu}" = "-1" ] || [ "${gpu}" = "auto" ]; then
        gpu=0
    fi
fi

if [ -z "${gpu}" ]; then
    echo "[ERROR] No available GPU."
    exit 1
fi
echo "[INFO] Using GPU: ${gpu}"

segformer_path="${ADSTUDIO_ROOT}/third_party/SegFormer"
checkpoint_path="${SEGFORMER_CHECKPOINT:-$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth}"

if [ ! -f "$checkpoint_path" ]; then
    echo "[ERROR] SegFormer checkpoint not found: $checkpoint_path"
    echo "[ERROR] You can set SEGFORMER_CHECKPOINT env var."
    exit 1
fi

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks.py \
    --data_root "$scene_root" \
    --segformer_path "$segformer_path" \
    --checkpoint "$checkpoint_path" \
    --start_idx "$start_idx" \
    --num_scenes "$num_scenes" \
    --process_dynamic_mask

if [ $? -ne 0 ]; then
    echo "[ERROR] Mask extraction failed"
    exit 1
fi

echo "[SUCCESS] Stage 2 completed."

# ============================================================================
# STAGE 3: SMPL Pose Extraction (4D-Humans)
# ============================================================================
echo "[INFO] ========== STAGE 3: SMPL Pose Extraction =========="
conda activate 4D-humans
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export EGL_PLATFORM=surfaceless
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$ADSTUDIO_ROOT:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --dataset nuscenes \
    --data_root "$scene_root" \
    --start_idx "$start_idx" \
    --num_scenes "$num_scenes"

if [ $? -ne 0 ]; then
    echo "[ERROR] SMPL extraction failed"
    exit 1
fi

echo "[SUCCESS] Stage 3 completed."
echo "[SUCCESS] ========== NuScenes preprocessing completed =========="
