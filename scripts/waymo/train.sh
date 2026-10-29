#!/bin/bash
###############################################################################
# 训练脚本
# 用法: ./train.sh <scene_id> [processed_dir] [gpu_id]
###############################################################################

# 参数处理
if [ $# -lt 1 ]; then
    echo "ERROR: 用法: $0 <scene_id> [processed_dir] [gpu_id]"
    echo "ERROR: 示例: $0 172 /path/to/processed 0"
    exit 1
fi

scene_idx="$1"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    processed_dir="/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed"
fi

# 规范化路径（移除末尾斜杠）
processed_dir="${processed_dir%/}"

echo "[INFO] Scene ID: $scene_idx"
echo "[INFO] Processed Dir: $processed_dir"

# 参数设置
################################################################################
gpu="${3:--1}"

camera_ids=(0 1 2)  # camera IDs to use, e.g., (0), (0 2 4)
# camera_ids=(0)  # camera IDs to use, e.g., (0), (0 2 4)
lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）

config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="waymo/3cams"
extra_config_info="adstudiov_pbr_3view"  # 额外信息，如 depth_loss

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
################################################################################

# GPU 选择优先级：
# 1) 第三个参数 gpu_id
# 2) 外部传入的 CUDA_VISIBLE_DEVICES（用于 batch/pipeline 统一调度）
# 3) 自动 pick_gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        gpu="${CUDA_VISIBLE_DEVICES}"
    fi
fi

if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"


output_root="outputfargs"
project_name="waymo_${scene_idx}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"

run_dir="${output_root}/${project_name}/${run_name}"

# 检查输出目录是否存在，防止覆盖
if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！"
    rm -r "$run_dir"
    # exit 1
fi
mkdir -p "$run_dir"

# 备份当前脚本到输出目录
script_name=$(basename "$0")
backup_path="${run_dir}/${script_name}"
cp "$0" "$backup_path"

# 启动训练
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py \
    --config_file $config_file \
    --output_root $output_root \
    --project $project_name \
    --run_name $run_name \
    dataset=$dataset_config \
    data.scene_idx=$scene_idx \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
