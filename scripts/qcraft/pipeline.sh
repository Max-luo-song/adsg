# 参数设置
################################################################################
scene_id="621"
gpu=-1
segformer_path=/opt/mmsegmentation
camera_ids=(0 1 2 3 5 6 9 10 12)
raw_data_root="/nas_thoru/scenario_data/"
processed_data_root="/nas_thoru/scenario_output/zhangyingjun/processed"
work_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
################################################################################

# step 1: preprocess
source /opt/conda/etc/profile.d/conda.sh && conda activate main
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root ${raw_data_root} \
    --target_dir ${processed_data_root} \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar \
    --skip_front_wide_side_cameras

# step 2: extract mask
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh
source /opt/conda/etc/profile.d/conda.sh && conda activate mmseg2

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks_mmseg2.py \
    --data_root ${processed_data_root}/training \
    --config=${segformer_path}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py \
    --checkpoint=${segformer_path}/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
    --split_file data/qcraft_scenes.txt \
    --process_dynamic_mask

# step 3: process human pose
source /opt/conda/etc/profile.d/conda.sh && conda activate 4d-humans
export PYTHONPATH=$PYTHONPATH:${work_root}/third_party/Humans4D

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --data_root ${processed_data_root}/training \
    --dataset qcraft \
    --scene_id $scene_id

# 4: train
source /opt/conda/etc/profile.d/conda.sh && conda activate main

lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）

config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/9cams_${lidar_type}"
extra_config_info=""  # 额外信息，如 depth_loss

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
################################################################################

output_root="output"
project_name="qcraft_${scene_id}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}"

run_dir="${output_root}/${project_name}/${run_name}"

# 检查输出目录是否存在，防止覆盖
if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！"
    exit 1
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
    data.scene_idx=$scene_id \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
