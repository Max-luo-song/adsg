# 参数设置
################################################################################
scene_id="20251105_152839_QCOYSD504206_1240_1255"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 1 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar mix_novel_views\
    --skip_front_wide_side_cameras
