#!/bin/bash
###############################################################################
# Waymo 数据下载脚本
# 用法: ./waymo_download.sh [scene_id] [target_dir]
# 注意: 需要先执行 gcloud auth login
###############################################################################

# 参数处理
SCENE_ID="${1:-172}"
TARGET_DIR="${2:-/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/raw}"

echo "[INFO] Scene ID: $SCENE_ID"
echo "[INFO] Target Directory: $TARGET_DIR"

# 执行下载
python datasets/waymo/waymo_download.py \
    --target_dir "$TARGET_DIR" \
    --scene_ids "$SCENE_ID"