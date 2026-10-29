#!/bin/bash
###############################################################################
# NuScenes Pipeline 验证脚本
# 检查所有脚本的参数和路径是否正确
###############################################################################

echo "========================================="
echo "NuScenes Pipeline 完整性验证"
echo "========================================="
echo ""

# 测试场景
SCENE_NAME="scene-0061"
SPLIT="v1.0-mini"
SPLIT_SHORT="mini"
RAW_DATA_ROOT="/data/sets/nuscenes"
PROCESSED_DATA_ROOT="/data/nuscenes/processedads"
GPU_ID="0"
TRAIN_OUTPUT_ROOT="output6cams"
TRAIN_PROJECT_NAME="nuscene"
TRAIN_DATASET_CONFIG="nuscenes/6cams"
TRAIN_CAMERA_IDS="0,1,2,3,4,5"
RENDER_CAMERA_IDS="0,1,2,3,4,5"

# 计算场景目录名
scene_idx=$(echo "$SCENE_NAME" | sed 's/scene-0*//')
scene_dir_name=$(printf "%03d" "$scene_idx")

echo "✅ 场景转换测试："
echo "   输入: $SCENE_NAME"
echo "   索引: $scene_idx"
echo "   目录: $scene_dir_name"
echo ""

# 验证各阶段路径
echo "=== 阶段 1-3: Preprocess ==="
echo "✅ 调用: preprocess.sh $scene_idx 1 $RAW_DATA_ROOT $PROCESSED_DATA_ROOT $SPLIT 4 32 $GPU_ID"
echo "✅ 输出: $PROCESSED_DATA_ROOT/$SPLIT_SHORT/$scene_dir_name/"
echo ""

echo "=== 阶段 4: Extract Traffic Lights ==="
echo "✅ 调用: extract_traffic_lights_v3.sh $SCENE_NAME $PROCESSED_DATA_ROOT/$SPLIT_SHORT $GPU_ID"
echo "✅ 场景目录转换: $SCENE_NAME → $scene_dir_name"
echo "✅ 查找目录: $PROCESSED_DATA_ROOT/$SPLIT_SHORT/$scene_dir_name/"
echo ""

echo "=== 阶段 5: Train ==="
echo "✅ 调用: train.sh $SCENE_NAME $PROCESSED_DATA_ROOT/$SPLIT_SHORT 0 -1 $GPU_ID $TRAIN_OUTPUT_ROOT $TRAIN_DATASET_CONFIG $TRAIN_CAMERA_IDS"
echo "✅ 配置: data.data_root=$PROCESSED_DATA_ROOT/$SPLIT_SHORT"
echo "✅ 配置: dataset=$TRAIN_DATASET_CONFIG"
echo "✅ 配置: cameras=$TRAIN_CAMERA_IDS"
echo "✅ 输出: $TRAIN_OUTPUT_ROOT/$TRAIN_PROJECT_NAME/$SCENE_NAME"
echo ""

echo "=== 阶段 6: Render ==="
echo "✅ 调用: render_novel_trajectory.sh $SCENE_NAME $PROCESSED_DATA_ROOT/$SPLIT_SHORT $GPU_ID $TRAIN_OUTPUT_ROOT $RENDER_CAMERA_IDS"
echo "✅ GT目录转换: $SCENE_NAME → $scene_dir_name"
echo "✅ GT路径: $PROCESSED_DATA_ROOT/$SPLIT_SHORT/$scene_dir_name/images/"
echo "✅ Checkpoint路径: $TRAIN_OUTPUT_ROOT/$TRAIN_PROJECT_NAME/$SCENE_NAME"
echo ""

echo "=== Marker 文件 ==="
echo "✅ 位置: $PROCESSED_DATA_ROOT/$SPLIT_SHORT/$scene_dir_name/"
echo "   - .preprocess_complete"
echo "   - .extract_traffic_lights_complete"
echo "   - .train_complete"
echo "   - .render_complete"
echo ""

echo "========================================="
echo "关键修改点："
echo "========================================="
echo "✅ 1. pipeline.sh 统一管理 train/render 配置并显式传参"
echo "✅ 2. 训练和渲染验证目录统一到 $TRAIN_OUTPUT_ROOT/$TRAIN_PROJECT_NAME"
echo "✅ 3. 红绿灯阶段切换为 extract_traffic_lights_v3.sh"
echo "✅ 4. train.sh 接收 gpu/output_root/dataset/cameras"
echo "✅ 5. Marker目录使用数字格式 (061 而不是 scene-0061)"
echo ""

echo "========================================="
echo "待验证事项："
echo "========================================="
echo "⚠️  1. 建议测试流程："
echo "   a. 手动运行 preprocess.sh，检查输出目录"
echo "   b. 确认目录名为 061 而非 scene-0061"
echo "   c. 运行 pipeline.sh 单场景测试"
echo "   d. 检查所有 marker 和输出是否正确"
echo ""

echo "========================================="
echo "验证完成"
echo "========================================="
