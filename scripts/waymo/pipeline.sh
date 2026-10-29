#!/bin/bash
###############################################################################
# Waymo Single Scene Processing Pipeline
# 处理单个Waymo场景的完整流程
# 用法: ./pipeline.sh <scene_id> [gpu_id]
###############################################################################

# ============================================================================
# 配置
# ============================================================================
# PROJECT_ROOT是当前项目路径，PARENT_ROOT是数据路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
LOG_DIR="${PROJECT_ROOT}/logs/pipeline"

# 数据路径 - 使用父级项目目录下的独立数据目录
RAW_DATA_ROOT="${PARENT_ROOT}/data/waymo/raw"              # 原始数据下载目录
PROCESSED_DATA_ROOT="${PARENT_ROOT}/data/waymo/processed"  # 处理后数据目录

# 默认GPU ID为0，如果指定了第二个参数则使用该GPU
GPU_ID="${2:-0}"

# 初始化日志文件（在validate_inputs之前需要）
SCENE_LOG="/tmp/pipeline_init.log"

set -euo pipefail

# ============================================================================
# 场景级隔离：获取场景特有的标志文件目录
# ============================================================================
get_scene_marker_dir() {
    # 返回该场景专有的标志文件存储目录
    # 这样不同场景的验证文件互不影响
    echo "${PROCESSED_DATA_ROOT}/scene_${SCENE_ID}"
}

# 初始化场景标志文件和锁文件
# 注意：这个函数在 main() 中 SCENE_ID 被设置后调用
init_scene_markers() {
    # 定义各阶段的标志文件（用于增量处理）
    # 所有标志文件使用场景级目录，确保多场景处理时互不干扰
    DOWNLOAD_MARKER="$(get_scene_marker_dir)/.download_complete"
    CONVERT_DATA_MARKER="$(get_scene_marker_dir)/.convert_data_complete"
    SEMANTIC_SEGMENT_MARKER="$(get_scene_marker_dir)/.semantic_segment_complete"
    EXTRACT_HUMAN_POSE_MARKER="$(get_scene_marker_dir)/.extract_human_pose_complete"
    EXTRACT_TL_MARKER="$(get_scene_marker_dir)/.extract_traffic_lights_complete"
    TRAIN_MARKER="$(get_scene_marker_dir)/.train_complete"
    RENDER_MARKER="$(get_scene_marker_dir)/.render_complete"

    # 定义各阶段的锁文件（用于并发控制）
    # 锁文件也使用场景级目录，避免不同场景竞争同一锁
    CONVERT_DATA_LOCK="$(get_scene_marker_dir)/.convert_data.lock"
    SEMANTIC_SEGMENT_LOCK="$(get_scene_marker_dir)/.semantic_segment.lock"
    EXTRACT_HUMAN_POSE_LOCK="$(get_scene_marker_dir)/.extract_human_pose.lock"
}

# ============================================================================
# 日志函数
# ============================================================================
log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [INFO] $*" | tee -a "$SCENE_LOG"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [ERROR] $*" | tee -a "$SCENE_LOG" >&2
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [SUCCESS] $*" | tee -a "$SCENE_LOG"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [WARN] $*" | tee -a "$SCENE_LOG"
}

# ============================================================================
# 检测函数：智能识别已完成的阶段
# ============================================================================
check_data_converted() {
    # 如果有场景级标志文件，直接返回成功
    # （标志文件表示数据已验证通过）
    [[ -f "$CONVERT_DATA_MARKER" ]] && return 0
    
    # 如果没有标志文件，则不认为数据已转换
    # 这确保没有标志文件就必须重新处理，即使目录存在
    return 1
}

check_segmentation_done() {
    # 如果有场景级标志文件，直接返回成功
    # （标志文件表示分割结果已验证通过）
    [[ -f "$SEMANTIC_SEGMENT_MARKER" ]] && return 0
    
    # 如果没有标志文件，则不认为分割已完成
    return 1
}

check_pose_extracted() {
    # 如果有场景级标志文件，直接返回成功
    # （标志文件表示姿态提取结果已验证通过）
    [[ -f "$EXTRACT_HUMAN_POSE_MARKER" ]] && return 0
    
    # 如果没有标志文件，则不认为姿态提取已完成
    return 1
}

# ============================================================================
# 数据验证函数：严格检查每个步骤的输出完整性
# ============================================================================


verify_downloaded_data() {
    # 验证原始数据是否成功下载
    # 通过查找 data/waymo_train_list.txt 来确定应该下载哪个文件
    
    if [[ ! -d "$RAW_DATA_ROOT" ]]; then
        log_error "验证失败: 原始数据目录不存在 $RAW_DATA_ROOT"
        return 1
    fi
    
    # 尝试查找对应的 tfrecord 文件名
    local train_list_file="${PROJECT_ROOT}/data/waymo_train_list.txt"
    if [[ -f "$train_list_file" ]]; then
        # 读取对应行 (注意: sed 是 1-based index，而 python 列表是 0-based，所以 scene_id 对应行号是 scene_id + 1)
        # 假设 SCENE_ID 是 train_list 中的索引
        local line_num=$((SCENE_ID + 1))
        local expected_file=$(sed -n "${line_num}p" "$train_list_file" | tr -d '\r\n')
        
        if [[ -n "$expected_file" ]]; then
            # 兼容两种命名：直接是文件名 或 文件名.tfrecord
            local expected_path=""
            if [[ "$expected_file" == *".tfrecord" ]]; then
                expected_path="${RAW_DATA_ROOT}/${expected_file}"
            else
                expected_path="${RAW_DATA_ROOT}/${expected_file}.tfrecord"
            fi
            
            log_info "检查特定数据文件: $expected_path"
            
            if [[ -f "$expected_path" ]] && [[ -s "$expected_path" ]]; then
                log_info "验证成功: 目标数据文件存在且非空"
                return 0
            else
                log_warning "找不到目标文件 $expected_file"
                log_warning "尝试按 pattern 模糊查找..."
                # 有时候列表里的名字可能不包含后缀，或者有些许差异
                # 去掉可能的后缀再查
                local base_name="${expected_file%.tfrecord}"
                local fuzzy_match=$(find "$RAW_DATA_ROOT" -name "*${base_name}*" -type f | head -n 1)
                if [[ -n "$fuzzy_match" ]] && [[ -s "$fuzzy_match" ]]; then
                    log_info "模糊匹配成功: $fuzzy_match"
                    return 0
                fi
                
                log_error "验证失败: 找不到对应的数据文件 $expected_file"
                return 1
            fi
        else
            log_warning "无法从列表文件中找到 Scene $SCENE_ID 对应的文件名，降级为检查任意 tfrecord"
        fi
    fi
    
    # 降级检查：检查是否有任何 .tfrecord 文件（仅当无法查找到特定文件时）
    local tfrecord_count=$(find "$RAW_DATA_ROOT" -maxdepth 1 -name "*.tfrecord" -type f 2>/dev/null | wc -l)
    if [[ $tfrecord_count -eq 0 ]]; then
        log_error "验证失败: $RAW_DATA_ROOT 中找不到任何 .tfrecord 文件"
        return 1
    fi
    
    # 验证至少有一个文件大小不为0
    local valid_count=$(find "$RAW_DATA_ROOT" -maxdepth 1 -name "*.tfrecord" -type f -size +0 2>/dev/null | wc -l)
    if [[ $valid_count -eq 0 ]]; then
        log_error "验证失败: 所有 .tfrecord 文件都是空的"
        return 1
    fi
    
    log_info "验证成功: 找到 $valid_count 个有效的数据文件 (无法确认是否属于当前场景)"
    return 0
}

get_training_scene_dir() {
    local scene_dir="$PROCESSED_DATA_ROOT/training/$SCENE_ID"
    
    # 1. 检查直接匹配
    if [[ -d "$scene_dir" ]]; then
        echo "$scene_dir"
        return 0
    fi
    
    # 2. 检查零填充 (023 vs 23)
    local padded_id=$(printf "%03d" "$SCENE_ID" 2>/dev/null || echo "$SCENE_ID")
    if [[ -d "$PROCESSED_DATA_ROOT/training/$padded_id" ]]; then
        echo "$PROCESSED_DATA_ROOT/training/$padded_id"
        return 0
    fi
    
    # 3. 检查模糊匹配
    local found_dirs=$(find "$PROCESSED_DATA_ROOT/training" -maxdepth 1 -type d -name "*$SCENE_ID" 2>/dev/null)
    if [[ -n "$found_dirs" ]]; then
        echo "$found_dirs" | head -n 1
        return 0
    fi
    
    # 4. 检查旧结构 (直接在 training 下)
    if [[ -d "$PROCESSED_DATA_ROOT/training/images" ]]; then
        echo "$PROCESSED_DATA_ROOT/training"
        return 0
    fi
    
    return 1
}

verify_converted_data() {
    # 验证转换后的数据是否完整
    # 至少应该存在 training 目录
    if [[ ! -d "$PROCESSED_DATA_ROOT/training" ]]; then
        log_error "验证失败: 找不到 training 目录"
        return 1
    fi
    
    # 获取场景数据目录
    local scene_dir=$(get_training_scene_dir)
    if [[ -z "$scene_dir" ]]; then
         log_error "验证失败: 找不到场景数据目录。期望在 training/ 下找到 $SCENE_ID 相关目录"
         return 1
    fi
    
    log_info "验证数据目录: $scene_dir"
    
    # 检查是否有关键文件
    local required_count=0
    
    # 检查 metadata.json 或 frame_info.json
    if [[ -f "$scene_dir/metadata.json" ]] || [[ -f "$scene_dir/frame_info.json" ]]; then
        required_count=$((required_count + 1))
    fi
    
    # 至少应该有一些数据文件（帧图像、标定信息等）
    if [[ -d "$scene_dir/images" ]]; then
        local image_count=$(find "$scene_dir/images" -type f 2>/dev/null | wc -l)
        if [[ $image_count -gt 0 ]]; then
            required_count=$((required_count + 1))
        else
            log_error "验证失败: images 目录为空"
            return 1
        fi
    fi
    
    # 至少需要一些有效数据
    if [[ $required_count -gt 0 ]]; then
        return 0
    fi
    
    log_error "验证失败: $scene_dir 中没有有效数据"
    return 1
}

verify_segmentation_data() {
    # 验证语义分割的输出
    # 注意：extract_masks.py 生成的不是 segmentation_masks 目录
    # 而是生成:
    #   training/<scene_id>/sky_masks/
    #   training/<scene_id>/road_masks/
    #   training/<scene_id>/dynamic_masks/{human,vehicle}/
    
    # 获取场景数据目录 (使用新函数)
    local scene_dir=$(get_training_scene_dir)
    if [[ -z "$scene_dir" ]]; then
         log_error "验证失败: 找不到场景数据目录"
         return 1
    fi
    
    # 验证 sky_masks
    if [[ ! -d "$scene_dir/sky_masks" ]]; then
        log_error "验证失败: 找不到 sky_masks 目录在 $scene_dir"
        return 1
    fi
    local sky_count=$(find "$scene_dir/sky_masks" -type f 2>/dev/null | wc -l)
    if [[ $sky_count -eq 0 ]]; then
        log_error "验证失败: sky_masks 目录为空"
        return 1
    fi
    
    # 验证 road_masks
    if [[ ! -d "$scene_dir/road_masks" ]]; then
        log_error "验证失败: 找不到 road_masks 目录在 $scene_dir"
        return 1
    fi
    local road_count=$(find "$scene_dir/road_masks" -type f 2>/dev/null | wc -l)
    if [[ $road_count -eq 0 ]]; then
        log_error "验证失败: road_masks 目录为空"
        return 1
    fi
    
    # 验证 dynamic_masks (如果脚本中启用了 process_dynamic_mask)
    # 假设总是生成 dynamic_masks 因为这是默认配置
    if [[ ! -d "$scene_dir/dynamic_masks" ]]; then
        log_warning "警告: 找不到 dynamic_masks 目录 (可能未启用)"
    else
        local dyn_count=$(find "$scene_dir/dynamic_masks" -type f 2>/dev/null | wc -l)
        if [[ $dyn_count -eq 0 ]]; then
             log_warning "警告: dynamic_masks 目录存在但为空"
        fi
    fi
    
    log_info "验证成功: 语义分割数据完整 (Sky: $sky_count, Road: $road_count)"
    return 0
}

verify_pose_extracted() {
    # 验证人体姿态提取的输出
    # extract_human_pose.sh -> humanpose_process.py
    # 输出: training/<scene_id>/humanpose/smpl.pkl
    
    local scene_dir=$(get_training_scene_dir)
    if [[ -z "$scene_dir" ]]; then
         log_error "验证失败: 找不到场景数据目录"
         return 1
    fi
    
    if [[ ! -f "$scene_dir/humanpose/smpl.pkl" ]]; then
        log_error "验证失败: 找不到 smpl.pkl 在 $scene_dir/humanpose/"
        return 1
    fi
    
    return 0
}

verify_traffic_lights_data() {
    # 验证红绿灯提取的输出
    # extract_traffic_lights.sh -> extract_traffic_lights.py
    # 输出: training/<scene_id>/instances/instances_info.json (合并模式)
    #       training/<scene_id>/dynamic_masks/traffic_light/ (Masks)
    
    local scene_dir=$(get_training_scene_dir)
    if [[ -z "$scene_dir" ]]; then
         log_error "验证失败: 找不到场景数据目录"
         return 1
    fi
    
    # 验证 instances_info.json
    if [[ ! -f "$scene_dir/instances/instances_info.json" ]] && \
       [[ ! -f "$scene_dir/instances/traffic_light_instances_info.json" ]]; then
        log_error "验证失败: 找不到红绿灯实例信息 JSON"
        return 1
    fi
    
    return 0
}

cleanup_previous_traffic_lights_outputs() {
    # 清理上一轮红绿灯结果，确保每次预处理都从干净状态开始
    local scene_dir
    scene_dir=$(get_training_scene_dir || true)

    if [[ -n "${scene_dir:-}" ]] && [[ -d "$scene_dir" ]]; then
        rm -f "$scene_dir/instances/traffic_light_instances_info.json"
        rm -f "$scene_dir/instances/traffic_light_frame_instances.json"
        rm -rf "$scene_dir/dynamic_masks/traffic_light"
        rm -rf "$scene_dir/fine_dynamic_masks/traffic_light"
        log_info "已清理旧红绿灯标注与 masks: $scene_dir"
    else
        log_warning "未找到场景目录，跳过旧红绿灯输出清理"
    fi

    rm -f "$EXTRACT_TL_MARKER"
}

verify_training_complete() {
    # 验证模型训练是否完成
    # train.sh 输出到 output/waymo_${SCENE_ID}/...
    
    local train_output_dir="$PROJECT_ROOT/outputfargs/waymo_${SCENE_ID}"
    
    if [[ ! -d "$train_output_dir" ]]; then
        log_error "验证失败: 找不到训练输出目录 $train_output_dir"
        return 1
    fi
    
    # 检查是否有 checkpoint
    local checkpoints=$(find "$train_output_dir" -name "*.pth" | head -n 1)
    if [[ -z "$checkpoints" ]]; then
        log_error "验证失败: 在 $train_output_dir 中找不到模型权重 (.pth)"
        return 1
    fi
    
    return 0
}

verify_rendered_output() {
    # 验证渲染输出
    # render_novel_trajectory.sh 输出到 output/waymo_${SCENE_ID}/.../novel_traj/
    
    local train_output_dir="$PROJECT_ROOT/output/waymo_${SCENE_ID}"
    
    if [[ ! -d "$train_output_dir" ]]; then
        log_error "验证失败: 找不到训练输出目录 (渲染检查依赖)"
        return 1
    fi
    
    # 查找 novel_traj 目录
    local novel_traj_dirs=$(find "$train_output_dir" -type d -name "novel_traj")
    if [[ -z "$novel_traj_dirs" ]]; then
        log_error "验证失败: 找不到 novel_traj 渲染输出目录"
        return 1
    fi
    
    # 检查是否有视频或图片输出
    local render_files=$(find "$novel_traj_dirs" -name "*.mp4" -o -name "*.jpg" -o -name "*.png" | head -n 1)
    if [[ -z "$render_files" ]]; then
        log_error "验证失败: 找不到渲染生成的视频或图片"
        return 1
    fi
    
    return 0
}

# ============================================================================
# 输入验证
# ============================================================================
validate_inputs() {
    if [[ $# -lt 1 ]]; then
        echo "ERROR: 用法: $0 <scene_id> [gpu_id]"
        echo "ERROR: 示例: $0 12345 0"
        exit 1
    fi

    SCENE_ID="$1"
    
    if ! [[ "$SCENE_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: scene_id 必须是数字，收到: $SCENE_ID"
        exit 1
    fi

    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: gpu_id 必须是数字，收到: $GPU_ID"
        exit 1
    fi

     log_info "========================================="
    log_info "Waymo Pipeline - Scene $SCENE_ID"
    log_info "========================================="
    log_info "GPU ID: $GPU_ID"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Raw Data Root: $RAW_DATA_ROOT"
    log_info "Processed Data Root: $PROCESSED_DATA_ROOT"
    log_info "Log Dir: $SCENE_LOG"
}

# ============================================================================
# GCloud 认证
# ============================================================================
gcloud_auth() {
    log_info "========== GCloud Authentication =========="
    
    # 检查gcloud是否已安装
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI 未安装或不在PATH中"
        log_error "请先安装 Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
        return 1
    fi
    
    # 检查是否已认证
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        local active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
        if [[ -n "$active_account" ]]; then
            log_info "已使用账户认证: $active_account"
            log_info "跳过 gcloud login 流程"
            return 0
        fi
    fi
    
    log_info "需要进行 gcloud 认证，请按照提示登陆..."
    log_info "浏览器会自动打开，请使用您的 Google 账户登陆"
    
    # 执行 gcloud auth login
    if gcloud auth login; then
        log_success "gcloud 认证成功"
        
        # 验证认证
        local account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
        if [[ -n "$account" ]]; then
            log_info "当前认证账户: $account"
            return 0
        else
            log_error "认证后无法获取账户信息"
            return 1
        fi
    else
        log_error "gcloud 认证失败"
        return 1
    fi
}

# ============================================================================
# 阶段：数据下载
# ============================================================================
stage_download() {
    log_info "========== STAGE 1: Download Data =========="
    
    # 创建场景级标志文件目录
    mkdir -p "$(dirname "$DOWNLOAD_MARKER")"
    
    # 检查是否已下载且数据有效
    if [[ -f "$DOWNLOAD_MARKER" ]]; then
        if verify_downloaded_data; then
            log_warning "数据已下载且有效（标志文件存在: $DOWNLOAD_MARKER）"
            log_info "跳过下载阶段"
            return 0
        else
            log_warning "标志文件存在但数据无效，重新下载"
            rm -f "$DOWNLOAD_MARKER"
        fi
    fi

    # 创建原始数据目录
    mkdir -p "$RAW_DATA_ROOT"
    
    # 检查是否已有数据（通过 verify_downloaded_data 智能检查）
    if verify_downloaded_data; then
        log_warning "检测到已存在有效的数据文件"
        log_info "标记为已下载，跳过下载阶段"
        # 创建标志以供后续阶段检查
        mkdir -p "$(dirname "$DOWNLOAD_MARKER")"
        touch "$DOWNLOAD_MARKER"
        return 0
    fi

    log_info "开始下载 Scene $SCENE_ID 的数据至: $RAW_DATA_ROOT"
    
    if [[ -f "${SCRIPT_DIR}/waymo_download.sh" ]]; then
        # 向下载脚本传递原始数据存储目录
        bash "${SCRIPT_DIR}/waymo_download.sh" "$SCENE_ID" "$RAW_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            # 验证下载的数据
            if verify_downloaded_data; then
                log_success "数据下载完成且验证通过"
                mkdir -p "$(dirname "$DOWNLOAD_MARKER")"
                touch "$DOWNLOAD_MARKER"
            else
                log_error "数据下载成功但验证失败"
                return 1
            fi
        else
            log_error "数据下载失败"
            return 1
        fi
    else
        log_error "找不到 waymo_download.sh 脚本"
        return 1
    fi
}

# ============================================================================
# 阶段：数据转换
# ============================================================================
stage_convert_data() {
    log_info "========== STAGE 2: Convert Data =========="
    
    # 确保锁文件目录存在
    mkdir -p "$(dirname "$CONVERT_DATA_LOCK")"
    {
        flock 200 || { log_error "无法获取转换数据锁"; return 1; }
        
        # 智能检测：检查标志文件或实际输出目录
        if check_data_converted; then
            log_warning "数据已转换（检测到已处理的数据）"
            log_info "跳过转换阶段"
            # 确保标志文件存在（防止下次重复检测）
            mkdir -p "$(dirname "$CONVERT_DATA_MARKER")"
            touch "$CONVERT_DATA_MARKER"
            return 0
        fi

        # 检查必要的前置条件
        if [[ ! -f "$DOWNLOAD_MARKER" ]]; then
            log_error "数据未下载，无法进行转换"
            return 1
        fi

        # 创建处理后数据目录
        mkdir -p "$PROCESSED_DATA_ROOT"

        log_info "开始转换 Scene $SCENE_ID 的数据..."
        log_info "输入路径: $RAW_DATA_ROOT"
        log_info "输出路径: $PROCESSED_DATA_ROOT"
        
        if [[ -f "${SCRIPT_DIR}/convert_data.sh" ]]; then
            bash "${SCRIPT_DIR}/convert_data.sh" "$SCENE_ID" "$RAW_DATA_ROOT" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                # 验证转换后的数据
                if verify_converted_data; then
                    log_success "数据转换完成且验证通过"
                    mkdir -p "$(dirname "$CONVERT_DATA_MARKER")"
                    touch "$CONVERT_DATA_MARKER"
                else
                    log_error "数据转换成功但验证失败，输出数据不完整"
                    return 1
                fi
            else
                log_error "数据转换失败"
                return 1
            fi
        else
            log_error "找不到 convert_data.sh 脚本"
            return 1
        fi
    } 200>"$CONVERT_DATA_LOCK"
}

# ============================================================================
# 阶段：语义分割
# ============================================================================
stage_semantic_segment() {
    log_info "========== STAGE 3: Semantic Segmentation =========="
    
    # 确保锁文件目录存在
    mkdir -p "$(dirname "$SEMANTIC_SEGMENT_LOCK")"
    {
        flock 200 || { log_error "无法获取语义分割锁"; return 1; }
        
        # 智能检测：检查标志文件或实际输出目录
        if check_segmentation_done; then
            log_warning "语义分割已完成（检测到已处理的数据）"
            log_info "跳过语义分割阶段"
            # 确保标志文件存在（防止下次重复检测）
            mkdir -p "$(dirname "$SEMANTIC_SEGMENT_MARKER")"
            touch "$SEMANTIC_SEGMENT_MARKER"
            return 0
        fi

        # 检查必要的前置条件
        if ! check_data_converted; then
            log_error "数据未转换，无法进行语义分割"
            return 1
        fi

        log_info "开始进行 Scene $SCENE_ID 的语义分割..."
        log_info "数据路径: $PROCESSED_DATA_ROOT"
        
        if [[ -f "${SCRIPT_DIR}/semantic_segment.sh" ]]; then
            bash "${SCRIPT_DIR}/semantic_segment.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                # 验证分割结果
                if verify_segmentation_data; then
                    log_success "语义分割完成且验证通过"
                    mkdir -p "$(dirname "$SEMANTIC_SEGMENT_MARKER")"
                    touch "$SEMANTIC_SEGMENT_MARKER"
                else
                    log_error "语义分割成功但验证失败，输出数据不完整"
                    return 1
                fi
            else
                log_error "语义分割失败"
                return 1
            fi
        else
            log_error "找不到 semantic_segment.sh 脚本"
            return 1
        fi
    } 200>"$SEMANTIC_SEGMENT_LOCK"
}

# ============================================================================
# 阶段：SMPL 人体姿态提取
# ============================================================================
stage_extract_human_pose() {
    log_info "========== STAGE 4: Extract Human Pose =========="
    
    # 确保锁文件目录存在
    mkdir -p "$(dirname "$EXTRACT_HUMAN_POSE_LOCK")"
    {
        flock 200 || { log_error "无法获取人体姿态提取锁"; return 1; }
        
        # 智能检测：检查标志文件或实际输出目录
        if check_pose_extracted; then
            log_warning "人体姿态已提取（检测到已处理的数据）"
            log_info "跳过人体姿态提取阶段"
            # 确保标志文件存在（防止下次重复检测）
            mkdir -p "$(dirname "$EXTRACT_HUMAN_POSE_MARKER")"
            touch "$EXTRACT_HUMAN_POSE_MARKER"
            return 0
        fi

        # 检查必要的前置条件
        if ! check_segmentation_done; then
            log_error "语义分割未完成，无法提取人体姿态"
            return 1
        fi

        log_info "开始提取 Scene $SCENE_ID 的人体姿态..."
        log_info "数据路径: $PROCESSED_DATA_ROOT"
        
        if [[ -f "${SCRIPT_DIR}/extract_human_pose.sh" ]]; then
            bash "${SCRIPT_DIR}/extract_human_pose.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                # 验证人体姿态提取结果
                if verify_pose_extracted; then
                    log_success "人体姿态提取完成且验证通过"
                    mkdir -p "$(dirname "$EXTRACT_HUMAN_POSE_MARKER")"
                    touch "$EXTRACT_HUMAN_POSE_MARKER"
                else
                    log_error "人体姿态提取成功但验证失败，输出数据不完整"
                    return 1
                fi
            else
                log_error "人体姿态提取失败"
                return 1
            fi
        else
            log_error "找不到 extract_human_pose.sh 脚本"
            return 1
        fi
    } 200>"$EXTRACT_HUMAN_POSE_LOCK"
}

# ============================================================================
# 阶段：红绿灯提取
# ============================================================================
stage_extract_traffic_lights() {
    log_info "========== STAGE 5: Extract Traffic Lights =========="
    
    # 创建场景级标志文件目录
    mkdir -p "$(dirname "$EXTRACT_TL_MARKER")"

    # 每次预处理都强制清理并重跑红绿灯提取，避免历史残留
    cleanup_previous_traffic_lights_outputs

    # 检查必要的前置条件
    if [[ ! -f "$EXTRACT_HUMAN_POSE_MARKER" ]]; then
        log_error "人体姿态未提取，无法提取红绿灯"
        return 1
    fi

    log_info "开始提取 Scene $SCENE_ID 的红绿灯..."
    log_info "数据路径: $PROCESSED_DATA_ROOT"
    
    if [[ -f "${SCRIPT_DIR}/extract_traffic_lights_v3.sh" ]]; then
        TL_EXPAND_RATIO="${TL_EXPAND_RATIO:-0.45}" \
        bash "${SCRIPT_DIR}/extract_traffic_lights_v3.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            # 验证红绿灯提取结果
            if verify_traffic_lights_data; then
                log_success "红绿灯提取完成且验证通过"
                mkdir -p "$(dirname "$EXTRACT_TL_MARKER")"
                touch "$EXTRACT_TL_MARKER"
            else
                log_error "红绿灯提取成功但验证失败，输出数据不完整"
                return 1
            fi
        else
            log_error "红绿灯提取失败"
            return 1
        fi
    else
        log_error "找不到 extract_traffic_lights.sh 脚本"
        return 1
    fi
}

# ============================================================================
# 阶段：模型训练
# ============================================================================
stage_train() {
    log_info "========== STAGE 6: Training =========="
    
    # 创建场景级标志文件目录
    mkdir -p "$(dirname "$TRAIN_MARKER")"
    
    # 检查是否已训练且数据有效
    if [[ -f "$TRAIN_MARKER" ]]; then
        if verify_training_complete; then
            log_warning "模型已训练且有效（标志文件存在: $TRAIN_MARKER）"
            log_info "跳过训练阶段"
            return 0
        else
            log_warning "标志文件存在但模型无效，重新训练"
            rm -f "$TRAIN_MARKER"
        fi
    fi

    # 检查必要的前置条件
    if [[ ! -f "$EXTRACT_TL_MARKER" ]]; then
        log_error "红绿灯未提取，无法进行训练"
        return 1
    fi

    log_info "开始训练 Scene $SCENE_ID 的模型（GPU: $GPU_ID）..."
    log_info "数据路径: $PROCESSED_DATA_ROOT"
    
    if [[ -f "${SCRIPT_DIR}/train.sh" ]]; then
        # 传递GPU ID和数据路径给训练脚本
        CUDA_VISIBLE_DEVICES="$GPU_ID" bash "${SCRIPT_DIR}/train.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            # 验证训练结果
            if verify_training_complete; then
                log_success "模型训练完成且验证通过"
                mkdir -p "$(dirname "$TRAIN_MARKER")"
                touch "$TRAIN_MARKER"
            else
                log_error "模型训练成功但验证失败，训练输出不完整"
                return 1
            fi
        else
            log_error "模型训练失败"
            return 1
        fi
    else
        log_error "找不到 train.sh 脚本"
        return 1
    fi
}

# ============================================================================
# 阶段：新轨迹渲染
# ============================================================================
stage_render() {
    log_info "========== STAGE 7: Render Novel Trajectory =========="
    
    # 创建场景级标志文件目录
    mkdir -p "$(dirname "$RENDER_MARKER")"
    
    # 检查是否已渲染且数据有效
    if [[ -f "$RENDER_MARKER" ]]; then
        if verify_rendered_output; then
            log_warning "轨迹已渲染且有效（标志文件存在: $RENDER_MARKER）"
            log_info "跳过渲染阶段"
            return 0
        else
            log_warning "标志文件存在但渲染结果无效，重新渲染"
            rm -f "$RENDER_MARKER"
        fi
    fi

    # 检查必要的前置条件
    if [[ ! -f "$TRAIN_MARKER" ]]; then
        log_error "模型未训练，无法进行渲染"
        return 1
    fi

    log_info "开始渲染 Scene $SCENE_ID 的新轨迹（GPU: $GPU_ID）..."
    log_info "数据路径: $PROCESSED_DATA_ROOT"
    
    if [[ -f "${SCRIPT_DIR}/render_novel_trajectory.sh" ]]; then
        # 传递GPU ID和数据路径给渲染脚本
        CUDA_VISIBLE_DEVICES="$GPU_ID" bash "${SCRIPT_DIR}/render_novel_trajectory.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            # 验证渲染结果
            if verify_rendered_output; then
                log_success "新轨迹渲染完成且验证通过"
                mkdir -p "$(dirname "$RENDER_MARKER")"
                touch "$RENDER_MARKER"
            else
                log_error "新轨迹渲染成功但验证失败，渲染输出不完整"
                return 1
            fi
        fi
    else
        log_error "找不到 render_novel_trajectory.sh 脚本"
        return 1
    fi
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    # 先验证基本输入（不需要日志）
    if [[ $# -lt 1 ]]; then
        echo "ERROR: 用法: $0 <scene_id> [gpu_id]"
        echo "ERROR: 示例: $0 12345 0"
        exit 1
    fi

    local scene_id_arg="$1"
    if ! [[ "$scene_id_arg" =~ ^[0-9]+$ ]]; then
        echo "ERROR: scene_id 必须是数字，收到: $scene_id_arg"
        exit 1
    fi

    # 设置SCENE_ID以初始化日志文件
    SCENE_ID="$scene_id_arg"
    
    # 初始化场景级标志和锁文件
    init_scene_markers
    
    # 初始化日志目录和文件
    mkdir -p "$LOG_DIR"
    export SCENE_LOG="${LOG_DIR}/scene_${SCENE_ID}.log"
    
    # 现在验证输入（使用日志）
    validate_inputs "$@"
    
    # 记录开始时间
    local start_time=$(date +%s)
    log_info "开始处理 Scene $SCENE_ID"
    
    # 在下载之前先进行 gcloud 认证
    log_info "========================================="
    if ! gcloud_auth; then
        log_error "Pipeline 中止：gcloud 认证失败"
        exit 1
    fi
    log_info "========================================="
    
    # 执行各阶段
    if ! stage_download; then
        log_error "Pipeline 中止：数据下载失败"
        exit 1
    fi
    
    if ! stage_convert_data; then
        log_error "Pipeline 中止：数据转换失败"
        exit 1
    fi
    
    if ! stage_semantic_segment; then
        log_error "Pipeline 中止：语义分割失败"
        exit 1
    fi
    
    if ! stage_extract_human_pose; then
        log_error "Pipeline 中止：人体姿态提取失败"
        exit 1
    fi
    
    if ! stage_extract_traffic_lights; then
        log_error "Pipeline 中止：红绿灯提取失败"
        exit 1
    fi
    
    if ! stage_train; then
        log_error "Pipeline 中止：模型训练失败"
        exit 1
    fi
    
    if ! stage_render; then
        log_error "Pipeline 中止：新轨迹渲染失败"
        exit 1
    fi
    
    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    log_success "========================================="
    log_success "Pipeline 完成！Scene $SCENE_ID"
    log_success "总耗时: ${hours}h ${minutes}m ${seconds}s"
    log_success "========================================="
}

# 运行主函数
main "$@"
