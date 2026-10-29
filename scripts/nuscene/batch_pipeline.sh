#!/bin/bash
###############################################################################
# NuScenes Batch Pipeline - GPU Worker Pool Scheduler
#
# 用法:
#   ./batch_pipeline.sh <scene_list.txt> [num_gpus] [max_parallel_tasks] [split]
#
# 示例:
#   ./batch_pipeline.sh scenes.txt 4 4 v1.0-mini
#   ./batch_pipeline.sh scenes_ids.txt 8 8 full
#
# scene_list.txt 支持两种格式（每行一个，支持注释和空行）:
#   - scene-0061
#   - 0061 / 61  (会自动规范为 scene-0061)
#
# 调度规则:
#   - GPU 使用 0...(num_gpus-1)
#   - 并发上限 = min(max_parallel_tasks, num_gpus)
#   - 任一任务结束后，立即从队列补下一个任务
###############################################################################

set -euo pipefail

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="${PROJECT_ROOT}/logs/batch_nuscene"
mkdir -p "$LOG_DIR"

DEFAULT_NUM_GPUS=4
DEFAULT_SPLIT="v1.0-mini"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] [INFO] $*${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] [SUCCESS] $*${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] [WARN] $*${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] [ERROR] $*${NC}" >&2
}

normalize_scene() {
    local raw="$1"
    # 去掉首尾空格
    raw="$(echo "$raw" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ "$raw" =~ ^scene-([0-9]+)$ ]]; then
        local n="${BASH_REMATCH[1]}"
        printf "scene-%04d" "$((10#$n))"
        return 0
    fi

    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        printf "scene-%04d" "$((10#$raw))"
        return 0
    fi

    return 1
}

count_running() {
    local cnt=0
    local pid
    for pid in "${!PID_TO_SCENE[@]}"; do
        if [[ "${PID_ACTIVE[$pid]:-0}" == "1" ]]; then
            cnt=$((cnt + 1))
        fi
    done
    echo "$cnt"
}

find_free_gpu() {
    local i
    for i in "${!GPUS[@]}"; do
        if [[ "${GPU_BUSY[$i]}" == "0" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
}

start_task() {
    local scene="$1"
    local gpu_idx="$2"
    local gpu_id="${GPUS[$gpu_idx]}"

    local log_file="${LOG_DIR}/pipeline_${scene}.log"
    log_info "启动任务: ${scene} on GPU ${gpu_id}"

    bash "${SCRIPT_DIR}/pipeline.sh" "$scene" "$gpu_id" "$SPLIT" > "$log_file" 2>&1 &
    local pid=$!

    PID_TO_SCENE[$pid]="$scene"
    PID_TO_GPUIDX[$pid]="$gpu_idx"
    PID_ACTIVE[$pid]=1
    GPU_BUSY[$gpu_idx]=1
}

collect_finished() {
    local pid
    for pid in "${!PID_TO_SCENE[@]}"; do
        if [[ "${PID_ACTIVE[$pid]:-0}" != "1" ]]; then
            continue
        fi

        if ! kill -0 "$pid" 2>/dev/null; then
            local scene="${PID_TO_SCENE[$pid]}"
            local gpu_idx="${PID_TO_GPUIDX[$pid]}"
            local gpu_id="${GPUS[$gpu_idx]}"

            set +e
            wait "$pid"
            local exit_code=$?
            set -e

            PID_ACTIVE[$pid]=0
            GPU_BUSY[$gpu_idx]=0

            if [[ $exit_code -eq 0 ]]; then
                log_success "完成: ${scene} on GPU ${gpu_id}"
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            else
                log_error "失败: ${scene} on GPU ${gpu_id} (exit=${exit_code})"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                FAILED_SCENES+=("$scene")
            fi
        fi
    done
}

cleanup_all() {
    log_warning "接收到中断信号，正在停止所有子进程..."
    local pid
    for pid in "${!PID_TO_SCENE[@]}"; do
        if [[ "${PID_ACTIVE[$pid]:-0}" == "1" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    exit 1
}

# ============================================================================
# 参数解析
# ============================================================================
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <scene_list.txt> [num_gpus] [max_parallel_tasks] [split]"
    echo "示例: $0 scenes.txt 4 4 v1.0-mini"
    echo "示例: $0 scenes_ids.txt 8 8 full"
    exit 1
fi

SCENE_LIST_FILE="$1"
NUM_GPUS="${2:-$DEFAULT_NUM_GPUS}"
MAX_PARALLEL="${3:-$NUM_GPUS}"
SPLIT="${4:-$DEFAULT_SPLIT}"

if [[ ! -f "$SCENE_LIST_FILE" ]]; then
    log_error "场景列表文件不存在: $SCENE_LIST_FILE"
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
    log_error "num_gpus 必须是正整数，当前: $NUM_GPUS"
    exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -le 0 ]]; then
    log_error "max_parallel_tasks 必须是正整数，当前: $MAX_PARALLEL"
    exit 1
fi

if [[ "$MAX_PARALLEL" -gt "$NUM_GPUS" ]]; then
    log_warning "max_parallel_tasks($MAX_PARALLEL) > num_gpus($NUM_GPUS)，已自动限制为 $NUM_GPUS"
    MAX_PARALLEL="$NUM_GPUS"
fi

# 构造 GPU 列表: 0..num_gpus-1
GPUS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPUS+=("$i")
done

# 读取并规范化场景列表
ALL_SCENES=()
while IFS= read -r line || [[ -n "$line" ]]; do
    # 跳过空行和注释
    if [[ "$line" =~ ^[[:space:]]*$ ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    if normalized="$(normalize_scene "$line")"; then
        ALL_SCENES+=("$normalized")
    else
        log_warning "跳过非法场景ID: '$line'（支持 scene-XXXX 或纯数字）"
    fi
done < "$SCENE_LIST_FILE"

TOTAL_SCENES=${#ALL_SCENES[@]}
if [[ $TOTAL_SCENES -eq 0 ]]; then
    log_error "没有可执行的场景任务（文件为空或格式非法）"
    exit 1
fi

log_info "========================================="
log_info "NuScenes Batch Pipeline (Worker Pool)"
log_info "========================================="
log_info "场景文件: $SCENE_LIST_FILE"
log_info "数据集分割: $SPLIT"
log_info "GPU 数量: $NUM_GPUS"
log_info "并发上限: $MAX_PARALLEL"
log_info "GPU 列表: ${GPUS[*]}"
log_info "总任务数: $TOTAL_SCENES"
log_info "日志目录: $LOG_DIR"
log_info "即将开始调度任务..."

# ============================================================================
# 运行态状态
# ============================================================================
declare -A GPU_BUSY
for i in "${!GPUS[@]}"; do
    GPU_BUSY[$i]=0
done

declare -A PID_TO_SCENE
declare -A PID_TO_GPUIDX
declare -A PID_ACTIVE

COMPLETED_COUNT=0
FAILED_COUNT=0
FAILED_SCENES=()

trap cleanup_all SIGINT SIGTERM

# ============================================================================
# 调度循环
# ============================================================================
QUEUE_IDX=0

while true; do
    # 1) 回收已完成任务，释放GPU
    collect_finished

    # 2) 在并发和GPU可用前提下，持续补任务
    while [[ $QUEUE_IDX -lt $TOTAL_SCENES ]]; do
        running_now="$(count_running)"
        if [[ "$running_now" -ge "$MAX_PARALLEL" ]]; then
            break
        fi

        free_gpu_idx="$(find_free_gpu)"
        if [[ "$free_gpu_idx" == "-1" ]]; then
            break
        fi

        scene="${ALL_SCENES[$QUEUE_IDX]}"
        QUEUE_IDX=$((QUEUE_IDX + 1))
        start_task "$scene" "$free_gpu_idx"
    done

    # 3) 结束条件：队列清空 + 无运行任务
    running_now="$(count_running)"
    if [[ $QUEUE_IDX -ge $TOTAL_SCENES ]] && [[ "$running_now" -eq 0 ]]; then
        break
    fi

    sleep 1
done

# ============================================================================
# 总结
# ============================================================================
log_info "========================================="
log_info "批量处理完成"
log_info "========================================="
log_info "总任务: $TOTAL_SCENES"
log_success "成功: $COMPLETED_COUNT"
if [[ $FAILED_COUNT -gt 0 ]]; then
    log_error "失败: $FAILED_COUNT"
    log_error "失败场景: ${FAILED_SCENES[*]}"
    exit 1
else
    log_success "失败: $FAILED_COUNT"
    log_success "所有任务执行成功"
    exit 0
fi
