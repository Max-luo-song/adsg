#!/bin/bash
###############################################################################
# Waymo Batch Pipeline (Drivestudio) - GPU Pool Scheduler
# 批量处理多个场景，自动管理GPU资源池，使用 Drivestudio 代码库
#
# 用法: ./batch_drivestudio_pipeline.sh <scene_list.txt> [gpu_list]
# 示例: ./batch_drivestudio_pipeline.sh scenes.txt "0 1 2 3"
#       ./batch_drivestudio_pipeline.sh scenes.txt "0 2" (只使用GPU 0和2)
#
# 如果未指定 gpu_list，默认使用 0-7
###############################################################################

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="${PROJECT_ROOT}/logs/batch_drivestudio"
mkdir -p "$LOG_DIR"

# 默认配置
DEFAULT_GPUS="0 1 2 3 4 5 6 7"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 辅助函数
# ============================================================================
log_info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] [INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] [SUCCESS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] [WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] [ERROR] $1${NC}" >&2
}

# ============================================================================
# 参数解析
# ============================================================================
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <scene_list.txt> [gpu_list]"
    echo "示例: $0 scenes.txt \"0 1 2 3\""
    exit 1
fi

SCENE_LIST_FILE="$1"
GPU_LIST_STR="${2:-$DEFAULT_GPUS}"

if [[ ! -f "$SCENE_LIST_FILE" ]]; then
    log_error "场景列表文件不存在: $SCENE_LIST_FILE"
    exit 1
fi

# 解析 GPU 列表到数组
IFS=' ' read -r -a GPUS <<< "$GPU_LIST_STR"
NUM_GPUS=${#GPUS[@]}

if [[ $NUM_GPUS -eq 0 ]]; then
    log_error "未指定有效的 GPU"
    exit 1
fi

log_info "========================================="
log_info "Waymo Batch Pipeline (Drivestudio Mode)"
log_info "========================================="
log_info "场景文件: $SCENE_LIST_FILE"
log_info "可用 GPU: ${GPUS[*]} (共 $NUM_GPUS 个)"
log_info "日志目录: $LOG_DIR"

# ============================================================================
# 状态跟踪
# ============================================================================
declare -A GPU_PIDS
declare -A GPU_SCENES

# 初始化 GPU 状态
for i in "${!GPUS[@]}"; do
    GPU_PIDS[$i]=0
    GPU_SCENES[$i]=""
done

# 读取场景列表（忽略空行和注释）
mapfile -t ALL_SCENES < <(grep -v '^[[:space:]]*$' "$SCENE_LIST_FILE" | grep -v '^[[:space:]]*#')
TOTAL_SCENES=${#ALL_SCENES[@]}

log_info "待处理场景数: $TOTAL_SCENES"

# 任务统计
COMPLETED_COUNT=0
FAILED_COUNT=0
FAILED_SCENES=()

# ============================================================================
# 主循环
# ============================================================================
SCENE_IDX=0
trap 'log_warning "接收到中断信号，正在终止子进程..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT SIGTERM

while [[ $SCENE_IDX -lt $TOTAL_SCENES ]] || [[ $(jobs -r | wc -l) -gt 0 ]]; do
    
    # 1. 检查是否有任务完成
    for i in "${!GPUS[@]}"; do
        pid=${GPU_PIDS[$i]}
        
        # 如果 pid 不为 0，说明有任务在运行
        if [[ $pid -ne 0 ]]; then
            # 检查进程是否存活
            if ! kill -0 $pid 2>/dev/null; then
                # 进程已结束
                wait $pid
                exit_code=$?
                scene_id=${GPU_SCENES[$i]}
                
                # 清除状态
                GPU_PIDS[$i]=0
                GPU_SCENES[$i]=""
                
                if [[ $exit_code -eq 0 ]]; then
                    # 成功
                    log_success "Scene $scene_id finished on GPU ${GPUS[$i]}"
                    ((COMPLETED_COUNT++))
                else
                    # 失败
                    log_error "Scene $scene_id FAILED on GPU ${GPUS[$i]} (Exit Code: $exit_code)"
                    ((FAILED_COUNT++))
                    FAILED_SCENES+=("$scene_id")
                fi
            fi
        fi
    done
    
    # 2. 尝试调度新任务
    # 寻找空闲 GPU
    FREE_GPU_IDX=-1
    for i in "${!GPUS[@]}"; do
        if [[ ${GPU_PIDS[$i]} -eq 0 ]]; then
            FREE_GPU_IDX=$i
            break
        fi
    done
    
    # 如果有空闲 GPU 且 还有未分配的任务
    if [[ $FREE_GPU_IDX -ne -1 ]] && [[ $SCENE_IDX -lt $TOTAL_SCENES ]]; then
        # 获取下一个任务
        SCENE_ID="${ALL_SCENES[$SCENE_IDX]}"
        ((SCENE_IDX++))
        
        GPU_ID="${GPUS[$FREE_GPU_IDX]}"
        
        log_info "Starting Scene $SCENE_ID on GPU $GPU_ID ($SCENE_IDX/$TOTAL_SCENES)"
        
        # 异步启动任务 (调用 drivestudio_pipeline.sh)
        # 确保日志目录存在
        mkdir -p "${LOG_DIR}/logs"
        bash "${SCRIPT_DIR}/drivestudio_pipeline.sh" "$SCENE_ID" "$GPU_ID" > "${LOG_DIR}/logs/scene_${SCENE_ID}.log" 2>&1 &
        pid=$!
        
        # 记录状态
        GPU_PIDS[$FREE_GPU_IDX]=$pid
        GPU_SCENES[$FREE_GPU_IDX]=$SCENE_ID
        
        # 继续循环，以便尝试分配下一个任务给其他空闲 GPU
        continue 
    fi
    
    # 3. 如果所有 GPU 都忙，或者任务已分发完毕，则休眠等待
    # 只有当没有任何任务在运行时才退出循环 (由 while 条件控制)
    sleep 2
done

# ============================================================================
# 总结
# ============================================================================
log_info "========================================="
log_info "批量处理完成"
log_info "========================================="
log_info "总场景: $TOTAL_SCENES"
log_info "成功: $COMPLETED_COUNT"

if [[ $FAILED_COUNT -gt 0 ]]; then
    log_error "失败: $FAILED_COUNT"
    log_error "失败场景列表: ${FAILED_SCENES[*]}"
    exit 1
else
    log_success "所有任务执行成功"
    exit 0
fi
