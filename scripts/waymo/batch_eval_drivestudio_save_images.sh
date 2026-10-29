#!/usr/bin/env bash
set -euo pipefail

# 用法:
# 1. 默认扫描 ../drivestudio/output/waymo 下 basename 匹配 origintest_* 的结果目录
#    bash scripts/waymo/batch_eval_drivestudio_save_images.sh
#
# 2. 如果你的 checkpoint 都在一个单独目录里，例如 origintest_23videos/
#    bash scripts/waymo/batch_eval_drivestudio_save_images.sh \
#      /inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio/output/waymo/origintest_23videos \
#      0
#
# 3. 只预览不执行
#    DRY_RUN=1 bash scripts/waymo/batch_eval_drivestudio_save_images.sh <search_root> -1
#
# 可选环境变量:
#   DRIVESTUDIO_ROOT: drivestudio 仓库路径，默认自动定位到 ../drivestudio
#   RUN_FILTER:       结果目录 basename 过滤模式，默认在 search_root=.../waymo 时为 origintest_*，否则为 *
#   FORCE_EVAL=1:     即使已经存在 PNG 结果也强制重跑
#   DRY_RUN=1:        只打印将执行的场景，不真正运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRIVESTUDIO_ROOT="${DRIVESTUDIO_ROOT:-$(cd "${OMNI_ROOT}/../drivestudio" && pwd)}"

SEARCH_ROOT="${1:-${DRIVESTUDIO_ROOT}/output/waymo}"
GPU_ID="${2:--1}"
if (( $# > 0 )); then
    shift
fi
if (( $# > 0 )); then
    shift
fi
EXTRA_EVAL_ARGS=("$@")

RUN_FILTER="${RUN_FILTER:-}"
if [[ -z "${RUN_FILTER}" ]]; then
    if [[ "$(basename "${SEARCH_ROOT}")" == "waymo" ]]; then
        RUN_FILTER="origintest_*"
    else
        RUN_FILTER="*"
    fi
fi

FORCE_EVAL="${FORCE_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${DRIVESTUDIO_ROOT}" ]]; then
    echo "错误: drivestudio 仓库不存在: ${DRIVESTUDIO_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${SEARCH_ROOT}" ]]; then
    echo "错误: 搜索目录不存在: ${SEARCH_ROOT}" >&2
    exit 1
fi

if [[ "${GPU_ID}" == "-1" ]]; then
    # 兼容自动挑卡，但默认建议明确指定单卡，例如 0
    # shellcheck disable=SC1091
    source "${DRIVESTUDIO_ROOT}/scripts/utils.sh"
    GPU_ID="$(pick_gpu)"
    if [[ -z "${GPU_ID}" ]]; then
        echo "错误: 未找到可用 GPU" >&2
        exit 1
    fi
fi

has_png_outputs() {
    local run_dir="$1"
    local rgb_dir=""
    local gt_dir=""
    local expected_count=""
    local rgb_count=""
    local gt_count=""

    if [[ ! -d "${run_dir}/videos_eval" ]]; then
        return 1
    fi

    rgb_dir="$(find "${run_dir}/videos_eval" -maxdepth 1 -type d -name 'full_set_*_rgbs' | sort | tail -n 1 || true)"
    gt_dir="$(find "${run_dir}/videos_eval" -maxdepth 1 -type d -name 'full_set_*_gt_rgbs' | sort | tail -n 1 || true)"

    if [[ -z "${rgb_dir}" || -z "${gt_dir}" ]]; then
        return 1
    fi

    expected_count="$(get_expected_full_png_count "${run_dir}")"
    if [[ -z "${expected_count}" ]]; then
        echo "[warn] $(basename "${run_dir}"): 无法确定 full_set 期望图片数，按未完成处理"
        return 1
    fi

    rgb_count="$(find "${rgb_dir}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
    gt_count="$(find "${gt_dir}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"

    if [[ "${rgb_count}" == "${expected_count}" && "${gt_count}" == "${expected_count}" ]]; then
        return 0
    fi

    echo "[redo] $(basename "${run_dir}"): 发现未完成结果，expected=${expected_count}, rgbs=${rgb_count}, gt_rgbs=${gt_count}"
    return 1
}

get_expected_full_png_count() {
    local run_dir="$1"

    (
        cd "${DRIVESTUDIO_ROOT}"
        export PYTHONPATH="${DRIVESTUDIO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        python - "${run_dir}" <<'PY'
import os
import sys

run_dir = os.path.abspath(sys.argv[1])

try:
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
except Exception:
    print("")
    raise SystemExit(0)

config_path = os.path.join(run_dir, "config.yaml")
if not os.path.isfile(config_path):
    print("")
    raise SystemExit(0)

try:
    cfg = OmegaConf.load(config_path)
    dataset = DrivingDataset(data_cfg=cfg.data)
    print(len(dataset.full_image_set))
except Exception:
    print("")
PY
    )
}

run_eval_with_images() {
    local ckpt_path="$1"

    (
        cd "${DRIVESTUDIO_ROOT}"
        export PYTHONPATH="${DRIVESTUDIO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python - "${ckpt_path}" "${EXTRA_EVAL_ARGS[@]}" <<'PY'
import os
import runpy
import sys

ckpt_path = os.path.abspath(sys.argv[1])
extra_eval_args = sys.argv[2:]
drivestudio_root = os.getcwd()

sys.path.insert(0, drivestudio_root)
import models.video_utils as video_utils

original_save_videos = video_utils.save_videos

def save_videos_force_images(*args, **kwargs):
    kwargs["save_images"] = True
    return original_save_videos(*args, **kwargs)

video_utils.save_videos = save_videos_force_images

sys.argv = [
    "tools/eval.py",
    "--resume_from",
    ckpt_path,
    "logging.save_seperate_video=True",
    *extra_eval_args,
]

runpy.run_path(os.path.join(drivestudio_root, "tools/eval.py"), run_name="__main__")
PY
    )
}

mapfile -t ALL_CKPTS < <(find "${SEARCH_ROOT}" -type f -name 'checkpoint_final.pth' | sort)

TARGET_CKPTS=()
for ckpt_path in "${ALL_CKPTS[@]}"; do
    run_dir="$(dirname "${ckpt_path}")"
    run_name="$(basename "${run_dir}")"
    if [[ "${run_name}" == ${RUN_FILTER} ]]; then
        TARGET_CKPTS+=("${ckpt_path}")
    fi
done

if (( ${#TARGET_CKPTS[@]} == 0 )); then
    echo "错误: 在 ${SEARCH_ROOT} 下没有找到匹配 ${RUN_FILTER} 的 checkpoint_final.pth" >&2
    exit 1
fi

echo "Drivestudio root : ${DRIVESTUDIO_ROOT}"
echo "Search root      : ${SEARCH_ROOT}"
echo "Run filter       : ${RUN_FILTER}"
echo "GPU              : ${GPU_ID}"
echo "Scenes           : ${#TARGET_CKPTS[@]}"

for ckpt_path in "${TARGET_CKPTS[@]}"; do
    run_dir="$(dirname "${ckpt_path}")"
    run_name="$(basename "${run_dir}")"

    if [[ "${FORCE_EVAL}" != "1" ]] && has_png_outputs "${run_dir}"; then
        echo "[skip] ${run_name}: 已存在 full_set_*_{rgbs,gt_rgbs} PNG"
        continue
    fi

    echo "[eval] ${run_name}"
    echo "       ckpt: ${ckpt_path}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    run_eval_with_images "${ckpt_path}"
done

echo "完成。"
