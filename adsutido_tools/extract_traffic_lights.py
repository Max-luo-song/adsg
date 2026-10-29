#!/usr/bin/env python3
"""
使用 Grounding DINO 检测红绿灯，通过 LiDAR 深度反投影到 3D 世界坐标，
生成与 instances_info.json 兼容的伪标签。

输出（默认独立保存，使用 --merge 则合并写入已有文件）：
  {scene_dir}/instances/traffic_light_instances_info.json
  {scene_dir}/instances/traffic_light_frame_instances.json
  （--merge 时覆写 instances_info.json / frame_instances.json）
  {scene_dir}/dynamic_masks/traffic_light/{frame:03d}_{cam}.png  （--save_masks 时）

运行示例：
  python datasets/tools/extract_traffic_lights.py --scene_dir data/waymo/processed/training/621 --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --gdino_checkpoint ckpt/grounding_dino/groundingdino_swint_ogc.pth --cam_ids 0 1 2 --save_masks --device cuda
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 坐标系约定（与 waymo_sourceloader.py 保持一致）
#
# Waymo camera frame : x 前（光轴方向），y 左，z 上
# OpenCV camera frame: x 右，y 下，z 前（深度方向）
#
# extrinsics/{cam_id}.txt 存储的是 Waymo camera → vehicle 的 4×4 矩阵。
# sourceloader 在加载时执行  cam_to_ego = cam_to_ego @ OPENCV2DATASET
# 使其变为 OpenCV camera → vehicle 的矩阵，再用于渲染。
#
# OPENCV2DATASET : OpenCV frame → Waymo dataset frame
#   x_w =  z_o  (前向深度 → 相机光轴)
#   y_w = -x_o  (右 → 左)
#   z_w = -y_o  (下 → 上)
#
# DATASET2OPENCV = OPENCV2DATASET.T（纯旋转矩阵，转置即逆）
# ──────────────────────────────────────────────────────────────────────────────
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)
DATASET2OPENCV = OPENCV2DATASET.T  # inv of pure-rotation homogeneous matrix

# 红绿灯标准近似尺寸 [length(纵深), width(横), height(高)]，单位：米
TRAFFIC_LIGHT_BOX_SIZE = [0.5, 0.5, 1.2]


# ══════════════════════════════════════════════════════════════════════════════
# I/O 工具
# ══════════════════════════════════════════════════════════════════════════════

def load_intrinsics(scene_dir: str, cam_id: int) -> np.ndarray:
    """读取 1D 内参 [fu, fv, cu, cv, k1, k2, p1, p2, k3]（9 个元素）。"""
    return np.loadtxt(os.path.join(scene_dir, "intrinsics", f"{cam_id}.txt"))


def load_extrinsics_raw(
    scene_dir: str,
    cam_id: int,
    frame_idx: Optional[int] = None,
    dataset_type: str = "waymo",
) -> np.ndarray:
    """
    读取 4×4 extrinsics（dataset camera frame → vehicle/ego frame）。
    注意：这是"原始"矩阵，尚未乘 OPENCV2DATASET。

    支持两种常见存储格式：
    1) 静态外参（Waymo/KITTI 常见）: extrinsics/{cam_id}.txt
    2) 按帧外参（NuScenes 常见）:  extrinsics/{frame_idx:03d}_{cam_id}.txt
    """
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")
    candidates = []

    if frame_idx is not None:
        candidates.append(os.path.join(extrinsics_dir, f"{frame_idx:03d}_{cam_id}.txt"))

    # 兼容静态外参格式
    candidates.append(os.path.join(extrinsics_dir, f"{cam_id}.txt"))

    for path in candidates:
        if os.path.exists(path):
            return np.loadtxt(path)

    raise FileNotFoundError(
        f"Cannot find extrinsics for cam_id={cam_id}, frame_idx={frame_idx}, "
        f"dataset_type={dataset_type}. Tried: {candidates}"
    )


def _resolve_pose_dir(scene_dir: str) -> str:
    """
    解析场景位姿目录。
    优先使用 ego_pose（Waymo/KITTI 常见），若不存在则回退到 lidar_pose（NuScenes 常见）。
    """
    candidates = ["ego_pose", "lidar_pose"]
    for name in candidates:
        pose_dir = os.path.join(scene_dir, name)
        if os.path.isdir(pose_dir):
            return pose_dir
    raise FileNotFoundError(
        f"Cannot find pose directory under {scene_dir}. "
        f"Tried: {[os.path.join(scene_dir, c) for c in candidates]}"
    )


def load_ego_pose(scene_dir: str, frame_idx: int) -> np.ndarray:
    """读取 4×4 vehicle-to-world 矩阵（兼容 ego_pose/lidar_pose）。"""
    pose_dir = _resolve_pose_dir(scene_dir)
    pose_path = os.path.join(pose_dir, f"{frame_idx:03d}.txt")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    return np.loadtxt(pose_path)


def load_lidar_xyz_vehicle(scene_dir: str, frame_idx: int) -> np.ndarray:
    """
    读取 LiDAR 点的 3D 坐标（vehicle frame，float64）。
    Compatible with Waymo (14 dims) and KITTI (4 dims).

    Waymo bin 文件列布局（共 14 列）：
      0:3  origins      (ox, oy, oz)
      3:6  points       (x,  y,  z)   ← 实际空间坐标
      ...

    KITTI bin 文件列布局（共 4 列）：
      0:3  points       (x,  y,  z)
      3    intensity
    """
    path = os.path.join(scene_dir, "lidar", f"{frame_idx:03d}.bin")
    data = np.fromfile(path, dtype=np.float32)
    
    if data.size % 14 == 0:
        # Waymo format
        pc = data.reshape(-1, 14)
        return pc[:, 3:6].astype(np.float64)
    elif data.size % 4 == 0:
        # KITTI format
        pc = data.reshape(-1, 4)
        return pc[:, 0:3].astype(np.float64)
    else:
        raise ValueError(f"Unknown LiDAR data format at {path} with size {data.size}")


def count_frames(scene_dir: str) -> int:
    """统计位姿目录（ego_pose/lidar_pose）下的帧数。"""
    pose_dir = _resolve_pose_dir(scene_dir)
    return len([f for f in os.listdir(pose_dir) if f.endswith(".txt")])


# ══════════════════════════════════════════════════════════════════════════════
# 坐标投影 / 反投影
# ══════════════════════════════════════════════════════════════════════════════

def _cam_to_ego_ocv(cam_to_ego_raw: np.ndarray, dataset_type: str = "waymo") -> np.ndarray:
    """
    将原始 extrinsics（Waymo/KITTI cam → vehicle）转为
    OpenCV cam → vehicle，与 sourceloader 保持一致。
    """
    if dataset_type in ["kitti", "nuscenes"]:
        # KITTI / NuScenes extrinsics are already OpenCV -> Vehicle
        return cam_to_ego_raw
    else:
        # Waymo extrinsics are Waymo Camera -> Vehicle
        # Need to convert OpenCV -> Waymo Camera first
        return cam_to_ego_raw @ OPENCV2DATASET


def project_vehicle_to_image(
    xyz_vehicle: np.ndarray,   # (N, 3) vehicle frame
    intrinsics: np.ndarray,    # [fu, fv, cu, cv, ...]
    cam_to_ego_raw: np.ndarray,  # (4, 4) Waymo cam → vehicle (raw from file)
    dataset_type: str = "waymo",
) -> tuple:
    """
    将 vehicle frame 的 3D 点投影到图像平面。

    投影链：
      vehicle → OpenCV camera（via inv(cam_to_ego_ocv)）→ 像素

    Returns
    -------
    uv         : (N, 2) 像素坐标（可能超出图像边界）
    depth      : (N,)   OpenCV 相机 z 轴深度
    valid_mask : (N,)   bool，depth > 0 的有效点
    """
    fu, fv, cu, cv = intrinsics[:4]

    # OpenCV cam → vehicle
    c2e_ocv = _cam_to_ego_ocv(cam_to_ego_raw, dataset_type)
    # vehicle → OpenCV camera
    e2c_ocv = np.linalg.inv(c2e_ocv)  # (4, 4)

    ones = np.ones((len(xyz_vehicle), 1), dtype=np.float64)
    pts_hom = np.hstack([xyz_vehicle, ones])        # (N, 4)
    pts_cam = (e2c_ocv @ pts_hom.T).T[:, :3]        # (N, 3) OpenCV camera

    depth = pts_cam[:, 2]
    valid = depth > 0.1

    safe_depth = np.where(valid, depth, 1.0)
    u = fu * pts_cam[:, 0] / safe_depth + cu
    v = fv * pts_cam[:, 1] / safe_depth + cv

    return np.stack([u, v], axis=1), depth, valid


def backproject_to_vehicle(
    u: float,
    v: float,
    depth: float,              # OpenCV camera z-depth (forward)
    intrinsics: np.ndarray,    # [fu, fv, cu, cv, ...]
    cam_to_ego_raw: np.ndarray,  # (4, 4) Waymo cam → vehicle (raw from file)
    dataset_type: str = "waymo",
) -> np.ndarray:
    """
    从图像像素 (u, v) + 已知深度反投影到 vehicle frame。

    反投影链：
      像素 → OpenCV camera → vehicle（via cam_to_ego_ocv）
    """
    fu, fv, cu, cv = intrinsics[:4]

    # 还原 OpenCV camera 坐标
    x_cam = (u - cu) * depth / fu
    y_cam = (v - cv) * depth / fv
    z_cam = depth
    pt_cam_hom = np.array([x_cam, y_cam, z_cam, 1.0])

    # OpenCV camera → vehicle
    c2e_ocv = _cam_to_ego_ocv(cam_to_ego_raw, dataset_type)
    pt_vehicle = c2e_ocv @ pt_cam_hom
    return pt_vehicle[:3]


def estimate_lidar_depth(
    lidar_uv: np.ndarray,    # (N, 2) LiDAR 点在图像上的投影坐标
    lidar_depth: np.ndarray, # (N,)  对应深度
    lidar_valid: np.ndarray, # (N,)  bool
    bbox: list,              # [x1, y1, x2, y2] pixel
    img_h: int,
    img_w: int,
    margin: int = 10,        # 向外扩展 bbox 以找到更多 LiDAR 点
    top_fraction: float = 0.6,   # 只取 bbox 上方 top_fraction 区域（信号头所在位置）
    depth_percentile: float = 15.0,  # 取低百分位深度（交通灯在背景前方）
    min_depth: float = 2.0,       # 有效深度下限（米）
    max_depth: float = 120.0,     # 有效深度上限（米）
) -> Optional[float]:
    """
    在 bbox 上部区域内取 LiDAR 点深度的低百分位值。

    改进策略：
    1. 只取 bbox 上方 top_fraction 区域（信号灯头部位置，避免路面/杆子干扰）。
    2. 使用低百分位（而非中值）——交通灯是前景物体，应取最近有效深度。
    3. 限制深度范围，过滤地面/天空噪声。
    4. 若上部区域无点，回退到完整 bbox 区域再试一次。
    """
    x1, y1, x2, y2 = bbox
    bh = y2 - y1

    # ── 策略1：优先在 bbox 上部区域搜索 ──────────────────────────────────────
    y2_top = int(y1 + bh * top_fraction)
    x1e = max(0, x1 - margin)
    y1e = max(0, y1 - margin)
    x2e = min(img_w - 1, x2 + margin)
    y2e_top = min(img_h - 1, y2_top + margin)

    in_top = (
        lidar_valid
        & (lidar_uv[:, 0] >= x1e)
        & (lidar_uv[:, 0] <= x2e)
        & (lidar_uv[:, 1] >= y1e)
        & (lidar_uv[:, 1] <= y2e_top)
        & (lidar_depth >= min_depth)
        & (lidar_depth <= max_depth)
    )
    if in_top.sum() > 0:
        return float(np.percentile(lidar_depth[in_top], depth_percentile))

    # ── 策略2：回退到完整 bbox（加深度范围过滤）──────────────────────────────
    x2e_full = min(img_w - 1, x2 + margin)
    y2e_full = min(img_h - 1, y2 + margin)
    in_full = (
        lidar_valid
        & (lidar_uv[:, 0] >= x1e)
        & (lidar_uv[:, 0] <= x2e_full)
        & (lidar_uv[:, 1] >= y1e)
        & (lidar_uv[:, 1] <= y2e_full)
        & (lidar_depth >= min_depth)
        & (lidar_depth <= max_depth)
    )
    if in_full.sum() == 0:
        return None
    return float(np.percentile(lidar_depth[in_full], depth_percentile))


# ══════════════════════════════════════════════════════════════════════════════
# 多视角三角化
# ══════════════════════════════════════════════════════════════════════════════

def compute_ray_world(
    u: float,
    v: float,
    intrinsics: np.ndarray,      # [fu, fv, cu, cv, ...]
    cam_to_ego_raw: np.ndarray,  # (4, 4) Waymo cam → vehicle (raw from file)
    ego_pose: np.ndarray,        # (4, 4) vehicle → world
    dataset_type: str = "waymo",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定像素坐标 (u, v)，计算该像素对应的世界坐标系光线。

    Returns
    -------
    origin_world : (3,)  相机光心在世界坐标系中的位置
    dir_world    : (3,)  归一化光线方向（世界坐标系）
    """
    fu, fv, cu, cv = intrinsics[:4]
    c2e_ocv = _cam_to_ego_ocv(cam_to_ego_raw, dataset_type)

    # 光线在 OpenCV 相机系的方向（未归一化）
    dir_cam = np.array([(u - cu) / fu, (v - cv) / fv, 1.0])
    dir_cam /= np.linalg.norm(dir_cam)

    # 相机光心在 vehicle 坐标系中的位置（外参的平移部分）
    origin_veh = c2e_ocv[:3, 3]
    # 光线方向转到 vehicle 坐标系
    dir_veh = c2e_ocv[:3, :3] @ dir_cam

    # 转到世界坐标系
    origin_world = (ego_pose @ np.append(origin_veh, 1.0))[:3]
    dir_world = ego_pose[:3, :3] @ dir_veh
    dir_world /= np.linalg.norm(dir_world)

    return origin_world, dir_world


def triangulate_rays(
    rays_o: np.ndarray,  # (N, 3) 光线起点（世界坐标系）
    rays_d: np.ndarray,  # (N, 3) 归一化光线方向
) -> Tuple[Optional[np.ndarray], float]:
    """
    最小二乘多视角三角化：找到与所有光线距离平方和最小的 3D 点。

    对每条光线 i，点 P 到光线的距离平方为：
        ||(I - d_i d_i^T)(P - o_i)||^2

    法方程：
        A P = b，其中
        A = sum_i (I - d_i d_i^T)
        b = sum_i (I - d_i d_i^T) o_i

    Returns
    -------
    xyz        : (3,) 三角化的 3D 点；若退化则返回 None
    cond_ratio : float  矩阵 A 的最小/最大奇异值比（越高越可靠，<1e-3 视为退化）
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(rays_o, rays_d):
        d = d / (np.linalg.norm(d) + 1e-12)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ o

    sv = np.linalg.svd(A, compute_uv=False)
    cond_ratio = float(sv[-1] / (sv[0] + 1e-12))

    if cond_ratio < 1e-6:
        # 所有光线近乎完全平行，三角化完全退化
        return None, cond_ratio

    xyz = np.linalg.lstsq(A, b, rcond=None)[0]
    return xyz, cond_ratio


# ══════════════════════════════════════════════════════════════════════════════
# Grounding DINO 检测
# ══════════════════════════════════════════════════════════════════════════════

def build_gdino_model(config_path: str, checkpoint_path: str, device: str):
    """加载 Grounding DINO 模型。"""
    from groundingdino.util.inference import load_model  # lazy import
    return load_model(config_path, checkpoint_path, device=device)


def detect_traffic_lights(
    model,
    image_path: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    device: str = "cuda",
) -> tuple:
    """
    使用 Grounding DINO 检测图像中的红绿灯。

    Returns
    -------
    boxes_pixel  : list of [x1, y1, x2, y2]（整数像素坐标）
    scores       : list of float（置信度）
    image_source : np.ndarray (H, W, 3) RGB，原始图像
    """
    from groundingdino.util.inference import load_image, predict  # lazy import

    image_source, image_tensor = load_image(image_path)
    boxes_norm, logits, _ = predict(
        model=model,
        image=image_tensor,
        caption="traffic light",
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    h, w = image_source.shape[:2]
    boxes_pixel = []
    for box in boxes_norm.cpu().numpy():
        cx, cy, bw, bh = box
        x1 = int(max(0, (cx - bw / 2) * w))
        y1 = int(max(0, (cy - bh / 2) * h))
        x2 = int(min(w - 1, (cx + bw / 2) * w))
        y2 = int(min(h - 1, (cy + bh / 2) * h))
        boxes_pixel.append([x1, y1, x2, y2])

    return boxes_pixel, logits.cpu().numpy().tolist(), image_source


# ══════════════════════════════════════════════════════════════════════════════
# 跨帧聚类
# ══════════════════════════════════════════════════════════════════════════════

def cluster_detections(
    all_dets: list,
    eps: float = 2.0,
    min_samples: int = 2,
) -> dict:
    """
    对所有帧/相机的 3D 检测结果做 DBSCAN 聚类，识别唯一红绿灯实例。

    Parameters
    ----------
    all_dets    : [(frame_idx, cam_id, xyz_world(3,), bbox, score), ...]
    eps         : DBSCAN 邻域半径（米）
    min_samples : 形成簇的最小样本数（过滤偶发误检）

    Returns
    -------
    clusters : {cluster_id: [detection, ...]}（不含噪声点 label=-1）
    """
    if not all_dets:
        return {}

    xyz_world = np.array([d[2] for d in all_dets])  # (M, 3)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz_world)

    clusters: dict = {}
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters.setdefault(int(lbl), []).append(all_dets[i])
    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# 可视化（调试用）
# ══════════════════════════════════════════════════════════════════════════════

def visualize_detections(image_source: np.ndarray, boxes: list, scores: list) -> np.ndarray:
    """在图像上绘制检测框，返回 BGR 格式 numpy 数组。"""
    vis = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    for bbox, score in zip(boxes, scores):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, f"{score:.2f}", (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Grounding DINO 红绿灯 3D 伪标签生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 路径
    parser.add_argument(
        "--scene_dir", required=True,
        help="已处理的 Waymo 场景目录，例如 data/waymo/processed/training/621",
    )
    parser.add_argument(
        "--gdino_config", required=True,
        help="Grounding DINO 配置文件路径",
    )
    parser.add_argument(
        "--gdino_checkpoint", required=True,
        help="Grounding DINO 权重文件路径",
    )
    # 检测参数
    parser.add_argument("--box_threshold", type=float, default=0.30,
                        help="Grounding DINO box 置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Grounding DINO text 置信度阈值")
    parser.add_argument("--device", default="cuda",
                        help="推理设备：cuda / cpu")
    parser.add_argument("--cam_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="参与检测的相机 ID（0=前，1=前左，2=前右，3=左，4=右）")
    # 深度参数
    parser.add_argument("--lidar_margin", type=int, default=10,
                        help="向外扩大 bbox 以搜索 LiDAR 点的像素数")
    parser.add_argument("--lidar_top_fraction", type=float, default=0.6,
                        help="仅使用 bbox 上方该比例区域搜索 LiDAR 深度（信号头位置）")
    parser.add_argument("--lidar_depth_percentile", type=float, default=15.0,
                        help="LiDAR 深度取该百分位值（低百分位=前景优先）")
    parser.add_argument("--lidar_min_depth", type=float, default=2.0,
                        help="LiDAR 有效深度下限（米）")
    parser.add_argument("--lidar_max_depth", type=float, default=120.0,
                        help="LiDAR 有效深度上限（米）")
    parser.add_argument("--default_depth", type=float, default=20.0,
                        help="bbox 内无 LiDAR 点时使用的后备深度（米），仅用于 DBSCAN 初始聚类")
    parser.add_argument("--bbox_head_fraction", type=float, default=0.5,
                        help="用 bbox 顶部该比例处的像素进行反投影（0=顶边,0.5=中心）。"
                             "对紧凑型检测框（仅信号头）用 0.5；含长杆的检测框可调小至 0.3~0.4")
    # 聚类参数
    parser.add_argument("--cluster_eps", type=float, default=2.0,
                        help="DBSCAN 邻域半径（米）；建议 1.5~3.0")
    parser.add_argument("--cluster_min_samples", type=int, default=2,
                        help="DBSCAN 形成簇的最小样本数；设 1 则保留所有单帧检测")
    parser.add_argument("--cluster_merge_eps", type=float, default=1.5,
                        help="三角化后二次 DBSCAN 合并邻近重复簇的半径（米）；设 0 禁用")
    # 三角化参数
    parser.add_argument("--triag_min_rays", type=int, default=3,
                        help="进行三角化所需的最少光线数（不足则回退到均值）")
    parser.add_argument("--triag_cond_thresh", type=float, default=1e-4,
                        help="三角化矩阵最小/最大奇异值比阈值；低于此值视为退化（增大以要求更宽基线）")
    # 后处理过滤参数
    parser.add_argument("--min_tl_frames", type=int, default=3,
                        help="实例最少出现帧数，不足则丢弃（过滤偶发误检）")
    parser.add_argument("--min_tl_rel_z", type=float, default=0,
                        help="实例 3D 位置相对自车 z 的最低高度（米），低于此值视为误检丢弃；设 0 禁用")
    # 输出控制
    parser.add_argument("--save_masks", action="store_true",
                        help="将 2D 检测框保存为 binary mask（dynamic_masks/traffic_light/）")
    parser.add_argument("--save_vis", action="store_true",
                        help="保存标注可视化图到 vis_traffic_light/")
    parser.add_argument("--merge", action="store_true",
                        help="将红绿灯实例合并写入已有 instances_info.json / frame_instances.json")
    parser.add_argument("--dataset_type", default="waymo", choices=["waymo", "kitti", "nuscenes"],
                        help="数据集类型，影响坐标系转换和数据读取")

    args = parser.parse_args()
    scene_dir = args.scene_dir

    assert os.path.isdir(scene_dir), f"scene_dir 不存在: {scene_dir}"
    num_frames = count_frames(scene_dir)
    print(f"[INFO] 场景: {scene_dir}")
    print(f"[INFO] 帧数: {num_frames}")
    print(f"[INFO] 处理相机: {args.cam_ids}")

    # ── 加载 Grounding DINO ───────────────────────────────────────────────────
    print("[INFO] 加载 Grounding DINO 模型...")
    model = build_gdino_model(args.gdino_config, args.gdino_checkpoint, args.device)

    # ── 遍历所有帧 × 相机 ────────────────────────────────────────────────────
    # all_dets : [(frame_idx, cam_id, xyz_world(3,), bbox, score), ...]
    all_dets: list = []
    # frame_dets[frame_idx][cam_id] = [{"bbox":…, "score":…, "xyz_world":…}, …]
    frame_dets: dict = {}

    for cam_id in args.cam_ids:
        intrinsics = load_intrinsics(scene_dir, cam_id)

        for frame_idx in tqdm(range(num_frames), desc=f"相机 {cam_id}", leave=True):
            cam_to_ego_raw = load_extrinsics_raw(
                scene_dir, cam_id, frame_idx=frame_idx, dataset_type=args.dataset_type
            )
            image_path = os.path.join(
                scene_dir, "images", f"{frame_idx:03d}_{cam_id}.jpg"
            )
            if not os.path.exists(image_path):
                continue

            # Grounding DINO 检测
            boxes_pixel, scores, image_source = detect_traffic_lights(
                model, image_path,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )

            img_h, img_w = image_source.shape[:2]

            # 可视化（可选）
            if args.save_vis and boxes_pixel:
                vis_dir = os.path.join(scene_dir, "vis_traffic_light")
                os.makedirs(vis_dir, exist_ok=True)
                vis = visualize_detections(image_source, boxes_pixel, scores)
                cv2.imwrite(
                    os.path.join(vis_dir, f"{frame_idx:03d}_{cam_id}.jpg"), vis
                )

            if not boxes_pixel:
                continue

            # 加载 LiDAR 并投影到图像，用于深度估计
            lidar_xyz = load_lidar_xyz_vehicle(scene_dir, frame_idx)
            if len(lidar_xyz) > 0:
                lidar_uv, lidar_depth, lidar_valid = project_vehicle_to_image(
                    lidar_xyz, intrinsics, cam_to_ego_raw, dataset_type=args.dataset_type
                )
            else:
                lidar_uv = np.zeros((0, 2))

                lidar_depth = np.zeros(0)
                lidar_valid = np.zeros(0, dtype=bool)

            ego_pose = load_ego_pose(scene_dir, frame_idx)  # vehicle → world

            frame_dets.setdefault(frame_idx, {})[cam_id] = []

            for bbox, score in zip(boxes_pixel, scores):
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                # 用 bbox 顶部 bbox_head_fraction 处的像素对准信号灯头部
                # 注意：图像坐标 y 向下增大，y1 是 bbox 上边（信号头）
                cy = y1 + (y2 - y1) * args.bbox_head_fraction

                depth = estimate_lidar_depth(
                    lidar_uv, lidar_depth, lidar_valid,
                    bbox, img_h, img_w,
                    margin=args.lidar_margin,
                    top_fraction=args.lidar_top_fraction,
                    depth_percentile=args.lidar_depth_percentile,
                    min_depth=args.lidar_min_depth,
                    max_depth=args.lidar_max_depth,
                )
                if depth is None:
                    depth = args.default_depth

                # vehicle frame
                xyz_veh = backproject_to_vehicle(
                    cx, cy, depth, intrinsics, cam_to_ego_raw,
                    dataset_type=args.dataset_type
                )
                # world frame
                xyz_world = (ego_pose @ np.append(xyz_veh, 1.0))[:3]

                # 计算该检测的观测光线（用于后续多视角三角化）
                ray_o, ray_d = compute_ray_world(
                    cx, cy, intrinsics, cam_to_ego_raw, ego_pose,
                    dataset_type=args.dataset_type
                )

                frame_dets[frame_idx][cam_id].append({
                    "bbox": bbox,
                    "score": float(score),
                    "xyz_world": xyz_world,
                })
                all_dets.append(
                    (frame_idx, cam_id, xyz_world, bbox, float(score), ray_o, ray_d)
                )

    print(f"\n[INFO] 累计 2D 检测数: {len(all_dets)}")

    # ── DBSCAN 聚类，识别唯一红绿灯 ──────────────────────────────────────────
    clusters = cluster_detections(
        all_dets,
        eps=args.cluster_eps,
        min_samples=args.cluster_min_samples,
    )
    print(f"[INFO] 聚类得到 {len(clusters)} 个唯一红绿灯实例")

    # ── 读取已有标注（确定新 ID 起点）────────────────────────────────────────
    inst_path = os.path.join(scene_dir, "instances", "instances_info.json")
    fi_path = os.path.join(scene_dir, "instances", "frame_instances.json")

    existing_inst: dict = {}
    existing_fi: dict = {}
    if os.path.exists(inst_path):
        with open(inst_path) as f:
            existing_inst = json.load(f)
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            existing_fi = json.load(f)

    # 新实例 ID 从现有最大 ID+1 开始（保证不冲突）
    existing_ids = [int(k) for k in existing_inst.keys()] if existing_inst else [-1]
    start_id = max(existing_ids) + 1

    # ── 构建红绿灯 instances_info / frame_instances ───────────────────────────
    tl_inst: dict = {}
    tl_fi: dict = {str(i): [] for i in range(num_frames)}

    # 第一阶段：每个簇独立三角化，收集结果
    cluster_xyz_final: dict = {}
    cluster_dets_map: dict = {}

    for cluster_id, dets in clusters.items():
        # ── 多视角三角化，精确确定 3D 位置 ──────────────────────────────────
        # 每条 detection 携带一条观测光线（d[5]=ray_o, d[6]=ray_d）
        rays_o = np.array([d[5] for d in dets])
        rays_d = np.array([d[6] for d in dets])

        xyz_lidar_mean = np.mean([d[2] for d in dets], axis=0)

        if len(dets) >= args.triag_min_rays:
            xyz_triag, cond = triangulate_rays(rays_o, rays_d)
            if xyz_triag is not None and cond >= args.triag_cond_thresh:
                # 合理性检查：三角化结果到各光线起点的平均距离须在合理范围内
                dists_to_cams = np.linalg.norm(rays_o - xyz_triag[None, :], axis=1)
                dist_ok = (dists_to_cams.min() > 1.0) and (dists_to_cams.max() < 200.0)
                if dist_ok:
                    xyz_mean = xyz_triag
                    print(f"  [TRIAG] cluster {cluster_id}: cond={cond:.4f}, "
                          f"xyz={xyz_triag.round(2)}, "
                          f"dist_range=[{dists_to_cams.min():.1f}, {dists_to_cams.max():.1f}]m")
                else:
                    xyz_mean = xyz_lidar_mean
                    print(f"  [WARN ] cluster {cluster_id}: triangulation distance out of range "
                          f"({dists_to_cams.min():.1f}~{dists_to_cams.max():.1f}m), using lidar mean")
            else:
                xyz_mean = xyz_lidar_mean
                print(f"  [WARN ] cluster {cluster_id}: triangulation degenerate "
                      f"(cond={cond:.6f}), using lidar mean")
        else:
            xyz_mean = xyz_lidar_mean

        cluster_xyz_final[cluster_id] = xyz_mean
        cluster_dets_map[cluster_id] = dets

    # ── 二次 DBSCAN：对三角化后坐标再合并距离过近的重复簇 ─────────────────
    if args.cluster_merge_eps > 0 and len(cluster_xyz_final) >= 2:
        cids_sorted = sorted(cluster_xyz_final.keys())
        pts_final = np.array([cluster_xyz_final[cid] for cid in cids_sorted])
        merge_labels = DBSCAN(
            eps=args.cluster_merge_eps, min_samples=1
        ).fit_predict(pts_final)

        merged_groups: dict = {}
        for i, ml in enumerate(merge_labels):
            merged_groups.setdefault(int(ml), []).append(cids_sorted[i])

        merged_cluster_xyz: dict = {}
        merged_cluster_dets: dict = {}
        for ml, cids in sorted(merged_groups.items()):
            rep_id = min(cids)  # 用最小 cluster_id 作为代表
            if len(cids) == 1:
                merged_cluster_xyz[rep_id] = cluster_xyz_final[cids[0]]
                merged_cluster_dets[rep_id] = cluster_dets_map[cids[0]]
            else:
                print(f"  [MERGE] clusters {cids} → merged under {rep_id}")
                merged_dets = []
                for cid in cids:
                    merged_dets.extend(cluster_dets_map[cid])
                # 重新三角化合并后的簇
                m_rays_o = np.array([d[5] for d in merged_dets])
                m_rays_d = np.array([d[6] for d in merged_dets])
                if len(merged_dets) >= args.triag_min_rays:
                    xyz_t, cond = triangulate_rays(m_rays_o, m_rays_d)
                    if xyz_t is not None and cond >= args.triag_cond_thresh:
                        dists = np.linalg.norm(m_rays_o - xyz_t[None, :], axis=1)
                        if dists.min() > 1.0 and dists.max() < 200.0:
                            xyz_rep = xyz_t
                        else:
                            xyz_rep = np.mean([d[2] for d in merged_dets], axis=0)
                    else:
                        xyz_rep = np.mean([d[2] for d in merged_dets], axis=0)
                else:
                    xyz_rep = np.mean([d[2] for d in merged_dets], axis=0)
                merged_cluster_xyz[rep_id] = xyz_rep
                merged_cluster_dets[rep_id] = merged_dets
    else:
        merged_cluster_xyz = cluster_xyz_final
        merged_cluster_dets = cluster_dets_map

    # ── 构建实例，应用后处理过滤 ─────────────────────────────────────────────
    inst_counter = 0
    for cluster_id in sorted(merged_cluster_xyz.keys()):
        xyz_mean = merged_cluster_xyz[cluster_id]
        dets = merged_cluster_dets[cluster_id]

        visible_frames = sorted(set(d[0] for d in dets))

        # 过滤：帧数不足
        if len(visible_frames) < args.min_tl_frames:
            print(f"  [SKIP ] cluster {cluster_id}: only {len(visible_frames)} frames "
                  f"< min_tl_frames={args.min_tl_frames}")
            continue

        # 过滤：相对高度过低（误检或深度严重错误）
        if args.min_tl_rel_z > 0:
            first_frame = visible_frames[0]
            ego_first = load_ego_pose(scene_dir, first_frame)
            ego_z = float(ego_first[2, 3])
            rel_z = float(xyz_mean[2]) - ego_z
            if rel_z < args.min_tl_rel_z:
                print(f"  [SKIP ] cluster {cluster_id}: rel_z={rel_z:.2f}m "
                      f"< min_tl_rel_z={args.min_tl_rel_z}m")
                continue

        inst_id = start_id + inst_counter
        inst_counter += 1

        tx, ty, tz = float(xyz_mean[0]), float(xyz_mean[1]), float(xyz_mean[2])

        # obj_to_world：单位旋转 + 固定平移（红绿灯静止不动）
        obj_to_world = [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]

        tl_inst[str(inst_id)] = {
            "id": f"traffic_light_{inst_counter - 1}",
            "class_name": "TrafficLight",
            "frame_annotations": {
                "frame_idx": visible_frames,
                "obj_to_world": [obj_to_world] * len(visible_frames),
                "box_size": [TRAFFIC_LIGHT_BOX_SIZE] * len(visible_frames),
            },
        }
        for fi in visible_frames:
            tl_fi[str(fi)].append(inst_id)

    # ── 保存 binary mask（可选）──────────────────────────────────────────────
    if args.save_masks:
        mask_dir = os.path.join(scene_dir, "dynamic_masks", "traffic_light")
        os.makedirs(mask_dir, exist_ok=True)
        print(f"[INFO] 保存 mask 到 {mask_dir} ...")

        # 确定使用哪个 dynamic_masks 根目录（与 pixel_source.py 逻辑保持一致）
        fine_mask_root = os.path.join(scene_dir, "fine_dynamic_masks")
        dyn_mask_root = fine_mask_root if os.path.exists(fine_mask_root) \
            else os.path.join(scene_dir, "dynamic_masks")
        all_mask_dir = os.path.join(dyn_mask_root, "all")
        os.makedirs(all_mask_dir, exist_ok=True)

        for frame_idx in range(num_frames):
            for cam_id in args.cam_ids:
                img_path = os.path.join(
                    scene_dir, "images", f"{frame_idx:03d}_{cam_id}.jpg"
                )
                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

                if frame_idx in frame_dets and cam_id in frame_dets[frame_idx]:
                    for det in frame_dets[frame_idx][cam_id]:
                        x1, y1, x2, y2 = det["bbox"]
                        mask[
                            max(0, y1):min(h, y2),
                            max(0, x1):min(w, x2),
                        ] = 255

                cv2.imwrite(
                    os.path.join(mask_dir, f"{frame_idx:03d}_{cam_id}.png"), mask
                )

                # 将 TL mask OR 合并进 all/ 目录，使背景训练时排除红绿灯像素
                all_mask_path = os.path.join(all_mask_dir, f"{frame_idx:03d}_{cam_id}.png")
                if os.path.exists(all_mask_path):
                    existing_all = cv2.imread(all_mask_path, cv2.IMREAD_GRAYSCALE)
                    if existing_all is not None and existing_all.shape == mask.shape:
                        merged = cv2.bitwise_or(existing_all, mask)
                    else:
                        merged = mask
                else:
                    merged = mask
                cv2.imwrite(all_mask_path, merged)

        print(f"[INFO] TL mask 已 OR 合并进 {all_mask_dir}")

    # ── 保存 / 合并 JSON ──────────────────────────────────────────────────────
    os.makedirs(os.path.join(scene_dir, "instances"), exist_ok=True)

    if args.merge:
        # 合并写入已有文件
        merged_inst = {**existing_inst, **tl_inst}

        # 合并 frame_instances：保留已有条目，追加新红绿灯 ID
        all_frame_keys = set(existing_fi.keys()) | set(tl_fi.keys())
        merged_fi: dict = {}
        for k in sorted(all_frame_keys, key=lambda x: int(x)):
            merged_fi[k] = list(existing_fi.get(k, [])) + tl_fi.get(k, [])

        with open(inst_path, "w") as f:
            json.dump(merged_inst, f, indent=4)
        with open(fi_path, "w") as f:
            json.dump(merged_fi, f, indent=4)

        print(f"[INFO] 已合并写入 {inst_path}")
        print(f"[INFO] 总实例数: {len(merged_inst)}（其中红绿灯 {len(tl_inst)} 个）")

    else:
        # 独立保存，不修改已有文件
        tl_inst_path = os.path.join(
            scene_dir, "instances", "traffic_light_instances_info.json"
        )
        tl_fi_path = os.path.join(
            scene_dir, "instances", "traffic_light_frame_instances.json"
        )
        with open(tl_inst_path, "w") as f:
            json.dump(tl_inst, f, indent=4)
        with open(tl_fi_path, "w") as f:
            json.dump(tl_fi, f, indent=4)

        print(f"[INFO] 红绿灯标注已保存（独立文件，共 {len(tl_inst)} 个实例）")
        print(f"       instances_info : {tl_inst_path}")
        print(f"       frame_instances: {tl_fi_path}")
        print(f"       验证无误后，用 --merge 合并到已有标注")


if __name__ == "__main__":
    main()
