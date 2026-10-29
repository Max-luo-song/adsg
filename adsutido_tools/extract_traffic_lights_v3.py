#!/usr/bin/env python3
"""
红绿灯检测与跟踪 v3 —— 综合版

综合 v2（2D跟踪 + 运动帧三角化）与 backup（LiDAR深度 + DBSCAN聚类）的优点：

核心策略：
1. Phase 1: 2D检测（Grounding DINO）+ bbox扩展
2. Phase 2: 2D跟踪（IoU关联）—— 来自 v2
3. Phase 3: 深度估计
   - 优先使用 LiDAR 深度（来自 backup）
   - 无 LiDAR 时用基线充足的三角化（保留静止帧信息，不仅限运动帧）
   - 最后回退到 bbox啊、 大小估计
4. Phase 4: 3D聚类合并（DBSCAN，来自 backup）
5. Phase 4.5: 反投影补全（将已知3D位置投影到所有帧的所有相机）
6. Phase 5: 生成输出

输出（与原版保持一致）：
  {scene_dir}/instances/traffic_light_instances_info.json
  {scene_dir}/instances/traffic_light_frame_instances.json
  （--merge 时覆写 instances_info.json / frame_instances.json）
  {scene_dir}/dynamic_masks/traffic_light/{frame:03d}_{cam}.png  （--save_masks 时）

Author: Copilot
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 坐标系约定（与 waymo_sourceloader.py 保持一致）
#
# Waymo camera frame : x 前（光轴方向），y 左，z 上
# OpenCV camera frame: x 右，y 下，z 前（深度方向）
#
# extrinsics/{cam_id}.txt 存储的是 Waymo camera → vehicle 的 4×4 矩阵。
# sourceloader 在加载时执行  cam_to_ego = cam_to_ego @ OPENCV2DATASET
# 使其变为 OpenCV camera → vehicle 的矩阵，再用于渲染。
# ══════════════════════════════════════════════════════════════════════════════
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)

# 红绿灯标准近似尺寸 [length(纵深), width(横), height(高)]，单位：米
TRAFFIC_LIGHT_BOX_SIZE = [0.8, 0.8, 3.0]

# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """单次检测"""
    frame_idx: int
    cam_id: int
    bbox: List[int]        # [x1, y1, x2, y2]
    score: float
    # 以下为可选，在需要时填充
    depth: Optional[float] = None
    xyz_world: Optional[np.ndarray] = None
    ray_o: Optional[np.ndarray] = None
    ray_d: Optional[np.ndarray] = None

@dataclass 
class Track:
    """一个红绿灯的跟踪轨迹"""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    xyz_world: Optional[np.ndarray] = None  # 最终确定的世界坐标
    
    def add_detection(self, det: Detection):
        self.detections.append(det)
    
    @property
    def frame_range(self) -> Tuple[int, int]:
        frames = [d.frame_idx for d in self.detections]
        return min(frames), max(frames)
    
    @property
    def num_frames(self) -> int:
        return len(set(d.frame_idx for d in self.detections))
    
    def get_detections_in_frame(self, frame_idx: int) -> List[Detection]:
        return [d for d in self.detections if d.frame_idx == frame_idx]

# ══════════════════════════════════════════════════════════════════════════════
# I/O 工具
# ══════════════════════════════════════════════════════════════════════════════

def load_intrinsics(scene_dir: str, cam_id: int) -> np.ndarray:
    """读取 1D 内参 [fu, fv, cu, cv, k1, k2, p1, p2, k3]（9 个元素）。"""
    return np.loadtxt(os.path.join(scene_dir, "intrinsics", f"{cam_id}.txt"))

def load_extrinsics(scene_dir: str, cam_id: int, frame_idx: int = 0) -> np.ndarray:
    """
    读取 4×4 extrinsics（camera frame → vehicle/ego frame）。
    兼容两种布局：
    1) 静态外参: extrinsics/{cam_id}.txt
    2) 按帧外参: extrinsics/{frame_idx:03d}_{cam_id}.txt
    """
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")
    candidates = [
        os.path.join(extrinsics_dir, f"{frame_idx:03d}_{cam_id}.txt"),
        os.path.join(extrinsics_dir, f"{cam_id}.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return np.loadtxt(path)
    raise FileNotFoundError(
        f"Cannot find extrinsics for cam_id={cam_id}, frame_idx={frame_idx}. Tried: {candidates}"
    )

def get_pose_dir(scene_dir: str) -> str:
    """
    选择位姿目录：
    1) 优先 ego_pose（Waymo/KITTI 等）
    2) 回退 lidar_pose（NuScenes 预处理常见）
    """
    ego_dir = os.path.join(scene_dir, "ego_pose")
    lidar_dir = os.path.join(scene_dir, "lidar_pose")
    if os.path.isdir(ego_dir):
        return ego_dir
    if os.path.isdir(lidar_dir):
        return lidar_dir
    raise FileNotFoundError(
        f"Neither 'ego_pose' nor 'lidar_pose' exists under scene_dir: {scene_dir}"
    )

def load_ego_pose(scene_dir: str, frame_idx: int) -> np.ndarray:
    """读取 4×4 vehicle-to-world 矩阵。"""
    pose_dir = get_pose_dir(scene_dir)
    return np.loadtxt(os.path.join(pose_dir, f"{frame_idx:03d}.txt"))

def load_lidar_xyz_vehicle(scene_dir: str, frame_idx: int) -> np.ndarray:
    """
    读取 LiDAR 点的 3D 坐标（vehicle frame，float64）。
    Compatible with Waymo (14 dims) and KITTI (4 dims).
    """
    path = os.path.join(scene_dir, "lidar", f"{frame_idx:03d}.bin")
    if not os.path.exists(path):
        return np.zeros((0, 3), dtype=np.float64)
    
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
        return np.zeros((0, 3), dtype=np.float64)

def count_frames(scene_dir: str) -> int:
    """统计位姿目录（ego_pose 或 lidar_pose）下的帧数。"""
    pose_dir = get_pose_dir(scene_dir)
    return len([f for f in os.listdir(pose_dir) if f.endswith(".txt")])

# ══════════════════════════════════════════════════════════════════════════════
# 坐标投影 / 反投影
# ══════════════════════════════════════════════════════════════════════════════

def cam_to_ego_ocv(cam_to_ego_raw: np.ndarray, dataset_type: str = "waymo") -> np.ndarray:
    """
    将原始 extrinsics 转为 OpenCV cam → vehicle。
    - waymo: 原始是 Waymo camera frame，需要做坐标系转换
    - nuscenes/kitti: 原始已是 OpenCV 风格，直接返回
    """
    if dataset_type in ["nuscenes", "kitti"]:
        return cam_to_ego_raw
    return cam_to_ego_raw @ OPENCV2DATASET

def project_vehicle_to_image(
    xyz_vehicle: np.ndarray,   # (N, 3) vehicle frame
    intrinsics: np.ndarray,    # [fu, fv, cu, cv, ...]
    cam_to_ego_raw: np.ndarray,  # (4, 4) Waymo cam → vehicle (raw from file)
    dataset_type: str = "waymo",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 vehicle frame 的 3D 点投影到图像平面。

    Returns
    -------
    uv         : (N, 2) 像素坐标
    depth      : (N,)   OpenCV 相机 z 轴深度
    valid_mask : (N,)   bool，depth > 0 的有效点
    """
    fu, fv, cu, cv = intrinsics[:4]

    c2e_ocv = cam_to_ego_ocv(cam_to_ego_raw, dataset_type)
    e2c_ocv = np.linalg.inv(c2e_ocv)

    ones = np.ones((len(xyz_vehicle), 1), dtype=np.float64)
    pts_hom = np.hstack([xyz_vehicle, ones])
    pts_cam = (e2c_ocv @ pts_hom.T).T[:, :3]

    depth = pts_cam[:, 2]
    valid = depth > 0.1

    safe_depth = np.where(valid, depth, 1.0)
    u = fu * pts_cam[:, 0] / safe_depth + cu
    v = fv * pts_cam[:, 1] / safe_depth + cv

    return np.stack([u, v], axis=1), depth, valid

def backproject_to_world(
    u: float, v: float, depth: float,
    intrinsics: np.ndarray,
    cam_to_ego_raw: np.ndarray,
    ego_pose: np.ndarray,
    dataset_type: str = "waymo",
) -> np.ndarray:
    """从图像像素+深度反投影到世界坐标"""
    fu, fv, cu, cv = intrinsics[:4]
    
    # 像素 → OpenCV camera
    x_cam = (u - cu) * depth / fu
    y_cam = (v - cv) * depth / fv
    z_cam = depth
    
    # OpenCV camera → vehicle
    c2e_ocv = cam_to_ego_ocv(cam_to_ego_raw, dataset_type)
    xyz_veh = (c2e_ocv @ np.array([x_cam, y_cam, z_cam, 1.0]))[:3]
    
    # vehicle → world
    xyz_world = (ego_pose @ np.append(xyz_veh, 1.0))[:3]
    
    return xyz_world

def compute_ray_world(
    u: float, v: float,
    intrinsics: np.ndarray,
    cam_to_ego_raw: np.ndarray,
    ego_pose: np.ndarray,
    dataset_type: str = "waymo",
) -> Tuple[np.ndarray, np.ndarray]:
    """计算从相机到像素的光线 (origin, direction) in world frame"""
    fu, fv, cu, cv = intrinsics[:4]
    
    c2e_ocv = cam_to_ego_ocv(cam_to_ego_raw, dataset_type)
    
    # 光线在 OpenCV camera frame 中的方向
    dir_cam = np.array([(u - cu) / fu, (v - cv) / fv, 1.0])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)
    
    # 相机光心在 vehicle 坐标系
    origin_veh = c2e_ocv[:3, 3]
    # 光线方向转到 vehicle 坐标系
    dir_veh = c2e_ocv[:3, :3] @ dir_cam
    
    # 转到世界坐标系
    origin_world = (ego_pose @ np.append(origin_veh, 1.0))[:3]
    dir_world = ego_pose[:3, :3] @ dir_veh
    dir_world = dir_world / np.linalg.norm(dir_world)
    
    return origin_world, dir_world

def project_world_to_image(
    xyz_world: np.ndarray,
    intrinsics: np.ndarray,
    cam_to_ego_raw: np.ndarray,
    ego_pose: np.ndarray,
    dataset_type: str = "waymo",
) -> Tuple[np.ndarray, float, bool]:
    """
    将世界坐标投影到图像平面。
    
    Returns: (uv, depth, valid)
    """
    fu, fv, cu, cv = intrinsics[:4]
    
    # world → vehicle
    world_to_ego = np.linalg.inv(ego_pose)
    xyz_veh = (world_to_ego @ np.append(xyz_world, 1.0))[:3]
    
    # vehicle → OpenCV camera
    c2e_ocv = cam_to_ego_ocv(cam_to_ego_raw, dataset_type)
    e2c_ocv = np.linalg.inv(c2e_ocv)
    xyz_cam = (e2c_ocv @ np.append(xyz_veh, 1.0))[:3]
    
    depth = xyz_cam[2]
    if depth <= 0.1:
        return np.array([0, 0]), depth, False
    
    u = fu * xyz_cam[0] / depth + cu
    v = fv * xyz_cam[1] / depth + cv
    
    return np.array([u, v]), depth, True

# ══════════════════════════════════════════════════════════════════════════════
# LiDAR 深度估计（来自 backup）
# ══════════════════════════════════════════════════════════════════════════════

def estimate_lidar_depth(
    lidar_uv: np.ndarray,
    lidar_depth: np.ndarray,
    lidar_valid: np.ndarray,
    bbox: List[int],
    img_h: int,
    img_w: int,
    margin: int = 10,
    top_fraction: float = 0.6,
    depth_percentile: float = 15.0,
    min_depth: float = 2.0,
    max_depth: float = 120.0,
) -> Optional[float]:
    """
    在 bbox 上部区域内取 LiDAR 点深度的低百分位值。

    改进策略：
    1. 只取 bbox 上方 top_fraction 区域（信号灯头部位置）
    2. 使用低百分位（交通灯是前景物体）
    3. 限制深度范围
    """
    x1, y1, x2, y2 = bbox
    bh = y2 - y1

    # 优先在 bbox 上部区域搜索
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

    # 回退到完整 bbox
    y2e_full = min(img_h - 1, y2 + margin)
    in_full = (
        lidar_valid
        & (lidar_uv[:, 0] >= x1e)
        & (lidar_uv[:, 0] <= min(img_w - 1, x2 + margin))
        & (lidar_uv[:, 1] >= y1e)
        & (lidar_uv[:, 1] <= y2e_full)
        & (lidar_depth >= min_depth)
        & (lidar_depth <= max_depth)
    )
    if in_full.sum() > 0:
        return float(np.percentile(lidar_depth[in_full], depth_percentile))
    
    return None

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: 2D 检测
# ══════════════════════════════════════════════════════════════════════════════

def build_gdino_model(config_path: str, checkpoint_path: str, device: str):
    """加载 Grounding DINO 模型"""
    sys.path.insert(0, "third_party/GroundingDINO")
    from groundingdino.util.inference import load_model
    return load_model(config_path, checkpoint_path, device=device)

def expand_bbox(
    bbox: List[int],
    img_h: int,
    img_w: int,
    expand_ratio: float = 0.35,
) -> List[int]:
    """
    扩展 bbox，确保完整覆盖红绿灯。
    
    Parameters
    ----------
    bbox : [x1, y1, x2, y2]
    expand_ratio : 各方向扩展比例（相对于bbox尺寸）
    
    Returns
    -------
    expanded_bbox : [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    # 行人灯底部标识更容易漏框，向下方向适当放大
    expand_x = int(bw * expand_ratio * 1.1)
    expand_y_top = int(bh * expand_ratio * 0.8)
    expand_y_bottom = int(bh * expand_ratio * 1.8)

    x1_new = max(0, x1 - expand_x)
    y1_new = max(0, y1 - expand_y_top)
    x2_new = min(img_w - 1, x2 + expand_x)
    y2_new = min(img_h - 1, y2 + expand_y_bottom)
    
    return [x1_new, y1_new, x2_new, y2_new]

def detect_frame(
    model, image_path: str,
    box_threshold: float, text_threshold: float,
    device: str,
    expand_ratio: float = 0.2,
) -> Tuple[List[List[int]], List[float], np.ndarray]:
    """
    检测单帧图像中的红绿灯，并扩展bbox。
    
    Returns: (boxes_pixel, scores, image_source)
    """
    from groundingdino.util.inference import load_image, predict
    
    image_source, image_tensor = load_image(image_path)
    boxes, logits, _ = predict(
        model, image_tensor, "traffic light", box_threshold, text_threshold, device=device
    )
    
    h, w = image_source.shape[:2]
    boxes_pixel = []
    scores = []
    
    for box, logit in zip(boxes, logits):
        cx, cy, bw, bh = box.tolist()
        x1 = int(max(0, (cx - bw / 2) * w))
        y1 = int(max(0, (cy - bh / 2) * h))
        x2 = int(min(w - 1, (cx + bw / 2) * w))
        y2 = int(min(h - 1, (cy + bh / 2) * h))
        
        # 扩展 bbox
        expanded = expand_bbox([x1, y1, x2, y2], h, w, expand_ratio)
        boxes_pixel.append(expanded)
        scores.append(float(logit))
    
    return boxes_pixel, scores, image_source

def detect_all_frames(
    model, scene_dir: str, cam_ids: List[int],
    box_threshold: float, text_threshold: float,
    device: str, num_frames: int,
    cam_intrinsics: Dict[int, np.ndarray],
    cam_extrinsics: Dict[int, np.ndarray],
    lidar_margin: int = 10,
    lidar_top_fraction: float = 0.6,
    lidar_depth_percentile: float = 15.0,
    bbox_head_fraction: float = 0.3,
    expand_ratio: float = 0.2,
    dataset_type: str = "waymo",
) -> Dict[int, Dict[int, List[Detection]]]:
    """
    检测所有帧的红绿灯，同时估计深度和世界坐标。
    
    Returns: {frame_idx: {cam_id: [Detection, ...]}}
    """
    detections = defaultdict(lambda: defaultdict(list))
    
    for cam_id in cam_ids:
        intrinsics = cam_intrinsics[cam_id]
        extrinsics = cam_extrinsics[cam_id]
        
        print(f"  检测相机 {cam_id}...")
        for frame_idx in tqdm(range(num_frames), desc=f"Cam {cam_id}", leave=False):
            img_path = os.path.join(scene_dir, "images", f"{frame_idx:03d}_{cam_id}.jpg")
            if not os.path.exists(img_path):
                continue
            
            boxes, scores, image_source = detect_frame(
                model, img_path, box_threshold, text_threshold, device,
                expand_ratio
            )
            
            if not boxes:
                continue
            
            img_h, img_w = image_source.shape[:2]
            ego_pose = load_ego_pose(scene_dir, frame_idx)
            
            # 加载 LiDAR 用于深度估计
            lidar_xyz = load_lidar_xyz_vehicle(scene_dir, frame_idx)
            if len(lidar_xyz) > 0:
                lidar_uv, lidar_depth, lidar_valid = project_vehicle_to_image(
                    lidar_xyz, intrinsics, extrinsics, dataset_type
                )
            else:
                lidar_uv = np.zeros((0, 2))
                lidar_depth = np.zeros(0)
                lidar_valid = np.zeros(0, dtype=bool)
            
            for bbox, score in zip(boxes, scores):
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                # 用 bbox 上部位置对准信号灯头部
                cy = y1 + (y2 - y1) * bbox_head_fraction
                
                # 尝试 LiDAR 深度
                depth = estimate_lidar_depth(
                    lidar_uv, lidar_depth, lidar_valid,
                    bbox, img_h, img_w,
                    margin=lidar_margin,
                    top_fraction=lidar_top_fraction,
                    depth_percentile=lidar_depth_percentile,
                )
                
                # 计算光线（用于后续三角化）
                ray_o, ray_d = compute_ray_world(
                    cx, cy, intrinsics, extrinsics, ego_pose, dataset_type
                )
                
                # 计算世界坐标（如果有深度）
                xyz_world = None
                if depth is not None:
                    xyz_world = backproject_to_world(
                        cx, cy, depth, intrinsics, extrinsics, ego_pose, dataset_type
                    )
                
                det = Detection(
                    frame_idx=frame_idx,
                    cam_id=cam_id,
                    bbox=bbox,
                    score=score,
                    depth=depth,
                    xyz_world=xyz_world,
                    ray_o=ray_o,
                    ray_d=ray_d,
                )
                detections[frame_idx][cam_id].append(det)
    
    return detections

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: 2D 跟踪（IoU 关联，来自 v2）
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def track_detections(
    detections: Dict[int, Dict[int, List[Detection]]],
    iou_threshold: float = 0.3,
    max_gap: int = 10,
) -> List[Track]:
    """
    使用 IoU 关联进行 2D 跟踪。
    对每个相机独立跟踪。
    """
    all_frames = sorted(detections.keys())
    if not all_frames:
        return []
    
    all_cam_ids = set()
    for frame_dets in detections.values():
        all_cam_ids.update(frame_dets.keys())
    
    # 每个相机独立跟踪
    global_tracks = []
    global_id = 0
    
    for cam_id in sorted(all_cam_ids):
        active_tracks: List[Track] = []
        
        for frame_idx in all_frames:
            frame_dets = detections.get(frame_idx, {}).get(cam_id, [])
            
            # 清理过期轨迹
            active_tracks = [
                t for t in active_tracks 
                if frame_idx - t.detections[-1].frame_idx <= max_gap
            ]
            
            # 贪婪匹配
            matched_tracks = set()
            matched_dets = set()
            
            iou_pairs = []
            for di, det in enumerate(frame_dets):
                for ti, track in enumerate(active_tracks):
                    last_det = track.detections[-1]
                    iou = compute_iou(det.bbox, last_det.bbox)
                    if iou >= iou_threshold:
                        iou_pairs.append((iou, di, ti))
            
            iou_pairs.sort(reverse=True)
            for iou, di, ti in iou_pairs:
                if di in matched_dets or ti in matched_tracks:
                    continue
                active_tracks[ti].add_detection(frame_dets[di])
                matched_tracks.add(ti)
                matched_dets.add(di)
            
            # 未匹配的检测 → 新轨迹
            for di, det in enumerate(frame_dets):
                if di not in matched_dets:
                    new_track = Track(track_id=global_id)
                    new_track.add_detection(det)
                    active_tracks.append(new_track)
                    global_id += 1
        
        global_tracks.extend(active_tracks)
    
    print(f"  2D跟踪完成: {len(global_tracks)} 条轨迹")
    return global_tracks

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: 深度估计（综合 LiDAR + 三角化）
# ══════════════════════════════════════════════════════════════════════════════

def compute_ego_motion(scene_dir: str, num_frames: int) -> np.ndarray:
    """计算每帧的自车位移"""
    motion = np.zeros(num_frames)
    prev_pose = None
    
    for frame_idx in range(num_frames):
        pose = load_ego_pose(scene_dir, frame_idx)
        if prev_pose is not None:
            delta = np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3])
            motion[frame_idx] = delta
        prev_pose = pose
    
    return motion

def triangulate_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """
    多光线三角化，最小二乘求交点。
    """
    n = len(rays_o)
    if n < 2:
        return None, 0.0
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(n):
        d = rays_d[i] / (np.linalg.norm(rays_d[i]) + 1e-12)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ rays_o[i]
    
    try:
        sv = np.linalg.svd(A, compute_uv=False)
        cond = sv[-1] / (sv[0] + 1e-12)
        
        if cond < 1e-6:
            return None, cond
        
        xyz = np.linalg.lstsq(A, b, rcond=None)[0]
        return xyz, cond
    except:
        return None, 0.0

def compute_max_baseline(origins: np.ndarray) -> float:
    """计算一组相机中心的最大两两距离（米）。"""
    n = len(origins)
    if n < 2:
        return 0.0
    diff = origins[:, None, :] - origins[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return float(dists.max())

def estimate_track_depth(
    track: Track,
    scene_dir: str,
    ego_motion: np.ndarray,
    cam_intrinsics: Dict[int, np.ndarray],
    cam_extrinsics: Dict[int, np.ndarray],
    motion_thresh: float = 0.05,
    min_motion_dets: int = 2,
    triag_cond_thresh: float = 1e-4,
    min_triangulation_baseline: float = 0.8,
    dataset_type: str = "waymo",
) -> Optional[np.ndarray]:
    """
    估计单个轨迹的 3D 世界坐标。
    
    策略（按优先级）：
    1. 使用已有的 LiDAR 深度估计（取均值）
    2. 运动帧三角化
    3. 从 bbox 大小估计深度（fallback）
    """
    # 策略1：使用 LiDAR 深度
    lidar_xyz = [d.xyz_world for d in track.detections if d.xyz_world is not None]
    if len(lidar_xyz) >= 2:
        # 有足够的 LiDAR 深度，取均值
        xyz_mean = np.mean(lidar_xyz, axis=0)
        return xyz_mean
    
    # 策略2：运动帧三角化（仅在基线足够时启用）
    motion_dets = []
    for det in track.detections:
        if det.frame_idx < len(ego_motion) and ego_motion[det.frame_idx] > motion_thresh:
            if det.ray_o is not None and det.ray_d is not None:
                motion_dets.append(det)
    
    if len(motion_dets) >= min_motion_dets:
        rays_o = np.array([d.ray_o for d in motion_dets])
        rays_d = np.array([d.ray_d for d in motion_dets])
        baseline = compute_max_baseline(rays_o)
        if baseline >= min_triangulation_baseline:
            xyz, cond = triangulate_rays(rays_o, rays_d)
            if xyz is not None and cond >= triag_cond_thresh:
                # 合理性检查
                dists = np.linalg.norm(rays_o - xyz[None, :], axis=1)
                if dists.min() > 0.5 and dists.max() < 150.0:
                    return xyz
    
    # 策略2.5：用全部观测三角化（静止帧不丢权重）
    all_ray_dets = [d for d in track.detections if d.ray_o is not None and d.ray_d is not None]
    if len(all_ray_dets) >= 3:
        rays_o_all = np.array([d.ray_o for d in all_ray_dets])
        rays_d_all = np.array([d.ray_d for d in all_ray_dets])
        baseline_all = compute_max_baseline(rays_o_all)
        if baseline_all >= min_triangulation_baseline:
            xyz_all, cond_all = triangulate_rays(rays_o_all, rays_d_all)
            if xyz_all is not None and cond_all >= triag_cond_thresh:
                dists_all = np.linalg.norm(rays_o_all - xyz_all[None, :], axis=1)
                if dists_all.min() > 0.5 and dists_all.max() < 150.0:
                    # 少量 LiDAR 时做融合，避免被错误三角化主导
                    if len(lidar_xyz) >= 1:
                        xyz_lidar = np.mean(lidar_xyz, axis=0)
                        return 0.7 * xyz_lidar + 0.3 * xyz_all
                    return xyz_all

    # 如果有少量 LiDAR 深度，用它（优先于 bbox 尺寸回退）
    if len(lidar_xyz) >= 1:
        return np.mean(lidar_xyz, axis=0)
    
    # 策略3：从 bbox 大小估计深度（fallback）
    if len(track.detections) >= 1:
        # 选置信度最高的检测
        best_det = max(track.detections, key=lambda d: d.score)
        bbox_height = best_det.bbox[3] - best_det.bbox[1]
        
        intrinsics = cam_intrinsics[best_det.cam_id]
        extrinsics = cam_extrinsics[best_det.cam_id]
        fv = intrinsics[1]
        
        # 红绿灯高度约1.2米
        estimated_depth = (1.2 * fv) / (bbox_height + 1e-6)
        estimated_depth = np.clip(estimated_depth, 5.0, 80.0)
        
        ego_pose = load_ego_pose(scene_dir, best_det.frame_idx)
        cx = (best_det.bbox[0] + best_det.bbox[2]) / 2
        cy = best_det.bbox[1] + (best_det.bbox[3] - best_det.bbox[1]) * 0.3
        
        xyz = backproject_to_world(
            cx, cy, estimated_depth, intrinsics, extrinsics, ego_pose, dataset_type
        )
        return xyz
    
    return None

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: 3D 聚类合并（来自 backup 的 DBSCAN）
# ══════════════════════════════════════════════════════════════════════════════

def merge_tracks_by_3d(
    tracks: List[Track],
    merge_dist: float = 2.5,
    min_overlap_ratio: float = 0.0,
) -> List[Track]:
    """
    根据 3D 位置合并属于同一红绿灯的轨迹（使用 DBSCAN）。
    """
    valid_tracks = [t for t in tracks if t.xyz_world is not None]
    
    if len(valid_tracks) <= 1:
        return valid_tracks
    
    # 使用 DBSCAN 聚类
    xyz_array = np.array([t.xyz_world for t in valid_tracks])
    labels = DBSCAN(eps=merge_dist, min_samples=1).fit_predict(xyz_array)
    
    # 合并同一簇的轨迹
    clusters: Dict[int, List[Track]] = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(valid_tracks[i])
    
    merged = []
    for label, group in clusters.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # 合并组内轨迹
            merged_track = Track(track_id=group[0].track_id)
            for t in group:
                for det in t.detections:
                    merged_track.add_detection(det)
            
            # 重新计算3D位置（检测数量加权）
            xyz_sum = np.zeros(3)
            weight_sum = 0
            for t in group:
                w = len(t.detections)
                xyz_sum += t.xyz_world * w
                weight_sum += w
            merged_track.xyz_world = xyz_sum / weight_sum
            
            merged.append(merged_track)
    
    print(f"  3D聚类合并: {len(valid_tracks)} → {len(merged)} 条轨迹")
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4.5: 反投影补全（将已知3D位置投影到所有相机的所有帧）
# ══════════════════════════════════════════════════════════════════════════════

def reproject_tracks_to_all_frames(
    tracks: List[Track],
    scene_dir: str,
    num_frames: int,
    cam_ids: List[int],
    cam_intrinsics: Dict[int, np.ndarray],
    cam_extrinsics: Dict[int, np.ndarray],
    expand_ratio: float = 0.2,
    min_depth: float = 5.0,
    max_depth: float = 120.0,
    margin_pixels: int = 50,
    refine_existing: bool = True,
    refine_iou_thresh: float = 0.2,
    refine_score_thresh: float = 0.6,
    dataset_type: str = "waymo",
) -> List[Track]:
    """
    将已知3D位置的红绿灯反投影到所有帧的所有相机。
    
    这解决了"静止帧检测不到，但运动帧能检测到"的问题：
    一旦确定了红绿灯的3D位置，就可以反投影到所有帧生成补全的检测框。
    
    Parameters
    ----------
    tracks : 已有3D位置的轨迹
    scene_dir : 场景目录
    num_frames : 总帧数
    cam_ids : 相机ID列表
    cam_intrinsics : 相机内参
    cam_extrinsics : 相机外参
    expand_ratio : bbox扩展比例
    min_depth : 最小有效深度
    max_depth : 最大有效深度
    margin_pixels : 图像边界margin
    
    Returns
    -------
    tracks : 补全后的轨迹列表
    """
    # 获取图像尺寸（从第一张图）
    img_h, img_w = None, None
    for cam_id in cam_ids:
        img_path = os.path.join(scene_dir, "images", f"000_{cam_id}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            break
    
    if img_h is None:
        print("  [WARN] 无法获取图像尺寸，跳过反投影补全")
        return tracks
    
    total_added = 0
    
    for track in tracks:
        if track.xyz_world is None:
            continue
        
        xyz_world = track.xyz_world
        
        # 估计bbox大小（从已有检测中取中位数）
        bbox_heights = []
        bbox_widths = []
        for det in track.detections:
            bh = det.bbox[3] - det.bbox[1]
            bw = det.bbox[2] - det.bbox[0]
            if bh > 0 and bw > 0:
                bbox_heights.append(bh)
                bbox_widths.append(bw)
        
        if not bbox_heights:
            continue
        
        # 使用中位数bbox大小
        median_height = np.median(bbox_heights)
        median_width = np.median(bbox_widths)
        
        # 对所有帧的所有相机反投影
        for frame_idx in range(num_frames):
            ego_pose = load_ego_pose(scene_dir, frame_idx)
            
            for cam_id in cam_ids:
                intrinsics = cam_intrinsics[cam_id]
                extrinsics = cam_extrinsics[cam_id]
                
                # 反投影到图像
                uv, depth, valid = project_world_to_image(
                    xyz_world, intrinsics, extrinsics, ego_pose, dataset_type
                )
                
                if not valid:
                    continue
                
                # 检查深度范围
                if depth < min_depth or depth > max_depth:
                    continue
                
                u, v = uv
                
                # 检查是否在图像内（带margin）
                if u < margin_pixels or u > img_w - margin_pixels:
                    continue
                if v < margin_pixels or v > img_h - margin_pixels:
                    continue
                
                # 根据深度缩放bbox大小（简单的相似三角形）
                # 假设参考深度是已有检测的平均深度
                ref_depths = [d.depth for d in track.detections if d.depth is not None]
                if ref_depths:
                    ref_depth = np.median(ref_depths)
                    scale = ref_depth / depth
                else:
                    scale = 1.0
                
                scaled_height = int(median_height * scale)
                scaled_width = int(median_width * scale)
                
                # 生成bbox（中心在投影点）
                x1 = int(u - scaled_width / 2)
                y1 = int(v - scaled_height / 2)
                x2 = int(u + scaled_width / 2)
                y2 = int(v + scaled_height / 2)
                
                # 边界裁剪
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w - 1, x2)
                y2 = min(img_h - 1, y2)
                
                # 扩展bbox
                bbox = expand_bbox([x1, y1, x2, y2], img_h, img_w, expand_ratio)
                
                # 创建补全的检测
                existing_dets = [
                    d for d in track.detections
                    if d.frame_idx == frame_idx and d.cam_id == cam_id
                ]

                if not existing_dets:
                    new_det = Detection(
                        frame_idx=frame_idx,
                        cam_id=cam_id,
                        bbox=bbox,
                        score=0.5,  # 补全的检测给一个中等置信度
                        depth=depth,
                        xyz_world=xyz_world,
                    )
                    track.add_detection(new_det)
                    total_added += 1
                    continue

                if not refine_existing:
                    continue

                best_existing = max(existing_dets, key=lambda d: d.score)
                iou_proj = compute_iou(best_existing.bbox, bbox)
                should_refine = (
                    (best_existing.depth is None and iou_proj < refine_iou_thresh)
                    or (best_existing.score < refine_score_thresh and iou_proj < 0.5)
                )

                if should_refine:
                    best_existing.bbox = bbox
                    best_existing.depth = depth
                    best_existing.xyz_world = xyz_world
                    best_existing.score = max(best_existing.score, 0.55)
    
    print(f"  反投影补全: 新增 {total_added} 个检测框")
    return tracks

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: 生成输出
# ══════════════════════════════════════════════════════════════════════════════

def generate_output(
    tracks: List[Track],
    scene_dir: str,
    num_frames: int,
    min_frames: int = 3,
    min_height: float = 1.5,
) -> Tuple[dict, dict]:
    """
    生成 instances_info.json 和 frame_instances.json
    """
    # 读取现有实例（确定起始ID）
    inst_path = os.path.join(scene_dir, "instances", "instances_info.json")
    existing_inst = {}
    if os.path.exists(inst_path):
        with open(inst_path) as f:
            existing_inst = json.load(f)
    
    existing_ids = [int(k) for k in existing_inst.keys()] if existing_inst else [-1]
    start_id = max(existing_ids) + 1
    
    # 构建输出
    tl_inst = {}
    tl_fi = {str(i): [] for i in range(num_frames)}
    
    inst_counter = 0
    
    for track in tracks:
        if track.xyz_world is None:
            continue
        
        # 过滤：帧数不足
        if track.num_frames < min_frames:
            print(f"  [SKIP] Track {track.track_id}: {track.num_frames} frames < {min_frames}")
            continue
        
        # 过滤：高度太低
        first_frame = track.detections[0].frame_idx
        ego_pose = load_ego_pose(scene_dir, first_frame)
        ego_z = ego_pose[2, 3]
        rel_z = track.xyz_world[2] - ego_z
        
        if rel_z < min_height:
            print(f"  [SKIP] Track {track.track_id}: rel_z={rel_z:.1f}m < {min_height}m")
            continue
        
        inst_id = start_id + inst_counter
        inst_counter += 1
        
        visible_frames = sorted(set(d.frame_idx for d in track.detections))
        
        tx, ty, tz = track.xyz_world.tolist()
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
        
        print(f"  [OK] Track {track.track_id} → inst {inst_id}: "
              f"frames={len(visible_frames)}, xyz={track.xyz_world.round(2)}")
    
    return tl_inst, tl_fi

def generate_masks(
    tracks: List[Track],
    scene_dir: str,
    num_frames: int,
    cam_ids: List[int],
):
    """
    为每帧生成红绿灯 mask。
    使用轨迹中的检测bbox（2D跟踪结果）。
    """
    mask_dir = os.path.join(scene_dir, "dynamic_masks", "traffic_light")
    os.makedirs(mask_dir, exist_ok=True)
    
    # 组织检测：{(frame, cam): [bboxes]}
    frame_cam_bboxes: Dict[Tuple[int, int], List[List[int]]] = defaultdict(list)
    
    for track in tracks:
        if track.xyz_world is None:
            continue
        for det in track.detections:
            frame_cam_bboxes[(det.frame_idx, det.cam_id)].append(det.bbox)
    
    # 确定 all mask 目录
    fine_mask_root = os.path.join(scene_dir, "fine_dynamic_masks")
    dyn_mask_root = fine_mask_root if os.path.exists(fine_mask_root) else os.path.join(scene_dir, "dynamic_masks")
    all_mask_dir = os.path.join(dyn_mask_root, "all")
    os.makedirs(all_mask_dir, exist_ok=True)
    
    # 生成 mask
    for frame_idx in tqdm(range(num_frames), desc="生成mask"):
        for cam_id in cam_ids:
            img_path = os.path.join(scene_dir, "images", f"{frame_idx:03d}_{cam_id}.jpg")
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            bboxes = frame_cam_bboxes.get((frame_idx, cam_id), [])
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
            
            # 保存 traffic_light mask
            cv2.imwrite(os.path.join(mask_dir, f"{frame_idx:03d}_{cam_id}.png"), mask)
            
            # 合并到 all mask
            all_mask_path = os.path.join(all_mask_dir, f"{frame_idx:03d}_{cam_id}.png")
            if os.path.exists(all_mask_path):
                existing = cv2.imread(all_mask_path, cv2.IMREAD_GRAYSCALE)
                if existing is not None and existing.shape == mask.shape:
                    mask = cv2.bitwise_or(existing, mask)
            cv2.imwrite(all_mask_path, mask)
    
    print(f"  TL mask 已 OR 合并进 {all_mask_dir}")

def cleanup_previous_traffic_light_outputs(scene_dir: str):
    """
    清理旧的红绿灯输出，避免重复运行时残留历史结果。
    只删除 traffic light 相关文件，不影响其他类别标注。
    """
    stale_files = [
        os.path.join(scene_dir, "instances", "traffic_light_instances_info.json"),
        os.path.join(scene_dir, "instances", "traffic_light_frame_instances.json"),
    ]
    stale_dirs = [
        os.path.join(scene_dir, "dynamic_masks", "traffic_light"),
        os.path.join(scene_dir, "fine_dynamic_masks", "traffic_light"),
    ]

    removed = 0
    for p in stale_files:
        if os.path.exists(p):
            os.remove(p)
            removed += 1
    for p in stale_dirs:
        if os.path.isdir(p):
            shutil.rmtree(p)
            removed += 1

    if removed > 0:
        print(f"[INFO] 已清理旧红绿灯输出: {removed} 项")

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="红绿灯检测与跟踪 v3 —— 综合版（2D跟踪 + LiDAR深度 + 三角化 + DBSCAN）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 路径
    parser.add_argument("--scene_dir", required=True, help="场景目录")
    parser.add_argument("--gdino_config", required=True, help="Grounding DINO 配置")
    parser.add_argument("--gdino_checkpoint", required=True, help="Grounding DINO 权重")
    
    # 检测参数
    parser.add_argument("--box_threshold", type=float, default=0.55,
                        help="Grounding DINO box 置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.40,
                        help="Grounding DINO text 置信度阈值")
    parser.add_argument("--cam_ids", type=int, nargs="+", default=[0, 1, 2],
                        help="参与检测的相机 ID（0=前，1=前左，2=前右）")
    parser.add_argument("--dataset_type", type=str, default="waymo",
                        choices=["waymo", "nuscenes", "kitti"],
                        help="数据集类型（影响外参坐标系与文件命名兼容）")
    parser.add_argument("--device", default="cuda", help="推理设备")
    
    # LiDAR 深度参数（来自 backup）
    parser.add_argument("--lidar_margin", type=int, default=15,
                        help="向外扩大 bbox 以搜索 LiDAR 点的像素数")
    parser.add_argument("--lidar_top_fraction", type=float, default=0.6,
                        help="仅使用 bbox 上方该比例区域搜索 LiDAR 深度")
    parser.add_argument("--lidar_depth_percentile", type=float, default=15.0,
                        help="LiDAR 深度取该百分位值")
    parser.add_argument("--bbox_head_fraction", type=float, default=0.3,
                        help="用 bbox 顶部该比例处的像素进行反投影")
    
    # 跟踪参数（来自 v2）
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="IoU 关联阈值")
    parser.add_argument("--max_gap", type=int, default=10,
                        help="轨迹最大允许断开帧数")
    
    # 三角化参数
    parser.add_argument("--motion_thresh", type=float, default=0.05,
                        help="判定运动帧的位移阈值（米）")
    parser.add_argument("--min_motion_dets", type=int, default=2,
                        help="三角化最少需要的运动帧检测数")
    parser.add_argument("--triag_cond_thresh", type=float, default=1e-4,
                        help="三角化矩阵条件数阈值")
    parser.add_argument("--min_triangulation_baseline", type=float, default=0.8,
                        help="三角化最小相机基线（米），低于此值不执行三角化")
    
    # 聚类合并参数（来自 backup）
    parser.add_argument("--merge_dist", type=float, default=2.5,
                        help="合并轨迹的3D距离阈值（米）")
    
    # bbox扩展参数（新增）
    parser.add_argument("--expand_ratio", type=float, default=0.45,
                        help="bbox各方向扩展比例（0.3=各边扩展30%）")
    
    # 反投影补全参数（新增）
    parser.add_argument("--reproject", action="store_true",
                        help="启用反投影补全（将已知3D位置投影到所有帧所有相机）")
    parser.add_argument("--reproject_min_depth", type=float, default=5.0,
                        help="反投影最小有效深度（米）")
    parser.add_argument("--reproject_max_depth", type=float, default=80.0,
                        help="反投影最大有效深度（米）")
    parser.add_argument("--reproject_margin", type=int, default=50,
                        help="反投影图像边界margin（像素）")
    parser.add_argument("--reproject_refine_existing", action="store_true", default=True,
                        help="反投影时修正已有但明显偏移的检测框")
    parser.add_argument("--reproject_refine_iou_thresh", type=float, default=0.2,
                        help="已有框与反投影框IoU低于该阈值时，允许修正")
    parser.add_argument("--reproject_refine_score_thresh", type=float, default=0.6,
                        help="低于该分数且与反投影不一致的框，优先被修正")
    
    # 过滤参数
    parser.add_argument("--min_frames", type=int, default=3,
                        help="实例最少帧数")
    parser.add_argument("--min_height", type=float, default=1.5,
                        help="实例最小相对高度（米）")
    
    # 输出
    parser.add_argument("--save_masks", action="store_true",
                        help="保存 2D 检测框为 binary mask")
    parser.add_argument("--merge", action="store_true",
                        help="合并到现有 JSON")
    
    args = parser.parse_args()
    scene_dir = args.scene_dir
    
    assert os.path.isdir(scene_dir), f"场景不存在: {scene_dir}"
    cleanup_previous_traffic_light_outputs(scene_dir)
    num_frames = count_frames(scene_dir)
    pose_dir = get_pose_dir(scene_dir)
    print(f"[INFO] 场景: {scene_dir}")
    print(f"[INFO] 位姿目录: {pose_dir}")
    print(f"[INFO] 帧数: {num_frames}")
    print(f"[INFO] 相机: {args.cam_ids}")
    print(f"[INFO] 数据集类型: {args.dataset_type}")
    
    # 加载相机参数
    print("[INFO] 加载相机参数...")
    cam_intrinsics = {cid: load_intrinsics(scene_dir, cid) for cid in args.cam_ids}
    cam_extrinsics = {cid: load_extrinsics(scene_dir, cid, 0) for cid in args.cam_ids}
    
    # Phase 1: 检测 + LiDAR 深度
    print("\n[Phase 1] 2D 检测 + LiDAR 深度估计（bbox扩展）...")
    gdino_model = build_gdino_model(args.gdino_config, args.gdino_checkpoint, args.device)
    detections = detect_all_frames(
        gdino_model, scene_dir, args.cam_ids,
        args.box_threshold, args.text_threshold, args.device, num_frames,
        cam_intrinsics, cam_extrinsics,
        args.lidar_margin, args.lidar_top_fraction, args.lidar_depth_percentile,
        args.bbox_head_fraction,
        args.expand_ratio,
        args.dataset_type,
    )
    total_dets = sum(len(d) for fd in detections.values() for d in fd.values())
    lidar_dets = sum(1 for fd in detections.values() for cams in fd.values() for d in cams if d.depth is not None)
    print(f"  检测总数: {total_dets}, 其中有LiDAR深度: {lidar_dets}")
    
    # Phase 2: 2D 跟踪
    print("\n[Phase 2] 2D 跟踪（IoU关联）...")
    tracks = track_detections(detections, args.iou_threshold, args.max_gap)
    
    # 计算自车运动
    print("\n[INFO] 计算自车运动...")
    ego_motion = compute_ego_motion(scene_dir, num_frames)
    motion_frames = (ego_motion > args.motion_thresh).sum()
    print(f"  运动帧数: {motion_frames}/{num_frames}")
    
    # Phase 3: 深度估计
    print("\n[Phase 3] 深度估计（LiDAR优先 + 三角化）...")
    for track in tracks:
        xyz = estimate_track_depth(
            track, scene_dir, ego_motion,
            cam_intrinsics, cam_extrinsics,
            args.motion_thresh, args.min_motion_dets, args.triag_cond_thresh,
            args.min_triangulation_baseline,
            args.dataset_type,
        )
        track.xyz_world = xyz
    
    valid_count = sum(1 for t in tracks if t.xyz_world is not None)
    print(f"  有效轨迹: {valid_count}/{len(tracks)}")
    
    # Phase 4: 3D 聚类合并
    print("\n[Phase 4] 3D 聚类合并...")
    tracks = merge_tracks_by_3d(tracks, args.merge_dist)
    
    # Phase 4.5: 反投影补全（可选）
    if args.reproject:
        print("\n[Phase 4.5] 反投影补全...")
        tracks = reproject_tracks_to_all_frames(
            tracks, scene_dir, num_frames, args.cam_ids,
            cam_intrinsics, cam_extrinsics,
            args.expand_ratio,
            args.reproject_min_depth, args.reproject_max_depth, args.reproject_margin,
            args.reproject_refine_existing,
            args.reproject_refine_iou_thresh,
            args.reproject_refine_score_thresh,
            args.dataset_type,
        )
    
    # Phase 5: 生成输出
    print("\n[Phase 5] 生成输出...")
    tl_inst, tl_fi = generate_output(
        tracks, scene_dir, num_frames,
        args.min_frames, args.min_height
    )
    print(f"  有效实例数: {len(tl_inst)}")
    
    # 保存 JSON
    os.makedirs(os.path.join(scene_dir, "instances"), exist_ok=True)
    
    if args.merge:
        inst_path = os.path.join(scene_dir, "instances", "instances_info.json")
        fi_path = os.path.join(scene_dir, "instances", "frame_instances.json")
        
        existing_inst = {}
        existing_fi = {}
        if os.path.exists(inst_path):
            with open(inst_path) as f:
                existing_inst = json.load(f)
        if os.path.exists(fi_path):
            with open(fi_path) as f:
                existing_fi = json.load(f)

        # 每次重跑前移除旧 traffic_light_*，避免历史残留导致脏标注
        existing_inst = {
            k: v for k, v in existing_inst.items()
            if not str(k).startswith("traffic_light_")
        }
        filtered_fi = {}
        for frame_key, ids in existing_fi.items():
            kept_ids = [
                obj_id for obj_id in ids
                if not str(obj_id).startswith("traffic_light_")
            ]
            if kept_ids:
                filtered_fi[frame_key] = kept_ids
        existing_fi = filtered_fi
        
        merged_inst = {**existing_inst, **tl_inst}
        merged_fi = {}
        for k in set(existing_fi.keys()) | set(tl_fi.keys()):
            merged_fi[k] = list(set(existing_fi.get(k, []) + tl_fi.get(k, [])))
        
        with open(inst_path, "w") as f:
            json.dump(merged_inst, f, indent=4)
        with open(fi_path, "w") as f:
            json.dump(merged_fi, f, indent=4)
        print(f"  已合并保存到 {inst_path}")
        print(f"  总实例数: {len(merged_inst)}（其中红绿灯 {len(tl_inst)} 个）")
    else:
        tl_inst_path = os.path.join(scene_dir, "instances", "traffic_light_instances_info.json")
        tl_fi_path = os.path.join(scene_dir, "instances", "traffic_light_frame_instances.json")
        
        with open(tl_inst_path, "w") as f:
            json.dump(tl_inst, f, indent=4)
        with open(tl_fi_path, "w") as f:
            json.dump(tl_fi, f, indent=4)
        print(f"  红绿灯标注已保存（独立文件，共 {len(tl_inst)} 个实例）")
        print(f"       instances_info : {tl_inst_path}")
        print(f"       frame_instances: {tl_fi_path}")
    
    # 生成 mask
    if args.save_masks:
        print("\n[Phase 6] 生成 mask...")
        generate_masks(tracks, scene_dir, num_frames, args.cam_ids)
        print(f"  Mask 已保存到 {os.path.join(scene_dir, 'dynamic_masks', 'traffic_light')}")
    
    print("\n[DONE]")

if __name__ == "__main__":
    main()
