#!/usr/bin/env python3
"""
将 traffic_light_instances_info.json 中的 3D box 投影到图像并输出 MP4 视频。

用法：
  python visualize_traffic_lights.py --scene_dir data/waymo/processed/training/621
  python visualize_traffic_lights.py --scene_dir data/waymo/processed/training/621 --cam_ids 0 1 2
  python visualize_traffic_lights.py --scene_dir data/waymo/processed/training/621 --output_dir /inspire/hdd/global_user/guoluosong-253108120129/project/adstudio/output_adstudio_trafficlightyx --cam_ids 0 1 2
"""

import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

# ── 坐标系常量（与 waymo_sourceloader.py 保持一致） ───────────────────────────
# OpenCV camera: x 右, y 下, z 前
# Waymo camera : x 前, y 左, z 上  (extrinsics 文件的约定)
# OPENCV2DATASET : OpenCV → Waymo
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)

# 3D box 的 12 条边（顶点索引）
# 顶点编号: bottom(0-3, z=-h/2)  top(4-7, z=+h/2)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical edges
]

# 颜色池（BGR），按实例 ID 循环
COLORS = [
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 128, 0),
    (0, 200, 200),
    (200, 0, 200),
    (128, 255, 0),
    (0, 80, 255),
]


# ══════════════════════════════════════════════════════════════════════════════
# 几何工具
# ══════════════════════════════════════════════════════════════════════════════

def get_box_corners_world(obj_to_world: np.ndarray, box_size: list) -> np.ndarray:
    """
    根据 obj_to_world 矩阵和 box_size 计算 3D box 的 8 个角点世界坐标。

    box_size : [length(x), width(y), height(z)]

    顶点在 object frame 中的位置：
      0:(+l/2,+w/2,-h/2)  1:(+l/2,-w/2,-h/2)
      2:(-l/2,-w/2,-h/2)  3:(-l/2,+w/2,-h/2)
      4:(+l/2,+w/2,+h/2)  5:(+l/2,-w/2,+h/2)
      6:(-l/2,-w/2,+h/2)  7:(-l/2,+w/2,+h/2)
    """
    l, w, h = box_size[0] / 2, box_size[1] / 2, box_size[2] / 2
    corners_obj = np.array([
        [ l,  w, -h, 1],
        [ l, -w, -h, 1],
        [-l, -w, -h, 1],
        [-l,  w, -h, 1],
        [ l,  w,  h, 1],
        [ l, -w,  h, 1],
        [-l, -w,  h, 1],
        [-l,  w,  h, 1],
    ], dtype=np.float64)  # (8, 4)

    corners_world = (obj_to_world @ corners_obj.T).T  # (8, 4)
    return corners_world[:, :3]  # (8, 3)


def project_pts_to_image(
    pts_world: np.ndarray,       # (N, 3)
    fu: float, fv: float, cu: float, cv: float,
    cam_to_ego_raw: np.ndarray,  # (4, 4)  Waymo cam → vehicle（文件原始值）
    ego_pose: np.ndarray,        # (4, 4)  vehicle → world
):
    """
    世界坐标 → 像素坐标。

    变换链：
      world → vehicle  (via inv(ego_pose))
      vehicle → OpenCV camera  (via inv(cam_to_ego_raw @ OPENCV2DATASET))
      → 透视投影
    """
    # cam → vehicle，OpenCV 约定（与 sourceloader 一致）
    c2e_ocv = cam_to_ego_raw @ OPENCV2DATASET       # (4, 4)
    world_to_cam = np.linalg.inv(c2e_ocv) @ np.linalg.inv(ego_pose)  # (4, 4)

    ones = np.ones((len(pts_world), 1), dtype=np.float64)
    pts_hom = np.hstack([pts_world, ones])           # (N, 4)
    pts_cam = (world_to_cam @ pts_hom.T).T[:, :3]   # (N, 3)

    depth = pts_cam[:, 2]
    safe_d = np.where(depth > 0, depth, 1e-6)
    u = fu * pts_cam[:, 0] / safe_d + cu
    v = fv * pts_cam[:, 1] / safe_d + cv
    return np.stack([u, v], axis=1), depth


def draw_box_on_image(
    img: np.ndarray,
    corners_world: np.ndarray,   # (8, 3)
    intrinsics: np.ndarray,      # [fu, fv, cu, cv, ...]
    cam_to_ego_raw: np.ndarray,
    ego_pose: np.ndarray,
    color,
    label: str = "",
    thickness: int = 2,
    max_dist: float = 80.0,
):
    """在图像上绘制 3D box 的投影线框。"""
    h_img, w_img = img.shape[:2]
    fu, fv, cu, cv = intrinsics[:4]

    uv, depth = project_pts_to_image(
        corners_world, fu, fv, cu, cv, cam_to_ego_raw, ego_pose
    )

    # 所有角点都在相机后方，跳过
    if (depth <= 0.5).all():
        return

    # 用中心点做距离过滤
    center_w = corners_world.mean(axis=0, keepdims=True)
    center_uv, center_d = project_pts_to_image(
        center_w, fu, fv, cu, cv, cam_to_ego_raw, ego_pose
    )
    if center_d[0] <= 0.5 or center_d[0] > max_dist:
        return

    margin = 300
    def in_frame(p):
        return -margin <= p[0] <= w_img + margin and -margin <= p[1] <= h_img + margin

    # 绘制 12 条边
    for i, j in BOX_EDGES:
        if depth[i] <= 0 and depth[j] <= 0:
            continue
        pt1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        pt2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
        if not (in_frame(pt1) or in_frame(pt2)):
            continue
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 标签
    if label:
        cx = int(round(center_uv[0, 0]))
        cy = int(round(center_uv[0, 1]))
        text = f"{label}  {center_d[0]:.1f}m"
        # 黑色描边 + 彩色文字，提高可读性
        cv2.putText(img, text, (cx, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (cx, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# 视频合成工具
# ══════════════════════════════════════════════════════════════════════════════

def make_video(frames: list, out_path: str, fps: float = 10.0):
    if not frames:
        print(f"  [WARN] 无帧，跳过 {out_path}")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  [OK] {out_path}  ({len(frames)} frames, {w}x{h}, {size_mb:.1f}MB)")


def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def make_grid(imgs: list, target_w: int = 960) -> np.ndarray:
    """
    将多张图片水平拼接（统一高度）。
    各图先缩放到 target_w 宽度后再按最小高度裁齐。
    """
    resized = [resize_to_width(im, target_w) for im in imgs]
    target_h = min(im.shape[0] for im in resized)
    cropped = [im[:target_h] for im in resized]
    return np.concatenate(cropped, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scene_dir",
                        default="/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed/training/621",
                        help="已处理的场景目录")
    parser.add_argument("--inst_json", default=None,
                        help="instances_info JSON 路径（默认用 traffic_light_instances_info.json）")
    parser.add_argument("--fi_json", default=None,
                        help="frame_instances JSON 路径（默认用 traffic_light_frame_instances.json）")
    parser.add_argument("--output_dir", default="output/vis_traffic_lights_621")
    parser.add_argument("--cam_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="要渲染的相机 ID")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--resize_width", type=int, default=960,
                        help="每路相机图像缩放宽度（0=不缩放）")
    parser.add_argument("--max_dist", type=float, default=60.0,
                        help="超过此距离的 box 不绘制（米）")
    parser.add_argument("--no_per_cam", action="store_true",
                        help="不生成每路相机单独视频")
    parser.add_argument("--no_combined", action="store_true",
                        help="不生成合并视频")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scene_dir = args.scene_dir

    # ── 加载标注 ──────────────────────────────────────────────────────────────
    # 优先使用独立的 traffic_light 文件，若不存在则回退到合并文件
    if args.inst_json:
        inst_path = args.inst_json
    else:
        tl_inst_path = os.path.join(scene_dir, "instances", "traffic_light_instances_info.json")
        merged_inst_path = os.path.join(scene_dir, "instances", "instances_info.json")
        inst_path = tl_inst_path if os.path.exists(tl_inst_path) else merged_inst_path
    
    if args.fi_json:
        fi_path = args.fi_json
    else:
        tl_fi_path = os.path.join(scene_dir, "instances", "traffic_light_frame_instances.json")
        merged_fi_path = os.path.join(scene_dir, "instances", "frame_instances.json")
        fi_path = tl_fi_path if os.path.exists(tl_fi_path) else merged_fi_path
    
    print(f"[INFO] 加载实例信息: {inst_path}")
    print(f"[INFO] 加载帧实例索引: {fi_path}")
    
    with open(inst_path) as f:
        inst_info = json.load(f)
    with open(fi_path) as f:
        frame_inst = json.load(f)
    
    # 如果是合并文件，只保留 TrafficLight 类型的实例
    if "traffic_light" not in inst_path:
        inst_info = {k: v for k, v in inst_info.items() if v.get("class_name") == "TrafficLight"}
        print(f"[INFO] 从合并文件中筛选出 {len(inst_info)} 个红绿灯实例")

    print(f"[INFO] 加载 {len(inst_info)} 个红绿灯实例")

    # 预建快速查找表：inst_id -> frame_idx -> (obj_to_world, box_size)
    inst_lookup = {}
    for k_str, v in inst_info.items():
        inst_id = int(k_str)
        ann = v["frame_annotations"]
        inst_lookup[inst_id] = {
            int(fi): (np.array(o2w, dtype=np.float64), bs)
            for fi, o2w, bs in zip(
                ann["frame_idx"], ann["obj_to_world"], ann["box_size"]
            )
        }
    min_inst_id = min(inst_lookup.keys()) if inst_lookup else 0

    # ── 统计帧数 ──────────────────────────────────────────────────────────────
    ego_dir = os.path.join(scene_dir, "ego_pose")
    num_frames = len([f for f in os.listdir(ego_dir) if f.endswith(".txt")])
    print(f"[INFO] 帧数: {num_frames}")

    # ── 相机标定 ──────────────────────────────────────────────────────────────
    calibs = {}
    for cam_id in args.cam_ids:
        intr = np.loadtxt(os.path.join(scene_dir, "intrinsics", f"{cam_id}.txt"))
        extr = np.loadtxt(os.path.join(scene_dir, "extrinsics", f"{cam_id}.txt"))
        calibs[cam_id] = (intr, extr)

    # ── 渲染每帧 ─────────────────────────────────────────────────────────────
    cam_names = {0: "front", 1: "front_left", 2: "front_right",
                 3: "left", 4: "right"}
    per_cam_frames = {c: [] for c in args.cam_ids}

    for frame_idx in tqdm(range(num_frames), desc="渲染"):
        ego_pose = np.loadtxt(
            os.path.join(ego_dir, f"{frame_idx:03d}.txt")
        )
        active_ids = frame_inst.get(str(frame_idx), [])

        for cam_id in args.cam_ids:
            img_path = os.path.join(
                scene_dir, "images", f"{frame_idx:03d}_{cam_id}.jpg"
            )
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            intr, extr = calibs[cam_id]

            for inst_id in active_ids:
                if inst_id not in inst_lookup:
                    continue
                ann = inst_lookup[inst_id].get(frame_idx)
                if ann is None:
                    continue

                obj_to_world, box_size = ann
                corners = get_box_corners_world(obj_to_world, box_size)
                color = COLORS[inst_id % len(COLORS)]
                disp_id = inst_id - min_inst_id
                draw_box_on_image(
                    img, corners, intr, extr, ego_pose,
                    color=color,
                    label=f"TL{disp_id}",
                    thickness=2,
                    max_dist=args.max_dist,
                )

            # 帧号 + 相机名水印
            label_txt = f"f:{frame_idx:03d}  {cam_names.get(cam_id, cam_id)}"
            cv2.putText(img, label_txt, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label_txt, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if args.resize_width > 0:
                img = resize_to_width(img, args.resize_width)

            per_cam_frames[cam_id].append(img)

    # ── 输出视频 ─────────────────────────────────────────────────────────────
    print("\n[INFO] 写入视频...")

    if not args.no_per_cam:
        for cam_id in args.cam_ids:
            frames = per_cam_frames[cam_id]
            if not frames:
                continue
            out = os.path.join(
                args.output_dir,
                f"cam{cam_id}_{cam_names.get(cam_id, cam_id)}.mp4"
            )
            make_video(frames, out, fps=args.fps)

    if not args.no_combined:
        # 横向拼接所有有效相机（统一 resize_width）
        valid_cams = [c for c in args.cam_ids if per_cam_frames.get(c)]
        if valid_cams:
            n_frames = min(len(per_cam_frames[c]) for c in valid_cams)
            grid_frames = []
            for fi in range(n_frames):
                imgs = [per_cam_frames[c][fi] for c in valid_cams]
                grid_frames.append(make_grid(imgs, target_w=args.resize_width or 960))
            cam_str = "".join(str(c) for c in valid_cams)
            out = os.path.join(args.output_dir, f"combined_cam{cam_str}.mp4")
            make_video(grid_frames, out, fps=args.fps)

    print("\n[DONE] 视频已保存至:", args.output_dir)


if __name__ == "__main__":
    main()
