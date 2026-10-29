# renderer_manager.py
import os
import numpy as np
from cam.cam_inference_online import Renderer
from infer_utils import extract_lidar_extrinsics, load_transform_matrix, find_min_frame_txt, pose_to_transform_matrix, QCRAFT_CAMERA_DICT
from lidar.lidar_inference_online import Renderer as LidarRenderer


class CamRendererManager:
    def __init__(self, resume_from, source_path):
        print("[Init] Initializing renderer...")

        self.renderer = Renderer(
            resume_from=resume_from,
            cam_ids=[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12],
            downscales=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            output_dir="./realtime_output"
        )

        # Preload the matrices required for trajectory coordinate transformation
        lidar2ego_dict = extract_lidar_extrinsics(
            json_path=os.path.join(source_path, "data_frame_car_info.json")
        )
        self.lidar2ego = pose_to_transform_matrix(**lidar2ego_dict)

        lidar_pose_txt = find_min_frame_txt(os.path.join(source_path, "lidar_pose"))
        lidar_to_world_start = load_transform_matrix(txt_path=lidar_pose_txt)
        self.cam2lidar = load_transform_matrix(txt_path=os.path.join(source_path, "extrinsics", "0.txt"))

        self.inv_start = np.linalg.inv(lidar_to_world_start)

        print("[Init] Renderer initialization done.")

    def render_from_pose(self, pose_msg):
        """
        pose_msg: 来自 protobuf 的 MainCarInfo，包括 x,y,z,yaw,pitch,roll
        """

        raw = {
            "x": float(pose_msg.x),
            "y": float(pose_msg.y),
            "z": float(pose_msg.z),
            "yaw": float(pose_msg.yaw),
            "roll": float(pose_msg.roll),
            "pitch": float(pose_msg.pitch),
        }

        pose = pose_to_transform_matrix(**raw)
        lidar2world = pose @ self.lidar2ego
        rel = self.inv_start @ lidar2world
        cam2world = rel @ self.cam2lidar

        print("[Render] Pose:\n", pose)
        
        # if pose_msg.camera_id not in QCRAFT_CAMERA_DICT.item().keys():
        #     return None

        cam_id = QCRAFT_CAMERA_DICT[pose_msg.camera_id]


        print("cam_id: ", cam_id)
        frame_id = round(max(0, (float(pose_msg.timestamp) - self.renderer.start_timestamp)) / 0.1)
        print("pose_msg.timestamp: ", pose_msg.timestamp)

        output_paths = self.renderer.render_single_frame(cam2world, cam_id, frame_id)

        # output = self.renderer.render_single_frame(cam2world)
        print("[Render] Output:", output_paths)
        
        return output_paths

class LidarRendererManager:
    def __init__(self, lidar_checkpoint_path, source_path, output_dir):
        print("[Init] Initializing LiDAR renderer...")

        self.renderer = LidarRenderer(
            lidar_checkpoint_path=lidar_checkpoint_path,
            source_path=source_path,
            output_dir=output_dir
        )

        # Preload the matrices required for trajectory coordinate transformation
        lidar2ego_dict = extract_lidar_extrinsics(
            json_path=os.path.join(source_path, "data_frame_car_info.json")
        )
        self.lidar2ego = pose_to_transform_matrix(**lidar2ego_dict)

        lidar_pose_txt = find_min_frame_txt(os.path.join(source_path, "lidar_pose"))
        lidar_to_world_start = load_transform_matrix(txt_path=lidar_pose_txt)
        
        self.inv_start = np.linalg.inv(lidar_to_world_start)

        print("[Init] Renderer initialization done.")

    def render_from_pose(self, pose_msg):
        
        raw = {
            "x": pose_msg.x,
            "y": pose_msg.y,
            "z": pose_msg.z,
            "yaw": pose_msg.yaw,
            "roll": pose_msg.roll,
            "pitch": pose_msg.pitch
        }

        pose = pose_to_transform_matrix(**raw)
        # novel_pose是ego坐标, 计算新视角下的lidar_pose(世界坐标系下)
        lidar2world = pose @ self.lidar2ego
        # 世界坐标系下的lidar_pose 转换到 第一帧lidar_pose定义的world坐标系下
        rel_pose = self.inv_start @ lidar2world

        output_paths = self.renderer.render_single_frame(lidar2world)
        print("[Render] Output:", output_paths)

        return output_paths

# initialized only once
cam_renderer_manager: CamRendererManager = None
lidar_renderer_manager: LidarRendererManager = None

def init_renderer_cam(resume_from, source_path):
    global cam_renderer_manager
    if cam_renderer_manager is None:
        cam_renderer_manager = CamRendererManager(resume_from, source_path)
    return cam_renderer_manager

def init_renderer_lidar(lidar_checkpoint_path, source_path, output_dir):
    global lidar_renderer_manager
    if lidar_renderer_manager is None:
        lidar_renderer_manager = LidarRendererManager(lidar_checkpoint_path, source_path, output_dir)
    return lidar_renderer_manager
