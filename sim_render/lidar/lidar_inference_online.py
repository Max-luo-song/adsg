import os
import argparse
from lidargs.arguments import ModelParams, PipelineParams
from sim_render.lidar.render import render_sets, get_logger
from sim_render.infer_utils import load_transform_matrix
import numpy as np

class Renderer:
    """
    初始化模型、数据集，并暴露 render_single_frame() 给外部脚本实时调用
    """

    def __init__(self, lidar_checkpoint_path: str, source_path: str, output_dir="./outputs"):
        parser = argparse.ArgumentParser(description="Lidar Renderer Parameters")
        self.model_params = ModelParams(parser)
        self.pipeline_params = PipelineParams(parser)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = 'chery'
        self.block_size = 50
        self.iteration = 5000
        self.lidar_checkpoint_path = lidar_checkpoint_path
        self.quiet = True

        # set some runtime overrides and produce the final params object via extract()
        # 首先从 parser 获取带默认值的 Namespace，然后覆盖我们需要的项
        args_ns = parser.parse_args([])  # use registered defaults
        args_ns.source_path = source_path
        args_ns.model_path = lidar_checkpoint_path
        args_ns.max_depth = 100.0
        args_ns.output_path = output_dir

        # extract() 会返回一个 group (没有下划线的属性)，和主脚本行为一致
        self.model_params = self.model_params.extract(args_ns)
        self.pipeline_params = self.pipeline_params.extract(args_ns)

        # 初始化 logger（使用最终的 model_path）
        self.logger = get_logger(output_dir)

    def find_nearest_frame(self, pose, frame_num, source_path):
        """
        找到当前pose对应的最近帧index，以及相对变换矩阵
        """
        min_dist = float('inf')
        min_idx = -1
        min_transform = None

        for idx in range(frame_num):
            lidar_pose_txt = os.path.join(source_path, "lidar_pose", f"{idx:06d}.txt")
            frame_transform = load_transform_matrix(txt_path=lidar_pose_txt)
            dist = np.linalg.norm(pose[0:3, 3] - frame_transform[0:3, 3])
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
                min_transform = frame_transform

        rel_transform = np.linalg.inv(min_transform) @ pose
        return min_idx, rel_transform

    def render_single_frame(self, pose):
        """ Render LiDAR data from the given pose message.
        Args:
            pose: LiDAR渲染定义的世界坐标系下的novel pose

        Returns:

        Description:
            计算当前pose到全部帧中的最近帧（获取动态障碍物），计算新pose和最近帧的相对变换
            渲染新pose下的LiDAR点云
        """
        bin_files = [f for f in os.listdir(os.path.join(self.model_params.source_path, "lidar", "bin")) if f.endswith(".bin")]
        frame_num = len(bin_files)
        frame_idx, rel_transfrom = self.find_nearest_frame(pose, frame_num, self.model_params.source_path)
        novel_pose = [{"frame_id": frame_idx, "trans": rel_transfrom[:3, 3].T}]
        if self.dataset == "chery":
            from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
        elif self.dataset == "zdrive":
            from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
        else:
            print("[Lidar Renderer] Unknown dataset!")

        block_id = frame_idx // self.block_size
        block_info = {block_id: [frame_idx]}
        for block_id, train_frame_times in block_info.items():
            print(f"[Lidar Renderer] Rendering frame {frame_idx} in block {block_id}...")
            self.model_params.block_id = block_id  # update block id
            gt_dynamic_model = GT_Dataloader(
                self.model_params, train=False, train_frame_times=train_frame_times
            )
            gt_dynamic_model.set_novel_poses_setting(novel_pose)
            edit_obj_info, objs, logger = None, None, self.logger
            render_sets(
                gt_dynamic_model,
                self.model_params,
                self.iteration,
                self.pipeline_params,
                edit_obj_info,
                objs,
                logger,
            )
        
        return self.output_dir

if __name__ == "__main__":
    # Example usage
    renderer = Renderer(
        lidar_checkpoint_path="/nas_thoru/users/yangtao/qcraft/processed/test/20251025_163358_QCOYSD504206_1595_1610",
        source_path="/nas_thoru/users/yangtao/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610",
        output_dir="./realtime_output"
    )

    # Example pose (4x4 transformation matrix)
    example_pose = np.array([
        [1, 0, 0, 7987.086396549613],
        [0, 1, 0, -2964.653905917249],
        [0, 0, 1, -396.08292995686327],
        [0, 0, 0, 1]
    ])

    renderer.render_single_frame(example_pose)
