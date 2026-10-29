import os
import time
import numpy as np
import torch

from omegaconf import OmegaConf
from utils.misc import import_str
from models.trainers import BasicTrainer
from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
from models.video_utils import render_novel_views
from imageio import imwrite


class Renderer:
    """
    初始化模型、数据集，并暴露 render_single_frame() 给外部脚本实时调用
    """

    def __init__(self, resume_from: str, cam_ids, downscales, output_dir="./outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ==== 加载配置 ====
        log_dir = os.path.dirname(resume_from)
        cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==== 数据集 ====
        self.dataset = DrivingDatasetNovelView(data_cfg=cfg.data)
        self.cam_ids = cam_ids
        self.downscales = downscales

        # 加载对应相机数据
        self.camera_data_dict = self.dataset.load_specified_cameras(
            self.cam_ids, self.downscales
        )

        # ==== 模型 ====
        self.trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=cfg.model,
            num_full_images=len(self.dataset.train_indices + self.dataset.test_indices),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=device,
        )
        self.trainer.resume_from_checkpoint(
            ckpt_path=resume_from,
            load_only_model=True,
        )
        self.trainer.set_eval()
        self.start_timestamp = 1761382840.001

        print("[Renderer] Initialization completed.")

    @torch.no_grad()
    def render_single_frame(
        self,
        pose_cam2world: np.ndarray,
        cam_id: int,
        frame_id: int,
        ref_cam_id: int = 0,
    ):

        print("start prepare_online_render_data")
        render_results = {}
        start_time = time.time()
        print("pose_cam2world: ", pose_cam2world)
        traj = torch.from_numpy(pose_cam2world).float().to(self.trainer.device)
        traj = traj.unsqueeze(0)   # (1, 4, 4)
        cam_data = self.camera_data_dict[cam_id]
        ref_cam_data = self.camera_data_dict[ref_cam_id]

        render_data = self.dataset.prepare_online_render_data(
            traj=traj,
            ref_cam_data=ref_cam_data,
            target_cam_data=cam_data,
            frame_id=frame_id,
        )
        print("start render")
        # 渲染
        results = render_novel_views(self.trainer, render_data, cam_data)
        end_time_single = time.time()
        # print(f"Single frame rendered cam {cam_id} in {end_time_single - start_time_single:.3f} seconds.")

        # rgb = results["rgbs"][0].cpu().numpy()
        rgb = results["rgbs"][0]
        rgb = (rgb * 255).astype(np.uint8)

        timestamp = time.time()
        # out_path = os.path.join(self.output_dir, f"cam{cam_id}_{timestamp:.0f}.png")
        out_path = os.path.join(self.output_dir, f"cam{cam_id}_{timestamp:.3f}.png")
        imwrite(out_path, rgb)

        render_results[cam_id] = out_path

        end_time = time.time()
        print(f"[Renderer] Rendered frame in {end_time - start_time:.2f} seconds.")
        return render_results
