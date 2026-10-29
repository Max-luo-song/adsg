from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import numpy as np

import torch
from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
from utils.misc import import_str
from utils.logging_utils import setup_logging
from models.trainers import BasicTrainer
from models.video_utils import render_novel_views, save_single_camera_video
from chery_tools.lidar_simulation import (
    unproject_depth_to_pointcloud,
    remove_ground_points,
    save_pointcloud_pcd,
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


@torch.no_grad()
def render_trajectory(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDatasetNovelView = None,
    cam_ids: Optional[List[int]] = None,
    downscales: Optional[List[int]] = None,
    traj_types: List[str] = None,
    fps: int = 10,
    render_rgb: bool = True,
    render_depth: bool = False,
    save_images: bool = True,
    generate_lidar_pc: bool = False,
):
    trainer.set_eval()

    logger.info("Rendering novel views...")

    render_keys = []
    if render_rgb:
        render_keys.append("rgbs")
    if render_depth:
        render_keys.append("depths")

    logger.info(f"Render keys: {render_keys}")

    camera_data_dict = dataset.load_specified_cameras(cam_ids, downscales)

    for traj_type in traj_types:
        logger.info(f"Trajectory type: {traj_type}")

        ref_cam_id = 0  # HACK
        render_traj = dataset.get_novel_render_traj(
            traj_type=traj_type,
            ref_cam_id=ref_cam_id,
            camera_data_dict=camera_data_dict,
            target_frames=dataset.frame_num,
            traj_path=None,
        )  # 单个相机（前向）的新轨迹
        if render_traj is None:
            continue

        output_dir = f"{cfg.log_dir}/novel_traj/{traj_type}_step{step}"
        os.makedirs(output_dir, exist_ok=True)

        # Render and save video
        video_output_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)
        video_output_path = os.path.join(video_output_dir, f"{traj_type}.mp4")

        image_output_dir = None
        if save_images:
            image_output_dir = os.path.join(output_dir, "images")
            os.makedirs(image_output_dir, exist_ok=True)

        ref_cam_data = camera_data_dict[ref_cam_id]

        depths_per_cam = {}
        for cam_id, target_cam_data in camera_data_dict.items():
            render_data = dataset.prepare_novel_view_render_data(
                traj=render_traj,
                ref_cam_data=ref_cam_data,
                target_cam_data=target_cam_data,
            )
            render_results = render_novel_views(
                trainer,
                render_data,
                target_cam_data,
            )
            del render_data

            save_single_camera_video(
                render_results,
                cam_id,
                dataset.start_timestep,
                dataset.end_timestep,
                video_output_path,
                image_output_dir,
                keys=render_keys,
                fps=fps,
                verbose=True,
            )

            # 鱼眼相机不保存 depth
            depths_per_cam[cam_id] = render_results["depths"]

            del render_results

        logger.info(f"Saved novel view videos for trajectory type: {traj_type}")

        # [DEPRECATED] 生成雷达点云
        if generate_lidar_pc:
            pc_output_dir = os.path.join(output_dir, "lidar_point_clouds")
            os.makedirs(pc_output_dir, exist_ok=True)

            pinhole_cam_ids = [
                id for id, cam in camera_data_dict.items() if not cam.is_fisheye
            ]
            intrinsics = {
                cam_id: camera_data_dict[cam_id].intrinsics.cpu().numpy()
                for cam_id in pinhole_cam_ids
            }
            T_cam_to_lidars = {
                cam_id: camera_data_dict[cam_id].cam_to_main_lidar
                for cam_id in pinhole_cam_ids
            }

            for frame_id, t in enumerate(range(dataset.start_timestep, dataset.end_timestep)):
                all_points = []
                for cam_id in pinhole_cam_ids:
                    points = unproject_depth_to_pointcloud(
                        depths_per_cam[cam_id][frame_id],
                        intrinsics[cam_id][frame_id],
                        T_cam_to_lidars[cam_id],
                    )
                    if len(points) > 0:
                        all_points.append(points)

                if len(all_points) > 0:
                    # 合并所有相机的点云
                    all_points = np.vstack(all_points)
                    all_points = remove_ground_points(all_points, ground_height=-3)

                    # 保存为PCD文件
                    pcd_path = os.path.join(pc_output_dir, f"frame_{t:06d}.pcd")
                    save_pointcloud_pcd(all_points, pcd_path)

                    logger.debug(
                        f"Frame {t}: Merged {len(all_points)} points -> {pcd_path}"
                    )

        del depths_per_cam


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False

    global logger
    setup_logging(level=logging.INFO, time_string=current_time)

    camera_ids = (
        args.cam_ids if len(args.cam_ids) > 0 else cfg.data.pixel_source.cameras
    )
    downscales = (
        args.downscales
        if len(args.downscales) > 0
        else cfg.data.pixel_source.downscale_when_loading
    )
    logger.info(f"Camera IDs: {camera_ids}")
    logger.info(f"Downscales: {downscales}")
    assert len(camera_ids) == len(downscales)

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build dataset
    dataset = DrivingDatasetNovelView(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        # num_train_images=len(dataset.train_indices),
        num_full_images=len(dataset.train_indices + dataset.test_indices),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    render_trajectory(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        cam_ids=camera_ids,
        downscales=downscales,
        traj_types=args.traj_types,
        fps=args.fps,
        render_rgb=args.render_rgb,
        render_depth=args.render_depth,
        save_images=args.save_images,
        generate_lidar_pc=args.generate_lidar_pc,
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render novel trajectory for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--traj_types",
        default=None,
        nargs="+",
        type=str,
        help="Types of novel trajectories",
    )
    parser.add_argument(
        "--traj_path",
        default=None,
        type=str,
        help="directory of self-defined trajectories (cam2world)",
    )

    parser.add_argument(
        "--cam_ids",
        default=None,
        nargs="*",
        type=int,
        help="Camera ID to render",
        required=True,
    )
    parser.add_argument(
        "--downscales",
        default=None,
        nargs="*",
        type=float,
        help="Downscales for each camera ID",
        required=True,
    )

    parser.add_argument(
        "--fps",
        default=10,
        type=float,
        help="Frame per second of the rendered video",
    )
    parser.add_argument(
        "--save_catted_videos",
        type=bool,
        default=False,
        help="visualize lidar on image",
    )

    parser.add_argument(
        "--render_rgb", action="store_true", help="render rgb novel views"
    )
    parser.add_argument(
        "--render_depth", action="store_true", help="render depth novel views"
    )
    parser.add_argument(
        "--save_images", action="store_true", help="save rendered images"
    )
    parser.add_argument(
        "--generate_lidar_pc",
        action="store_true",
        help="generate point cloud for each frame",
    )

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)
