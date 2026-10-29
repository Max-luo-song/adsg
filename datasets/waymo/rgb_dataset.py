"""ADSG dataset backed exclusively by RGB foundation-model predictions."""
import json
from pathlib import Path
import numpy as np
import torch

from datasets.driving_dataset import DrivingDataset, DepthMode
from datasets.base.scene_dataset import SceneDataset, ModelType
from datasets.waymo.rgb_sourceloader import RGBPixelSource
from utils.visualization import get_layout
from tools.rgb_only.assembly import validate_scene
from tools.rgb_only.common import read_scene


class RGBDrivingDataset(DrivingDataset):
    def __init__(self,data_cfg):
        SceneDataset.__init__(self,data_cfg)
        if data_cfg.lidar_source.load_lidar or data_cfg.pixel_source.load_smpl:
            raise ValueError('RGB-only mode forbids LiDAR and GT-dependent SMPL loading')
        self.type='waymo'
        self.data_path=str(Path(data_cfg.data_root)/f'{int(self.scene_idx):03d}')
        scene=read_scene(self.data_path)
        validate_scene(self.data_path,{})
        if list(map(str,data_cfg.pixel_source.cameras))!=scene['camera_ids']:
            raise ValueError('Training camera order must equal the prediction manifest camera order')
        if data_cfg.start_timestep!=0 or data_cfg.end_timestep!=-1:
            raise ValueError('Set frame ranges in the RGB scene list, then regenerate predictions in a new output root')
        self.start_timestep=scene['frames'][0]['frame_id']
        self.end_timestep=scene['frames'][-1]['frame_id']+1
        self.layout=get_layout(self.type)
        self.pixel_source,self.lidar_source=self.build_data_source()
        self.depth_mode=DepthMode.SINGLE_FRAME_RAW  # depth is MoGe camera-z, including moving objects
        (self.train_timesteps,self.test_timesteps,self.train_indices,self.test_indices)=self.split_train_test()
        self.train_image_set,self.test_image_set,self.full_image_set=self.build_split_wrapper()
        with np.load(Path(self.data_path)/'seed_points.npz',allow_pickle=False) as z:
            seeds={k:z[k].copy() for k in z.files}
        # Held-out RGB/depth must not initialize gaussians. Camera prediction is explicitly offline/all-RGB.
        allowed=set(self.start_timestep+int(f) for f in self.train_timesteps)
        mask=np.isin(seeds['frame_ids'],list(allowed))
        self.pseudo={k:torch.from_numpy(v[mask]).to(self.device) for k,v in seeds.items()}
        if len(self.pseudo['points'])<4:
            raise ValueError('Too few predicted seed points in the training split')
        self.aabb=self.get_aabb()

    def build_data_source(self):
        pixel=RGBPixelSource('waymo',self.data_cfg.pixel_source,self.data_path,
                             self.start_timestep,self.end_timestep,device=self.device)
        pixel.to(self.device)
        return pixel,None

    def get_aabb(self):
        if not hasattr(self,'pseudo'):
            return self.pixel_source.get_aabb()
        pts=self.pseudo['points']
        low=torch.quantile(pts,0.005,dim=0)-1
        high=torch.quantile(pts,0.995,dim=0)+1
        return torch.cat([low,high])

    def get_rgb_seed_points(self,road=False,max_points=800000):
        p=self.pseudo
        mask=(p['instance_ids']<0)&(p['road']==road)
        ids=torch.where(mask)[0]
        if len(ids)>max_points:
            ids=ids[torch.randperm(len(ids),device=ids.device)[:max_points]]
        if len(ids)<4:
            raise ValueError(f'Insufficient predicted {"road" if road else "background"} seeds')
        return p['points'][ids],p['colors'][ids]

    def get_init_objects(self,cur_node_type,instance_max_pts=5000,only_moving=True,
                         traj_length_thres=0.5,exclude_smpl=False):
        target={'RigidNodes':ModelType.RigidNodes,'DeformableNodes':ModelType.DeformableNodes,
                'TrafficLightNodes':ModelType.TrafficLightNodes}[cur_node_type]
        result={}
        source=self.pixel_source
        p=self.pseudo
        for idx in range(self.instance_num):
            if int(source.instances_model_types[idx])!=target:
                continue
            select=torch.where(p['instance_ids']==idx)[0]
            if len(select)<4:
                continue
            if len(select)>instance_max_pts:
                select=select[torch.randperm(len(select),device=select.device)[:instance_max_pts]]
            frame=p['frame_ids'][select].long()-self.start_timestep
            poses=source.instances_pose[frame,idx]
            # Same-frame world-to-object mapping: moving surfaces must not be fused in world coordinates.
            local=torch.bmm((p['points'][select]-poses[:,:3,3]).unsqueeze(1),poses[:,:3,:3]).squeeze(1)
            result[idx]=dict(node_type=cur_node_type,pts=local,colors=p['colors'][select],
                num_pts=len(select),poses=source.instances_pose[:,idx],size=source.instances_size[idx],
                frame_info=source.per_frame_instance_mask[:,idx])
        return result

    def get_lidar_samples(self,*args,**kwargs):
        raise RuntimeError('RGBDrivingDataset has no LiDAR. Use get_rgb_seed_points().')

    def get_init_smpl_objects(self,*args,**kwargs):
        raise RuntimeError('RGB-only humans use DeformableNodes; SMPL requires a separate RGB-only estimator.')
