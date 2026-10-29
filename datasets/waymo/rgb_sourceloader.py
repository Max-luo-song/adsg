"""Read predicted calibration/depth/objects; never open Waymo GT calibration files."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from datasets.base.pixel_source import CameraData, ScenePixelSource
from datasets.base.scene_dataset import ModelType
from tools.rgb_only.common import camera_table, key_name, read_scene


class RGBCameraData(CameraData):
    def __init__(self, **kwargs):
        if kwargs.get('undistort', False):
            raise ValueError('RGB-only predicted pinhole cameras do not have GT lens distortion')
        super().__init__(**kwargs)
        folder=Path(self.data_path)
        maps=[]
        for fid in range(self.start_timestep,self.end_timestep):
            with np.load(folder/'moge_depth'/(key_name(fid,self.cam_id)+'.npz'),allow_pickle=False) as z:
                maps.append(np.where(z['valid'],z['depth'],0).astype(np.float32))
        self.load_depth(torch.from_numpy(np.stack(maps)))

    def load_calibrations(self):
        scene=read_scene(self.data_path)
        table,size=camera_table(Path(self.data_path)/'predicted_cameras.npz',scene)
        self.load_size=list(size)
        with Image.open(self.img_filepaths[0]) as image:
            self.original_size=(image.height,image.width)
        selected=[table[(fid,str(self.cam_id))] for fid in range(self.start_timestep,self.end_timestep)]
        self.cam_to_worlds=torch.from_numpy(np.stack([x[0] for x in selected])).float()
        self.intrinsics=torch.from_numpy(np.stack([x[1] for x in selected])).float()
        self.distortions=torch.zeros((len(selected),5),dtype=torch.float32)

    def load_egocar_mask(self):
        # No manual/GT foreground template enters the RGB-only reconstruction.
        self.egocar_mask=torch.zeros(self.load_size,dtype=torch.float32)

    def load_ego_template(self):
        self.ego_template=None


class RGBPixelSource(ScenePixelSource):
    def __init__(self,dataset_name,pixel_data_config,data_path,start_timestep,end_timestep,
                 device=torch.device('cpu'),**kwargs):
        super().__init__(dataset_name,pixel_data_config,device=device)
        self.camera_data={}
        self.data_path=data_path
        self.start_timestep=start_timestep
        self.end_timestep=end_timestep
        self.load_data()

    def load_cameras(self):
        self._timesteps=torch.arange(self.start_timestep,self.end_timestep)
        self.register_normalized_timestamps()
        for idx,cam_id in enumerate(self.camera_list):
            camera=RGBCameraData(dataset_name=self.dataset_name,data_path=self.data_path,
                cam_id=cam_id,start_timestep=self.start_timestep,end_timestep=self.end_timestep,
                load_dynamic_mask=True,load_sky_mask=True,load_road_mask=True,
                downscale_when_loading=1,undistort=False,buffer_downscale=self.buffer_downscale,
                device=self.device)
            camera.load_time(self.normalized_time)
            camera.set_unique_ids(unique_cam_idx=idx,
                unique_img_idx=torch.arange(len(camera),device=self.device)*len(self.camera_list)+idx)
            self.camera_data[cam_id]=camera

    def load_objects(self):
        folder=Path(self.data_path)
        info=json.loads((folder/'instances/instances_info.json').read_text())
        n=self.end_timestep-self.start_timestep
        count=len(info)
        poses=np.broadcast_to(np.eye(4),(n,count,4,4)).copy()
        sizes=np.zeros((count,3),np.float32)
        visible=np.zeros((n,count),bool)
        types=np.zeros(count,np.int64)
        mapping={'Vehicle':ModelType.RigidNodes,'Pedestrian':ModelType.DeformableNodes,
                 'Cyclist':ModelType.DeformableNodes,'TrafficLight':ModelType.TrafficLightNodes}
        for key,item in info.items():
            idx=int(key)
            types[idx]=mapping[item['class_name']]
            ann=item['frame_annotations']
            sizes[idx]=np.max(np.asarray(ann['box_size']),axis=0)
            for fid,pose in zip(ann['frame_idx'],ann['obj_to_world']):
                if self.start_timestep<=fid<self.end_timestep:
                    poses[fid-self.start_timestep,idx]=pose
                    visible[fid-self.start_timestep,idx]=True
        self.instances_pose=torch.from_numpy(poses).float()
        self.instances_size=torch.from_numpy(sizes)
        self.per_frame_instance_mask=torch.from_numpy(visible)
        self.instances_model_types=torch.from_numpy(types)
        self.instances_true_id=torch.arange(count)
        self.smpl_human_all=None

    def load_specified_cameras(self,*args,**kwargs):
        raise NotImplementedError('Only cameras in the RGB prediction manifest may be rendered')
