"""CPU contract tests; these do not claim model-inference or rendering coverage."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.rgb_only.common import (camera_table, merge_camera_tracks, read_scene,
                                  resized_depth, save_npz, unproject, write_json)
from tools.rgb_only.pipeline import index_scene, parse_scene_list
from tools.rgb_only.assembly import assemble, validate_scene


class RGBContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cfg=dict(rgb_root=str(self.root/'source'),output_root=str(self.root/'pred'),
            cameras=[0],fps=10,image_size=[32,48],reference_camera=0,min_object_pixels=4,
            min_track_frames=3,cross_camera_distance=1.5,moving_threshold_m=0.3,
            point_stride=1,points_per_image=500,points_per_object_view=100)
        images=self.root/'source/621/images'
        images.mkdir(parents=True)
        for fid in range(30,35):
            Image.new('RGB',(48,32),(120,130,140)).save(images/f'{fid:03d}_0.jpg')
        # Poison GT files: indexing/assembly must never parse them or copy them.
        for kind in ('intrinsics','extrinsics','ego_pose','lidar','instances'):
            path=self.root/'source/621'/kind
            path.mkdir()
            (path/'DO_NOT_READ.txt').write_text('not valid calibration or label data')
        self.folder=index_scene(self.cfg,dict(scene_idx=621,start=30,end=34))

    def tearDown(self):
        self.temp.cleanup()

    def fixture_predictions(self):
        folder=self.folder
        K=np.array([[20,0,24],[0,20,16],[0,0,1]],np.float32)
        save_npz(folder/'predicted_cameras.npz',frame_ids=np.arange(30,35),camera_ids=np.array(['0']*5),
                 K=np.repeat(K[None],5,0),c2w=np.repeat(np.eye(4)[None],5,0),image_size=np.array([32,48]))
        for fid in range(30,35):
            key=f'{fid:03d}_0'
            save_npz(folder/'moge_depth'/(key+'.npz'),depth=np.full((32,48),5,np.float32),
                     valid=np.ones((32,48),bool))
            labels=np.zeros((32,48),np.uint16)
            offset=2*(fid-30)
            labels[10:18,8+offset:16+offset]=1
            path=folder/'sam_tracks/0/mask_data'/f'mask_{fid:06d}.npy'
            path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,labels)
            write_json(folder/'sam_tracks/0/json_data'/f'mask_{fid:06d}.json',
                       dict(labels={'1':dict(class_name='car')}))
            for kind in ('road_masks','sky_masks'):
                mask=np.zeros((32,48),np.uint8)
                if kind=='road_masks': mask[24:]=255
                else: mask[:2]=255
                path=folder/kind/(key+'.png')
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(mask).save(path)
        write_json(folder/'traffic_predictions.json',dict(instances=[]))

    def test_rgb_index_preserves_nonzero_frame_ids_and_ignores_gt(self):
        scene=read_scene(self.folder)
        self.assertEqual([f['frame_id'] for f in scene['frames']],[30,31,32,33,34])
        self.assertEqual(scene['sensor_inputs'],['rgb'])
        for kind in ('intrinsics','extrinsics','ego_pose','lidar'):
            self.assertFalse((self.folder/kind).exists())
        self.assertTrue((self.folder/'images/030_0.jpg').is_symlink())
        index_scene(self.cfg,dict(scene_idx=621,start=30,end=34))
        with self.assertRaises(RuntimeError):
            index_scene(self.cfg,dict(scene_idx=621,start=31,end=34))

    def test_camera_depth_convention_and_invalid_resize(self):
        K=np.array([[2,0,0],[0,2,0],[0,0,1.]])
        pose=np.eye(4);pose[:3,3]=[1,2,3]
        points,y,x=unproject(np.full((2,2),4.),K,pose)
        np.testing.assert_allclose(points,[[1,2,7],[3,2,7],[1,4,7],[3,4,7]])
        values,valid=resized_depth(np.array([[10,np.nan],[10,10.]]),np.array([[1,0],[1,1]],bool),(4,4))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values[~valid]==0).all())
        np.testing.assert_allclose(values[valid],10)

    def test_cross_camera_merges_cannot_collapse_same_camera_objects(self):
        def track(offset):
            return dict(class_name='Vehicle',centers={f:np.array([f+offset,0,0.]) for f in range(5)})
        obs={('0',1):track(0),('1',5):track(0.05),('0',2):track(0.1)}
        groups=merge_camera_tracks(obs,threshold=1,min_shared=3)
        self.assertEqual(groups[('0',1)],groups[('1',5)])
        self.assertNotEqual(groups[('0',1)],groups[('0',2)])

    def test_assembly_creates_dynamic_nodes_and_separate_road_background(self):
        self.fixture_predictions()
        assemble(self.cfg,self.folder)
        result=validate_scene(self.folder,self.cfg)
        self.assertEqual(result['instances'],1)
        info=json.loads((self.folder/'instances/instances_info.json').read_text())
        self.assertEqual(info['0']['class_name'],'Vehicle')
        self.assertEqual(info['0']['frame_annotations']['frame_idx'],list(range(30,35)))
        with np.load(self.folder/'seed_points.npz') as z:
            self.assertTrue((z['instance_ids']==0).any())
            self.assertTrue(z['road'].any())
            self.assertTrue(((z['instance_ids']==-1)&~z['road']).any())
        with np.load(self.folder/'instance_masks/030_0.npz') as z:
            self.assertTrue((z['ids']==0).any())

    def test_static_detected_vehicle_is_not_marked_moving(self):
        self.fixture_predictions()
        first=np.load(self.folder/'sam_tracks/0/mask_data/mask_000030.npy')
        for fid in range(31,35):
            np.save(self.folder/'sam_tracks/0/mask_data'/f'mask_{fid:06d}.npy',first)
        assemble(self.cfg,self.folder)
        info=json.loads((self.folder/'instances/instances_info.json').read_text())
        self.assertEqual(info,{})
        self.assertFalse(np.array(Image.open(self.folder/'dynamic_masks/all/030_0.png')).any())

    def test_pixel_source_cpu_reads_predicted_cameras_and_depth(self):
        import torch
        from omegaconf import OmegaConf
        from datasets.waymo.rgb_sourceloader import RGBPixelSource
        self.fixture_predictions()
        assemble(self.cfg,self.folder)
        config=OmegaConf.create(dict(cameras=[0],downscale=1,load_objects=True,load_smpl=False,
            sampler=dict(buffer_downscale=4,buffer_ratio=0.5,start_enhance_weight=3)))
        source=RGBPixelSource('waymo',config,str(self.folder),30,35,device=torch.device('cpu'))
        image,camera=source.get_image(0)
        self.assertEqual(tuple(image['pixels'].shape),(32,48,3))
        self.assertEqual(tuple(image['lidar_depth_map'].shape),(32,48))
        self.assertEqual(source.instances_pose.shape,(5,1,4,4))
        np.testing.assert_allclose(source.camera_data[0].intrinsics[0].numpy(),
                                   [[20,0,24],[0,20,16],[0,0,1]])
        np.testing.assert_allclose(source.camera_data[0].cam_to_worlds[0].numpy(),np.eye(4))

    def test_validation_rejects_corrupt_depth_and_camera_coverage(self):
        self.fixture_predictions()
        assemble(self.cfg,self.folder)
        save_npz(self.folder/'moge_depth/030_0.npz',depth=np.ones((3,3)),valid=np.ones((3,3),bool))
        with self.assertRaisesRegex(ValueError,'grid mismatch'):
            validate_scene(self.folder,self.cfg)
        save_npz(self.folder/'predicted_cameras.npz',frame_ids=np.array([30]),camera_ids=np.array(['0']),
                 K=np.eye(3)[None],c2w=np.eye(4)[None],image_size=np.array([32,48]))
        with self.assertRaisesRegex(ValueError,'exactly'):
            camera_table(self.folder/'predicted_cameras.npz',read_scene(self.folder))

    def test_reference_scene_list_parser(self):
        path=self.root/'scenes.csv'
        path.write_text('# scene_id, name, start, end\n172,seg138251,30,180\n621,seg584622,0,140\n')
        self.assertEqual(parse_scene_list(path),[dict(scene_idx=172,start=30,end=180),dict(scene_idx=621,start=0,end=140)])
        path.write_text('scene-0531\n')
        with self.assertRaises(ValueError):parse_scene_list(path)


if __name__=='__main__':
    unittest.main()
