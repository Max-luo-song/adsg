"""Existing ADSG rendering/losses, initialized from predicted RGB geometry."""
import logging
from models.trainers.scene_graph import MultiTrainer

logger=logging.getLogger(__name__)


class RGBMultiTrainer(MultiTrainer):
    def init_gaussians_from_dataset(self,dataset):
        if dataset.lidar_source is not None:
            raise ValueError('RGB trainer cannot consume LiDAR')
        if 'SMPLNodes' in self.model_config:
            raise ValueError('RGB-only human reconstruction uses DeformableNodes')
        empty=[]
        for name in list(self.gaussian_classes):
            model=self.models[name]
            if name in ('Background','RoadNodes'):
                config=self.model_config[name].get('init',{})
                points,colors=dataset.get_rgb_seed_points(road=name=='RoadNodes',
                    max_points=int(config.get('max_points',800000)))
                model.create_from_pcd(init_means=points,init_colors=colors)
            elif name in ('RigidNodes','DeformableNodes','TrafficLightNodes'):
                objects=dataset.get_init_objects(cur_node_type=name,**self.model_config[name].get('init',{}))
                if self.safe_init_models(model,objects):
                    empty.append(name)
            else:
                raise ValueError(f'Unsupported RGB gaussian class {name}')
        for name in empty:
            del self.models[name]
            del self.model_config[name]
            del self.gaussian_classes[name]
            logger.info('No predicted seeds for %s; removed empty class',name)
        logger.info('Initialized scene graph entirely from RGB-predicted depth, masks and camera poses')
