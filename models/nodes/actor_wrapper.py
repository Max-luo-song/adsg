"""
RigidActorNodes - Lightweight wrapper for per-instance tracking.

This module provides a proxy wrapper around a unified RigidNodes model,
enabling per-instance tracking and management while maintaining memory efficiency.

Design Pattern: Proxy Pattern
- External Interface: Mimics RigidActorNodes API from scene_reconstruction
- Internal Implementation: Delegates to a single shared RigidNodes instance
- Index Mapping: Translates instance_id to the correct index in shared tensors
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger()


class RigidActorNodes:
    """
    Proxy wrapper for tracking individual rigid actors in a shared RigidNodes model.
    
    This class provides a lightweight wrapper that gives the appearance of 
    independent RigidNodes objects while actually delegating all operations 
    to a shared RigidNodes instance.
    
    Args:
        rigid_nodes_ref: Reference to the shared RigidNodes model
        instance_id: The unique ID of this instance (used as index)
        class_name: Name of the class (for logging/tracking)
    
    Attributes:
        obj_track_name: Human-readable name (e.g., "obj_001")
        class_prefix: Prefix for logs (e.g., "RigidActorNodes_obj_001_#")
    """
    
    def __init__(
        self,
        rigid_nodes_ref,
        instance_id: int,
        class_name: str = "RigidActorNodes",
    ):
        """Initialize the RigidActorNodes wrapper."""
        self.rigid_nodes = rigid_nodes_ref
        self.instance_id = instance_id
        self.obj_track_name = f"obj_{instance_id:03d}"
        self.class_prefix = class_name + f"_{self.obj_track_name}" + "#"
        
    @property
    def optim_steps(self) -> int:
        """Get current optimization step count from shared RigidNodes."""
        return self.rigid_nodes.step
    
    def optim_update(self) -> None:
        """Increment the optimization step counter."""
        self.rigid_nodes.step += 1
    
    def is_cur_frame_visiable(self) -> bool:
        """
        Check if this instance is visible in the current frame.
        
        Returns:
            bool: True if visible, False otherwise
        """
        if not hasattr(self.rigid_nodes, 'cur_frame'):
            logger.warning(f"RigidNodes does not have cur_frame attribute")
            return False
        
        # instances_fv shape: (num_frame, num_instance)
        visibility = self.rigid_nodes.instances_fv[
            self.rigid_nodes.cur_frame,
            self.instance_id
        ]
        return visibility.item()
    
    def total_visiable_nums(self) -> int:
        """
        Count total number of frames this instance is visible in.
        
        Returns:
            int: Number of visible frames
        """
        # instances_fv shape: (num_frame, num_instance)
        visibility = self.rigid_nodes.instances_fv[:, self.instance_id]
        return visibility.sum().item()
    
    def get_instance_pose(self, frame_id: Optional[int] = None) -> torch.Tensor:
        """
        Get the pose matrix for this instance at a specific frame.
        
        Args:
            frame_id: Frame index (None uses cur_frame)
            
        Returns:
            torch.Tensor: Pose matrix (4, 4)
        """
        if frame_id is None:
            frame_id = self.rigid_nodes.cur_frame
        
        # instances_pose shape: (num_frame, num_instance, 4, 4)
        return self.rigid_nodes.instances_pose[frame_id, self.instance_id]
    
    def get_instance_size(self) -> torch.Tensor:
        """
        Get the size (bounding box) of this instance.
        
        Returns:
            torch.Tensor: Size vector (3,)
        """
        # instances_size shape: (num_instance, 3)
        return self.rigid_nodes.instances_size[self.instance_id]
    
    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"RigidActorNodes(obj_track_name='{self.obj_track_name}', instance_id={self.instance_id})"
