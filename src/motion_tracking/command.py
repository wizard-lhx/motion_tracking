import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp.commands.base import Command
from motion_tracking.dataset import MotionDataset
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat
)


class MotionTrackingCommand(Command):
    def __init__(self, env, data_path: str):
        super().__init__(env)
        
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        pass
    
    def reset(self, env_ids):
        self.t[env_ids] = 0

    @property
    def command(self):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass
