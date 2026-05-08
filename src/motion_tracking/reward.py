import torch

from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import axis_angle_from_quat


class root_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.25, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = self.command_manager.target_pos_w[:, 0] - self.asset.data.root_pos_w
        return torch.exp(-error.square().sum(dim=-1, keepdim=True) / self.sigma)


class root_quat_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.5, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = axis_angle_from_quat(self.command_manager.relative_quat[:, 0])
        return torch.exp(-error.square().sum(dim=-1, keepdim=True) / self.sigma)


class joint_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.5, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = self.command_manager.target_joint_pos - self.asset.data.joint_pos
        return torch.exp(-error.square() / self.sigma).mean(dim=-1, keepdim=True)
