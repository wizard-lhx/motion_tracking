import torch

from active_adaptation.envs.mdp.terminations.base import Termination
from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)


class body_pos_z_error(Termination):
    def __init__(
        self,
        env,
        body_names: str | list[str],
        threshold: float = 0.25,
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.threshold = threshold

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        target = self.command_manager.anchored_body_pos_w[:, self.body_ids]
        error_z = target[..., 2] - self.asset.data.body_link_pos_w[:, self.body_ids, 2]
        return (error_z.abs() > self.threshold).any(dim=-1, keepdim=True)


class anchor_pos_z_error(Termination):
    def __init__(self, env, threshold: float = 0.25):
        super().__init__(env)
        self.threshold = threshold

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        error_z = (
            self.command_manager.motion_anchor_pos_w[:, 2]
            - self.command_manager.robot_anchor_pos_w[:, 2]
        )
        return (error_z.abs() > self.threshold).reshape(self.num_envs, 1)


class anchor_ori_error(Termination):
    def __init__(self, env, threshold: float = 0.8):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.threshold = threshold

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        target = quat_mul(
            self.command_manager.anchor_yaw_delta_quat,
            self.command_manager.motion_anchor_quat_w,
        )
        error_quat = quat_mul(
            target,
            quat_conjugate(self.command_manager.robot_anchor_quat_w),
        )
        error = axis_angle_from_quat(error_quat).norm(dim=-1, keepdim=True)
        return error > self.threshold


class anchor_projected_gravity_error(Termination):
    def __init__(self, env, threshold: float = 0.8):
        super().__init__(env)
        self.threshold = threshold

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        gravity_w = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        gravity_w[:, 2] = -1.0
        motion_gravity_b = quat_rotate_inverse(
            self.command_manager.motion_anchor_quat_w,
            gravity_w,
        )
        robot_gravity_b = quat_rotate_inverse(
            self.command_manager.robot_anchor_quat_w,
            gravity_w,
        )
        error = (motion_gravity_b[:, 2] - robot_gravity_b[:, 2]).abs()
        return (error > self.threshold).reshape(self.num_envs, 1)


class motion_finished(Termination):
    def __init__(self, env):
        super().__init__(env, is_timeout=True)

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        max_t = (
            self.command_manager.motion_lengths[self.command_manager.motion_ids]
            - self.command_manager.max_future_step
            - 1
        ).clamp_min(0)
        return (self.command_manager.t >= max_t).reshape(self.num_envs, 1)
