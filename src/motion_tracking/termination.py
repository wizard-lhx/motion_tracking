import torch

from active_adaptation.envs.mdp.terminations.base import Termination
from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    yaw_quat,
)


def _yaw_delta_quat(anchor_quat_w, ref_anchor_quat_w):
    return yaw_quat(quat_mul(anchor_quat_w, quat_conjugate(ref_anchor_quat_w)))


def _desired_pos(anchor_pos_w, anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w, ref_body_pos_w):
    yaw_delta = _yaw_delta_quat(anchor_quat_w, ref_anchor_quat_w)
    pos_delta = torch.stack(
        [anchor_pos_w[:, 0], anchor_pos_w[:, 1], ref_anchor_pos_w[:, 2]],
        dim=-1,
    )
    return pos_delta.unsqueeze(1) + quat_rotate(
        yaw_delta.unsqueeze(1),
        ref_body_pos_w - ref_anchor_pos_w.unsqueeze(1),
    )


def _desired_quat(anchor_quat_w, ref_anchor_quat_w, ref_body_quat_w):
    return quat_mul(
        _yaw_delta_quat(anchor_quat_w, ref_anchor_quat_w).unsqueeze(1),
        ref_body_quat_w,
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


class anchor_ori_error(Termination):
    def __init__(self, env, threshold: float = 0.8):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.threshold = threshold

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        target = self.command_manager.motion_anchor_quat_w
        current = self.command_manager.robot_anchor_quat_w
        error_quat = quat_mul(target, quat_conjugate(current))
        error = axis_angle_from_quat(error_quat).norm(dim=-1, keepdim=True)
        return error > self.threshold


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
