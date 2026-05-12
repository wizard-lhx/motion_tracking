import torch

from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    yaw_quat,
)
from active_adaptation.envs.mdp.rewards.base import Reward


def _yaw_delta_quat(anchor_quat_w, ref_anchor_quat_w):
    return yaw_quat(quat_mul(anchor_quat_w, quat_conjugate(ref_anchor_quat_w)))


def _desired_pos(
    anchor_pos_w,
    anchor_quat_w,
    ref_anchor_pos_w,
    ref_anchor_quat_w,
    ref_body_pos_w,
):
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
    yaw_delta = _yaw_delta_quat(anchor_quat_w, ref_anchor_quat_w)
    return quat_mul(
        yaw_delta.unsqueeze(1),
        ref_body_quat_w,
    )


class root_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.25, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = self.command_manager.target_body_pos_w[:, 0, 0] - self.asset.data.body_link_pos_w[:, 0]
        return torch.exp(-error.square().sum(dim=-1, keepdim=True) / self.sigma)


class root_quat_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.5, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_quat_w[:, 0, 0]
        current = self.asset.data.body_link_quat_w[:, 0]
        error_quat = quat_mul(target, quat_conjugate(current))
        error = axis_angle_from_quat(error_quat).square().sum(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class joint_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.5, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = self.command_manager.target_joint_pos - self.asset.data.joint_pos
        return torch.exp(-error.square() / self.sigma).mean(dim=-1, keepdim=True)


class body_pos_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 0.3 ** 2,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = _desired_pos(
            self.asset.data.body_link_pos_w[:, 0],
            self.asset.data.body_link_quat_w[:, 0],
            self.command_manager.target_body_pos_w[:, 0, 0],
            self.command_manager.target_body_quat_w[:, 0, 0],
            self.command_manager.target_body_pos_w[:, 0, self.body_ids],
        )
        current = self.asset.data.body_link_pos_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1).mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)

    def debug_draw(self) -> None:
        target = _desired_pos(
            self.asset.data.body_link_pos_w[:, 0],
            self.asset.data.body_link_quat_w[:, 0],
            self.command_manager.target_body_pos_w[:, 0, 0],
            self.command_manager.target_body_quat_w[:, 0, 0],
            self.command_manager.target_body_pos_w[:, 0, self.body_ids],
        )
        self.env.debug_draw.point(
            target.reshape(-1, 3),
            color=(1.0, 0.0, 0.0, 1.0),
            size=20,
        )


class body_rotmat_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 0.4 ** 2,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = _desired_quat(
            self.asset.data.body_link_quat_w[:, 0],
            self.command_manager.target_body_quat_w[:, 0, 0],
            self.command_manager.target_body_quat_w[:, 0, self.body_ids],
        )
        current = self.asset.data.body_link_quat_w[:, self.body_ids]
        error_quat = quat_mul(target, quat_conjugate(current))
        error = axis_angle_from_quat(error_quat).square().sum(dim=-1)
        error = error.mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class body_lin_vel_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 1.0 ** 2,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_lin_vel_w[:, 0, self.body_ids]
        current = self.asset.data.body_lin_vel_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1).mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class body_ang_vel_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 3.14 ** 2,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_ang_vel_w[:, 0, self.body_ids]
        current = self.asset.data.body_ang_vel_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1).mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)
