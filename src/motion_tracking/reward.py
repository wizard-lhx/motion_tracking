import torch

from active_adaptation.envs.utils import find_bodies, find_sensor_bodies
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


def _desired_anchor_pos(anchor_pos_w, ref_anchor_pos_w):
    return torch.stack(
        [anchor_pos_w[:, 0], anchor_pos_w[:, 1], ref_anchor_pos_w[:, 2]],
        dim=-1,
    )


class root_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.3 ** 2, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = (
            self.command_manager.motion_anchor_pos_w
            - self.command_manager.robot_anchor_pos_w
        )
        return torch.exp(-error.square().sum(dim=-1, keepdim=True) / self.sigma)


class root_quat_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.4 ** 2, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.motion_anchor_quat_w
        current = self.command_manager.robot_anchor_quat_w
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
        target = self.command_manager.anchored_body_pos_w[:, self.body_ids]
        current = self.asset.data.body_link_pos_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1).mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)

    def debug_draw(self) -> None:
        if self.env.backend == "mjlab":
            # Use command.debug_draw() for the raw global-reference ghost.
            # Comment this return to draw the reward-aligned ghost instead.
            return
            self._debug_draw_mjlab_ghost()
            return
        target = self.command_manager.anchored_body_pos_w[:, self.body_ids]
        self.env.debug_draw.point(
            target.reshape(-1, 3),
            color=(1.0, 0.0, 0.0, 1.0),
            size=20,
        )

    def _debug_draw_mjlab_ghost(self) -> None:
        if not self.env.sim.has_gui():
            return
        scene = self.env.sim.viewer.scene
        if scene is None:
            return

        root_pos = self.command_manager.anchored_body_pos_w[:, 0]
        root_quat = self.command_manager.anchored_body_quat_w[:, 0]

        free_joint_q_adr = self.asset.data.indexing.free_joint_q_adr
        joint_q_adr = self.asset.data.indexing.joint_q_adr
        for env_id in scene.get_env_indices(self.num_envs):
            qpos = self.env.sim.data.qpos[env_id].clone()
            qpos[free_joint_q_adr] = torch.cat([root_pos[env_id], root_quat[env_id]])
            qpos[joint_q_adr] = self.command_manager.target_joint_pos[env_id]
            scene.add_ghost_mesh(
                qpos,
                self.env.sim.mj_model,
                alpha=0.35,
                label=f"tracking_target_{env_id}",
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
        target = self.command_manager.anchored_body_quat_w[:, self.body_ids]
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


class undesired_self_contacts(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        threshold: float = 1.0,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_sensor_bodies(
            self.asset,
            self.contact_sensor,
            body_names,
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.threshold = threshold

    def _compute(self) -> torch.Tensor:
        data = self.contact_sensor.data
        force_history = getattr(data, "net_forces_w_history", None)
        if force_history is not None:
            forces = force_history[:, :, self.body_ids].norm(dim=-1).max(dim=1).values
        else:
            force_history = getattr(data, "force_history", None)
            if force_history is not None:
                forces = force_history[:, self.body_ids].norm(dim=-1).max(dim=2).values
            else:
                forces = data.net_forces_w[:, self.body_ids].norm(dim=-1)
        return -(forces > self.threshold).float().sum(dim=-1, keepdim=True)
