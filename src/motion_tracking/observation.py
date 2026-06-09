import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.utils import find_bodies
import active_adaptation.utils.symmetry as sym_utils
from active_adaptation.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)


def _uniform_noise(value: torch.Tensor, amplitude: float) -> torch.Tensor:
    if amplitude <= 0.0:
        return value
    return value + torch.empty_like(value).uniform_(-amplitude, amplitude)


class teacher_command(Observation):
    def compute(self) -> torch.Tensor:
        return self.command_manager.teacher_command


class student_command(Observation):
    def compute(self) -> torch.Tensor:
        if hasattr(self.command_manager, "student_command"):
            return self.command_manager.student_command
        return self.command_manager.command[:, :3]

    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3, device=self.device),
            signs=torch.tensor([1.0, -1.0, -1.0], device=self.device),
        )


class motion_anchor_pos_b(Observation):
    def __init__(self, env, noise_amplitude: float = 0.0):
        super().__init__(env)
        self.noise_amplitude = float(noise_amplitude)

    def compute(self) -> torch.Tensor:
        error_w = (
            self.command_manager.motion_anchor_pos_w
            - self.command_manager.robot_anchor_pos_w
        )
        error_b = quat_rotate_inverse(
            self.command_manager.robot_anchor_quat_w,
            error_w,
        )
        return _uniform_noise(error_b, self.noise_amplitude)


class motion_anchor_ori_b(Observation):
    def __init__(self, env, noise_amplitude: float = 0.0):
        super().__init__(env)
        self.noise_amplitude = float(noise_amplitude)

    def compute(self) -> torch.Tensor:
        error_quat = quat_mul(
            quat_conjugate(self.command_manager.robot_anchor_quat_w),
            self.command_manager.motion_anchor_quat_w,
        )
        rot6d = matrix_from_quat(error_quat)[..., :2].reshape(self.num_envs, 6)
        return _uniform_noise(rot6d, self.noise_amplitude)


class root_lin_vel_b(Observation):
    def __init__(self, env, noise_amplitude: float = 0.0):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.noise_amplitude = float(noise_amplitude)

    def compute(self) -> torch.Tensor:
        value = quat_rotate_inverse(
            self.asset.data.root_link_quat_w,
            self.asset.data.root_com_lin_vel_w,
        )
        return _uniform_noise(value, self.noise_amplitude)


class root_ang_vel_b(Observation):
    def __init__(self, env, noise_amplitude: float = 0.0):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.noise_amplitude = float(noise_amplitude)

    def compute(self) -> torch.Tensor:
        value = quat_rotate_inverse(
            self.asset.data.root_link_quat_w,
            self.asset.data.root_com_ang_vel_w,
        )
        return _uniform_noise(value, self.noise_amplitude)


class joint_pos_bm(Observation):
    def __init__(self, env, noise_amplitude: float = 0.0):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.noise_amplitude = float(noise_amplitude)

    def compute(self) -> torch.Tensor:
        value = self.asset.data.joint_pos - self.asset.data.default_joint_pos
        return _uniform_noise(value, self.noise_amplitude)


class joint_vel_bm(Observation):
    def __init__(self, env, noise_amplitude: float = 0.0):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.noise_amplitude = float(noise_amplitude)

    def compute(self) -> torch.Tensor:
        return _uniform_noise(
            self.asset.data.joint_vel,
            self.noise_amplitude,
        )


class last_action_bm(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.action_manager.action_buf[:, 0]


class body_pose_b(Observation):
    def __init__(
        self,
        env,
        body_names: str | list[str],
        anchor_body_name: str,
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        anchor_ids, _ = find_bodies(self.asset, anchor_body_name)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.anchor_id = anchor_ids[0]

    def compute(self) -> torch.Tensor:
        body_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        body_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids]
        anchor_pos_w = self.asset.data.body_link_pos_w[:, self.anchor_id]
        anchor_quat_w = self.asset.data.body_link_quat_w[:, self.anchor_id]

        rel_pos = quat_rotate_inverse(
            anchor_quat_w.unsqueeze(1),
            body_pos_w - anchor_pos_w.unsqueeze(1),
        )
        rel_quat = quat_mul(
            quat_conjugate(anchor_quat_w).unsqueeze(1),
            body_quat_w,
        )
        rel_rotmat = matrix_from_quat(rel_quat).reshape(
            self.num_envs,
            len(self.body_ids),
            9,
        )
        return torch.cat([rel_pos, rel_rotmat], dim=-1).reshape(self.num_envs, -1)


class body_pose_bm(body_pose_b):
    def __init__(
        self,
        env,
        body_names: str | list[str],
        anchor_body_name: str,
    ):
        Observation.__init__(self, env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(
            body_names,
            preserve_order=True,
        )
        anchor_ids, _ = find_bodies(self.asset, anchor_body_name)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.anchor_id = anchor_ids[0]

    def compute(self) -> torch.Tensor:
        body_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        body_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids]
        anchor_pos_w = self.asset.data.body_link_pos_w[:, self.anchor_id]
        anchor_quat_w = self.asset.data.body_link_quat_w[:, self.anchor_id]

        rel_pos = quat_rotate_inverse(
            anchor_quat_w.unsqueeze(1),
            body_pos_w - anchor_pos_w.unsqueeze(1),
        )
        rel_quat = quat_mul(
            quat_conjugate(anchor_quat_w).unsqueeze(1),
            body_quat_w,
        )
        rel_rot6d = matrix_from_quat(rel_quat)[..., :2].reshape(
            self.num_envs,
            len(self.body_ids),
            6,
        )
        return torch.cat([rel_pos, rel_rot6d], dim=-1).reshape(self.num_envs, -1)
