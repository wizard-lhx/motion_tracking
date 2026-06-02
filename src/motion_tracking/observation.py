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


class teacher_command(Observation):
    def compute(self) -> torch.Tensor:
        return self.command_manager.teacher_command


class student_command(Observation):
    def compute(self) -> torch.Tensor:
        return self.command_manager.student_command

    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3, device=self.device),
            signs=torch.tensor([1.0, -1.0, -1.0], device=self.device),
        )


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
        rel_rotmat = matrix_from_quat(rel_quat).reshape(self.num_envs, len(self.body_ids), 9)
        return torch.cat([rel_pos, rel_rotmat], dim=-1).reshape(self.num_envs, -1)
