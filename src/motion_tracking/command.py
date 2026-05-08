from pathlib import Path
from typing import Sequence

import torch

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
)

from motion_tracking.dataset import DATASET_DIR, MotionDataset


class MotionTrackingCommand(Command):
    namespace = "motion_tracking"

    def __init__(
        self,
        env,
        data_path: str = "100style",
        future_steps: Sequence[int] = (0, 12, 24, 36),
        random_start: bool = False,
    ):
        super().__init__(env)

        data_path = Path(data_path)
        if not data_path.exists():
            data_path = DATASET_DIR / data_path
        self.dataset = MotionDataset(data_path, device=self.device)

        self.future_steps = torch.as_tensor(
            future_steps,
            dtype=torch.long,
            device=self.device,
        )
        self.max_future_step = int(self.future_steps.max().item())
        self.random_start = random_start

        self.joint_idx_motion = torch.tensor(
            [self.dataset.joint_names.index(name) for name in self.asset.joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.motion_lengths = self.dataset.lengths.to(self.device)

        self.motion_ids = self.dataset.sample_motion_ids(self.num_envs)
        self.t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cum_error = torch.zeros(self.num_envs, 1, device=self.device)

        self._update_targets()

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        self.motion_ids[env_ids] = self.dataset.sample_motion_ids(len(env_ids))
        if self.random_start:
            self.t[env_ids] = self.dataset.sample_frames(
                self.motion_ids[env_ids],
                max_offset=self.max_future_step,
            )
        else:
            self.t[env_ids] = 0

        motion = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids])
        self.init_joint_pos[env_ids] = motion.joint_pos[:, 0, self.joint_idx_motion]

        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] = (
            self.env.scene.get_spawn_origins(env_ids) + motion.root_pos_w[:, 0]
        )
        init_root_state[:, 3:7] = motion.root_quat_w[:, 0]
        return init_root_state

    def reset(self, env_ids: torch.Tensor) -> None:
        joint_pos = self.init_joint_pos[env_ids]
        self.asset.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(joint_pos),
            slice(None),
            env_ids,
        )
        self.asset.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._cum_error[env_ids] = 0.0

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.target_pos_b.reshape(self.num_envs, -1),
                self.relative_quat.reshape(self.num_envs, -1),
                self.target_joint_pos_future.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

    def _update_targets(self) -> None:
        self._motion = self.dataset.get_slice(
            self.motion_ids,
            self.t,
            offsets=self.future_steps,
        )

        origins = self.env.scene.env_origins.to(self.device).reshape(
            self.num_envs,
            1,
            3,
        )
        self.target_pos_w = self._motion.root_pos_w + origins
        self.target_pos_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            self.target_pos_w - self.asset.data.root_pos_w.unsqueeze(1),
        )

        self.target_quat_w = self._motion.root_quat_w
        self.relative_quat = quat_mul(
            quat_conjugate(self.asset.data.root_link_quat_w).unsqueeze(1),
            self.target_quat_w,
        )

        self.target_joint_pos_future = self._motion.joint_pos[:, :, self.joint_idx_motion]
        self.target_joint_pos = self.target_joint_pos_future[:, 0]

    def update(self) -> None:
        self._update_targets()

        root_error = self.target_pos_w[:, 0] - self.asset.data.root_pos_w
        self._cum_error.mul_(0.98).add_(
            root_error.norm(dim=-1, keepdim=True) * self.env.step_dt
        )

    def step(self) -> None:
        max_t = (
            self.motion_lengths[self.motion_ids] - self.max_future_step - 1
        ).clamp_min(0)
        self.t = torch.minimum(self.t + 1, max_t)
        self._update_targets()

    def debug_draw(self) -> None:
        target_pos_w = self.target_pos_w[:, 0]
        robot_pos_w = self.asset.data.root_pos_w
        self.env.debug_draw.point(target_pos_w, color=(1.0, 0.0, 0.0, 1.0))
        self.env.debug_draw.vector(
            robot_pos_w,
            target_pos_w - robot_pos_w,
            color=(0.0, 0.2, 1.0, 1.0),
        )
