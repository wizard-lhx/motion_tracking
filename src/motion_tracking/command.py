from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    yaw_quat,
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
        motion_id: int | None = None,
        adaptive_sampling: bool = False,
        adaptive_bin_seconds: float = 1.0,
        adaptive_kernel_size: int = 3,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
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
        self.motion_id = None if motion_id is None else int(motion_id)
        self.adaptive_sampling = adaptive_sampling
        if self.adaptive_sampling:
            self.adaptive_kernel_size = int(adaptive_kernel_size)
            self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
            self.adaptive_alpha = float(adaptive_alpha)
            self.frames_per_bin = max(
                1,
                round(float(adaptive_bin_seconds) / self.env.step_dt),
            )
            self._build_adaptive_bins(adaptive_lambda)

        self.joint_idx_motion = torch.tensor(
            [self.dataset.joint_names.index(name) for name in self.asset.joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.body_idx_motion = torch.tensor(
            [self.dataset.body_names.index(name) for name in self.asset.body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.motion_lengths = self.dataset.lengths.to(self.device)

        if self.motion_id is None:
            self.motion_ids = self.dataset.sample_motion_ids(self.num_envs)
        else:
            self.motion_ids = torch.full(
                (self.num_envs,),
                self.motion_id,
                dtype=torch.long,
                device=self.device,
            )
        self.t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cum_error = torch.zeros(self.num_envs, 1, device=self.device)

        self._update_targets()

    def _build_adaptive_bins(self, adaptive_lambda: float) -> None:
        valid_lengths = (self.dataset.lengths - self.max_future_step).clamp_min(1)
        num_bins = torch.div(
            valid_lengths + self.frames_per_bin - 1,
            self.frames_per_bin,
            rounding_mode="floor",
        ).clamp_min(1)
        offsets = torch.zeros(self.dataset.num_motions + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(num_bins, dim=0)

        self.valid_motion_lengths = valid_lengths.to(self.device)
        self.num_bins_per_motion = num_bins.to(self.device)
        self.motion_bin_offsets = offsets.to(self.device)
        self.total_bins = int(offsets[-1].item())
        self.bin_failed_count = torch.zeros(self.total_bins, device=self.device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)

        self.max_bins_per_motion = int(num_bins.max().item())
        local_bins = torch.arange(self.max_bins_per_motion, device=self.device)
        self.bin_mask_by_motion = local_bins.unsqueeze(0) < self.num_bins_per_motion.unsqueeze(1)
        bin_indices = self.motion_bin_offsets[:-1].unsqueeze(1) + local_bins
        self.bin_index_by_motion = torch.where(
            self.bin_mask_by_motion,
            bin_indices,
            torch.zeros_like(bin_indices),
        )

        kernel = torch.tensor(
            [float(adaptive_lambda) ** i for i in range(self.adaptive_kernel_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.adaptive_kernel = kernel / kernel.sum().clamp_min(1e-8)

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.motion_id is None:
            self.motion_ids[env_ids] = self.dataset.sample_motion_ids(len(env_ids))
        else:
            self.motion_ids[env_ids] = self.motion_id
        if self.adaptive_sampling:
            self.t[env_ids] = self._sample_adaptive_frames(self.motion_ids[env_ids])
        elif self.random_start:
            self.t[env_ids] = self.dataset.sample_frames(
                self.motion_ids[env_ids],
                max_offset=self.max_future_step,
            )
        else:
            self.t[env_ids] = 0

        motion = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids])
        self.init_joint_pos[env_ids] = motion.joint_pos[:, 0, self.joint_idx_motion]
        self.init_joint_vel[env_ids] = motion.joint_vel[:, 0, self.joint_idx_motion]

        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] = (
            self.env.scene.get_spawn_origins(env_ids) + motion.root_pos_w[:, 0]
        )
        init_root_state[:, 3:7] = motion.root_quat_w[:, 0]
        return init_root_state

    def reset(self, env_ids: torch.Tensor) -> None:
        joint_pos = self.init_joint_pos[env_ids]
        joint_vel = self.init_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            slice(None),
            env_ids,
        )
        if self.env.backend == "mujoco":
            self.asset.set_joint_position_target(joint_pos)
        else:
            self.asset.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._cum_error[env_ids] = 0.0

    @property
    def command(self) -> torch.Tensor:
        return self.teacher_command

    @property
    def teacher_command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.root_pos_error_b,
                self.root_ori_error,
                self.target_joint_pos_future.reshape(self.num_envs, -1),
                self.target_joint_vel_future.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

    @property
    def student_command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.cmd_linvel_b[:, :2],
                self.cmd_yawvel_b,
            ],
            dim=-1,
        )

    @property
    def root_pos_error_b(self) -> torch.Tensor:
        error_w = self.target_body_pos_w[:, 0, 0] - self.asset.data.body_link_pos_w[:, 0]
        return quat_rotate_inverse(self.asset.data.body_link_quat_w[:, 0], error_w)

    @property
    def root_ori_error(self) -> torch.Tensor:
        error_quat = quat_mul(
            quat_conjugate(self.asset.data.body_link_quat_w[:, 0]),
            self.target_body_quat_w[:, 0, 0],
        )
        return axis_angle_from_quat(error_quat)

    def _update_targets(self) -> None:
        self._motion = self.dataset.get_slice(
            self.motion_ids,
            self.t,
            offsets=self.future_steps,
        )

        origins = self.env.scene.env_origins.to(self.device).reshape(self.num_envs, 1, 1, 3)
        self.target_body_pos_w = self._motion.body_pos_w[:, :, self.body_idx_motion] + origins
        self.target_body_quat_w = self._motion.body_quat_w[:, :, self.body_idx_motion]
        self.target_body_rotmat_w = matrix_from_quat(self.target_body_quat_w)
        self.target_body_lin_vel_w = self._motion.body_lin_vel_w[:, :, self.body_idx_motion]
        self.target_body_ang_vel_w = self._motion.body_ang_vel_w[:, :, self.body_idx_motion]

        self.target_joint_pos_future = self._motion.joint_pos[:, :, self.joint_idx_motion]
        self.target_joint_vel_future = self._motion.joint_vel[:, :, self.joint_idx_motion]
        self.target_joint_pos = self.target_joint_pos_future[:, 0]
        self.target_joint_vel = self.target_joint_vel_future[:, 0]

        root_yaw_quat = yaw_quat(self._motion.root_quat_w[:, 0])
        self.cmd_linvel_b = quat_rotate_inverse(
            root_yaw_quat,
            self._motion.root_lin_vel_w[:, 0],
        )
        root_ang_vel_b = quat_rotate_inverse(
            root_yaw_quat,
            self._motion.root_ang_vel_w[:, 0],
        )
        self.cmd_yawvel_b = root_ang_vel_b[:, 2:3]

    def update(self) -> None:
        self._update_targets()

        root_error = self.target_body_pos_w[:, 0, 0] - self.asset.data.body_link_pos_w[:, 0]
        self._cum_error.mul_(0.98).add_(
            root_error.norm(dim=-1, keepdim=True) * self.env.step_dt
        )

    def step(self) -> None:
        if self.adaptive_sampling:
            self._update_adaptive_failures()

        max_t = (
            self.motion_lengths[self.motion_ids] - self.max_future_step - 1
        ).clamp_min(0)
        self.t = torch.minimum(self.t + 1, max_t)
        self._update_targets()

    def _sample_adaptive_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        bin_indices = self.bin_index_by_motion[motion_ids]
        bin_mask = self.bin_mask_by_motion[motion_ids]
        num_bins = self.num_bins_per_motion[motion_ids].unsqueeze(1)

        scores = self.bin_failed_count[bin_indices] + self.adaptive_uniform_ratio / num_bins
        scores = torch.where(bin_mask, scores, torch.zeros_like(scores))
        if self.adaptive_kernel_size > 1:
            last_scores = scores.gather(1, num_bins - 1)
            scores = torch.where(bin_mask, scores, last_scores)
            padded = F.pad(
                scores.unsqueeze(1),
                (0, self.adaptive_kernel_size - 1),
                mode="replicate",
            )
            scores = F.conv1d(
                padded,
                self.adaptive_kernel.reshape(1, 1, -1),
            ).squeeze(1)
            scores = torch.where(bin_mask, scores, torch.zeros_like(scores))

        probs = scores / scores.sum(dim=1, keepdim=True)
        sampled_bins = torch.multinomial(probs, 1).squeeze(1)

        bin_starts = sampled_bins * self.frames_per_bin
        bin_ends = torch.minimum(
            bin_starts + self.frames_per_bin,
            self.valid_motion_lengths[motion_ids],
        )
        bin_widths = (bin_ends - bin_starts).clamp_min(1)
        offsets = (torch.rand(len(motion_ids), device=self.device) * bin_widths).long()
        return bin_starts + offsets

    def _update_adaptive_failures(self) -> None:
        self._current_bin_failed.zero_()
        failed_envs = self._failed_envs()
        if failed_envs.any():
            failed_bins = self._phase_bins(
                self.motion_ids[failed_envs],
                self.t[failed_envs],
            )
            self._current_bin_failed.scatter_add_(
                0,
                failed_bins,
                torch.ones_like(failed_bins, dtype=self._current_bin_failed.dtype),
            )

        self.bin_failed_count.lerp_(self._current_bin_failed, self.adaptive_alpha)

    def _failed_envs(self) -> torch.Tensor:
        failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, term in self.env.termination_funcs.items():
            if term.is_timeout:
                continue
            stat = self.env.stats["termination", name]
            failed |= stat.reshape(self.num_envs).bool()
        return failed

    def _phase_bins(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        local_bins = torch.div(frames, self.frames_per_bin, rounding_mode="floor")
        max_bins = self.num_bins_per_motion[motion_ids] - 1
        local_bins = torch.minimum(local_bins, max_bins)
        return self.motion_bin_offsets[motion_ids] + local_bins

    def debug_draw(self) -> None:
        # Uncomment this return when drawing the reward-aligned ghost instead.
        # return
        if self.env.backend != "mjlab" or not self.env.sim.has_gui():
            return

        scene = self.env.sim.viewer.scene
        if scene is None:
            return

        free_joint_q_adr = self.asset.data.indexing.free_joint_q_adr
        joint_q_adr = self.asset.data.indexing.joint_q_adr
        for env_id in scene.get_env_indices(self.num_envs):
            qpos = self.env.sim.data.qpos[env_id].clone()
            qpos[free_joint_q_adr] = torch.cat(
                [
                    self.target_body_pos_w[env_id, 0, 0],
                    self.target_body_quat_w[env_id, 0, 0],
                ],
                dim=-1,
            )
            qpos[joint_q_adr] = self.target_joint_pos[env_id]
            scene.add_ghost_mesh(
                qpos,
                self.env.sim.mj_model,
                alpha=0.35,
                label=f"command_target_{env_id}",
            )
