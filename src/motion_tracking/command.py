from pathlib import Path
from typing import Sequence

import torch

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.utils.math import matrix_from_quat

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

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.motion_id is None:
            self.motion_ids[env_ids] = self.dataset.sample_motion_ids(len(env_ids))
        else:
            self.motion_ids[env_ids] = self.motion_id
        if self.random_start:
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
        return torch.cat(
            [
                self.target_joint_pos_future.reshape(self.num_envs, -1),
                self.target_joint_vel_future.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

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

    def update(self) -> None:
        self._update_targets()

        root_error = self.target_body_pos_w[:, 0, 0] - self.asset.data.body_link_pos_w[:, 0]
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
