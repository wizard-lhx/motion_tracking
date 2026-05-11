from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import MemoryMappedTensor


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "motion_tracking" / "dataset"


@dataclass
class MotionBatch:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor


class MotionDataset:
    def __init__(self, path: str | Path, device: str | torch.device = "cpu"):
        self.path = Path(path)
        self.name = self.path.name
        self.device = torch.device(device)

        td_dir = self.path / "_tensordict"
        with open(td_dir / "meta.json", "r") as f:
            meta = json.load(f)
        with open(self.path / "meta_motion.json", "r") as f:
            motion_meta = json.load(f)

        self.num_frames = int(meta["shape"][0])
        self.num_joints = int(meta["joint_pos"]["shape"][1])
        self.num_bodies = int(meta["body_pos_w"]["shape"][1])
        self.joint_names = list(motion_meta["joint_names"])
        self.body_names = list(motion_meta["body_names"])

        self.starts = torch.as_tensor(motion_meta["starts"], dtype=torch.long)
        self.ends = torch.as_tensor(motion_meta["ends"], dtype=torch.long)
        self.lengths = self.ends - self.starts

        self.root_pos_w = MemoryMappedTensor.from_filename(
            td_dir / "root_pos_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, 3),
        )
        self.root_quat_w = MemoryMappedTensor.from_filename(
            td_dir / "root_quat_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, 4),
        )
        self.joint_pos = MemoryMappedTensor.from_filename(
            td_dir / "joint_pos.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, self.num_joints),
        )
        self.joint_vel = MemoryMappedTensor.from_filename(
            td_dir / "joint_vel.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, self.num_joints),
        )
        self.root_lin_vel_w = MemoryMappedTensor.from_filename(
            td_dir / "root_lin_vel_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, 3),
        )
        self.root_ang_vel_w = MemoryMappedTensor.from_filename(
            td_dir / "root_ang_vel_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, 3),
        )
        self.body_pos_w = MemoryMappedTensor.from_filename(
            td_dir / "body_pos_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, self.num_bodies, 3),
        )
        self.body_quat_w = MemoryMappedTensor.from_filename(
            td_dir / "body_quat_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, self.num_bodies, 4),
        )
        self.body_lin_vel_w = MemoryMappedTensor.from_filename(
            td_dir / "body_lin_vel_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, self.num_bodies, 3),
        )
        self.body_ang_vel_w = MemoryMappedTensor.from_filename(
            td_dir / "body_ang_vel_w.memmap",
            dtype=torch.float16,
            shape=(self.num_frames, self.num_bodies, 3),
        )

    @property
    def num_motions(self) -> int:
        return len(self.starts)

    def sample_motion_ids(self, num: int) -> torch.Tensor:
        return torch.randint(self.num_motions, (num,), device=self.device)

    def sample_frames(self, motion_ids, max_offset: int = 0) -> torch.Tensor:
        motion_ids = self._to_cpu_long(motion_ids)
        valid_lengths = self.lengths[motion_ids] - int(max_offset)
        if torch.any(valid_lengths <= 0):
            raise ValueError("max_offset exceeds at least one motion length.")
        frames = (torch.rand(valid_lengths.shape) * valid_lengths).long()
        return frames.to(self.device)

    def get_slice(
        self,
        motion_ids,
        frames,
        offsets=0,
    ) -> MotionBatch:
        motion_ids = self._to_cpu_long(motion_ids)
        frames = self._to_cpu_long(frames, length=len(motion_ids))
        offsets = self._to_cpu_long(offsets)

        indices = self.starts[motion_ids, None] + frames[:, None] + offsets[None, :]
        if torch.any(indices >= self.ends[motion_ids, None]):
            raise IndexError("Motion slice exceeds motion length.")

        return MotionBatch(
            root_pos_w=self.root_pos_w[indices].to(device=self.device, dtype=torch.float32),
            root_quat_w=self.root_quat_w[indices].to(device=self.device, dtype=torch.float32),
            joint_pos=self.joint_pos[indices].to(device=self.device, dtype=torch.float32),
            joint_vel=self.joint_vel[indices].to(device=self.device, dtype=torch.float32),
            root_lin_vel_w=self.root_lin_vel_w[indices].to(device=self.device, dtype=torch.float32),
            root_ang_vel_w=self.root_ang_vel_w[indices].to(device=self.device, dtype=torch.float32),
            body_pos_w=self.body_pos_w[indices].to(device=self.device, dtype=torch.float32),
            body_quat_w=self.body_quat_w[indices].to(device=self.device, dtype=torch.float32),
            body_lin_vel_w=self.body_lin_vel_w[indices].to(device=self.device, dtype=torch.float32),
            body_ang_vel_w=self.body_ang_vel_w[indices].to(device=self.device, dtype=torch.float32),
        )

    def _to_cpu_long(self, value, length: int | None = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().long()
        else:
            value = torch.as_tensor(value, dtype=torch.long)
        if value.ndim == 0:
            if length:
                value = torch.full((length,), int(value), dtype=torch.long)
            else:
                value = value.reshape(1)
        return value.reshape(-1)


def load_dataset(name: str, device: str | torch.device = "cpu") -> MotionDataset:
    return MotionDataset(DATASET_DIR / name, device=device)


def list_datasets() -> list[str]:
    if not DATASET_DIR.exists():
        return []
    return sorted(path.name for path in DATASET_DIR.iterdir() if (path / "_tensordict").exists())
