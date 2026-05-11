from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import MemoryMappedTensor
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "motion_tracking" / "dataset"
MODEL_PATH = (
    PROJECT_ROOT
    / "active-adaptation"
    / "active_adaptation"
    / "assets"
    / "G1"
    / "mjcf"
    / "g1.xml"
)


@dataclass
class SourceDataset:
    path: Path
    num_frames: int
    joint_names: list[str]
    starts: torch.Tensor
    ends: torch.Tensor
    root_pos_w: MemoryMappedTensor
    root_quat_w: MemoryMappedTensor
    joint_pos: MemoryMappedTensor


def load_source_dataset(path: Path) -> SourceDataset:
    td_dir = path / "_tensordict"
    with open(td_dir / "meta.json", "r") as f:
        meta = json.load(f)
    with open(path / "meta_motion.json", "r") as f:
        motion_meta = json.load(f)

    num_frames = int(meta["shape"][0])
    num_joints = int(meta["joint_pos"]["shape"][1])
    return SourceDataset(
        path=path,
        num_frames=num_frames,
        joint_names=list(motion_meta["joint_names"]),
        starts=torch.as_tensor(motion_meta["starts"], dtype=torch.long),
        ends=torch.as_tensor(motion_meta["ends"], dtype=torch.long),
        root_pos_w=MemoryMappedTensor.from_filename(
            td_dir / "root_pos_w.memmap",
            dtype=torch.float16,
            shape=(num_frames, 3),
        ),
        root_quat_w=MemoryMappedTensor.from_filename(
            td_dir / "root_quat_w.memmap",
            dtype=torch.float16,
            shape=(num_frames, 4),
        ),
        joint_pos=MemoryMappedTensor.from_filename(
            td_dir / "joint_pos.memmap",
            dtype=torch.float16,
            shape=(num_frames, num_joints),
        ),
    )


def load_model(path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(path))
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
    )
    return spec.compile()


def qpos_from_frame(dataset: SourceDataset, model: mujoco.MjModel, joint_qposadr, frame: int):
    qpos = model.qpos0.copy()
    qpos[:3] = dataset.root_pos_w[frame].detach().cpu().numpy()
    qpos[3:7] = dataset.root_quat_w[frame].detach().cpu().numpy()
    qpos[3:7] /= np.linalg.norm(qpos[3:7])
    qpos[joint_qposadr] = dataset.joint_pos[frame].detach().cpu().numpy()
    return qpos


def object_velocities(model: mujoco.MjModel, data: mujoco.MjData, body_ids) -> np.ndarray:
    vel = np.empty((len(body_ids), 6), dtype=np.float32)
    tmp = np.empty(6, dtype=np.float64)
    for i, body_id in enumerate(body_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, tmp, 0)
        vel[i] = tmp
    return vel


def output_shapes(
    num_frames: int,
    num_joints: int,
    num_bodies: int,
) -> dict[str, tuple[int, ...]]:
    return {
        "joint_vel": (num_frames, num_joints),
        "root_lin_vel_w": (num_frames, 3),
        "root_ang_vel_w": (num_frames, 3),
        "body_pos_w": (num_frames, num_bodies, 3),
        "body_quat_w": (num_frames, num_bodies, 4),
        "body_lin_vel_w": (num_frames, num_bodies, 3),
        "body_ang_vel_w": (num_frames, num_bodies, 3),
    }


def make_outputs(
    td_dir: Path,
    num_frames: int,
    num_joints: int,
    num_bodies: int,
    overwrite: bool,
):
    shapes = output_shapes(num_frames, num_joints, num_bodies)
    return {
        name: MemoryMappedTensor.empty(
            shapes[name],
            dtype=torch.float16,
            filename=td_dir / f"{name}.memmap",
            existsok=overwrite,
        )
        for name in shapes
    }


def write_meta(dataset: SourceDataset, body_names: list[str]) -> None:
    td_dir = dataset.path / "_tensordict"
    with open(td_dir / "meta.json", "r") as f:
        meta = json.load(f)
    with open(dataset.path / "meta_motion.json", "r") as f:
        motion_meta = json.load(f)

    shapes = {
        "joint_vel": [dataset.num_frames, len(dataset.joint_names)],
        "root_lin_vel_w": [dataset.num_frames, 3],
        "root_ang_vel_w": [dataset.num_frames, 3],
        "body_pos_w": [dataset.num_frames, len(body_names), 3],
        "body_quat_w": [dataset.num_frames, len(body_names), 4],
        "body_lin_vel_w": [dataset.num_frames, len(body_names), 3],
        "body_ang_vel_w": [dataset.num_frames, len(body_names), 3],
    }
    for name, shape in shapes.items():
        meta[name] = {
            "device": "cpu",
            "shape": shape,
            "dtype": "torch.float16",
            "is_nested": False,
        }
    motion_meta["body_names"] = body_names

    with open(td_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    with open(dataset.path / "meta_motion.json", "w") as f:
        json.dump(motion_meta, f)


def build_kinematics(dataset: SourceDataset, model_path: Path, fps: float, overwrite: bool):
    model = load_model(model_path)
    data = mujoco.MjData(model)

    body_ids = list(range(1, model.nbody))
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id).split("/")[-1]
        for body_id in body_ids
    ]
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in dataset.joint_names
    ]
    joint_qposadr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
    joint_qveladr = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)

    outputs = make_outputs(
        dataset.path / "_tensordict",
        dataset.num_frames,
        len(dataset.joint_names),
        len(body_ids),
        overwrite,
    )

    qvel = np.zeros(model.nv)
    dt = 1.0 / fps

    with tqdm(total=dataset.num_frames, desc=dataset.path.name, unit="frame") as pbar:
        for start, end in zip(dataset.starts.tolist(), dataset.ends.tolist()):
            qvel[:] = 0.0
            qpos = qpos_from_frame(dataset, model, joint_qposadr, start)
            for frame in range(start, end):
                if frame + 1 < end:
                    qpos_next = qpos_from_frame(dataset, model, joint_qposadr, frame + 1)
                    if np.dot(qpos[3:7], qpos_next[3:7]) < 0.0:
                        qpos_next[3:7] *= -1.0
                    mujoco.mj_differentiatePos(model, qvel, dt, qpos, qpos_next)

                outputs["joint_vel"][frame] = torch.as_tensor(qvel[joint_qveladr], dtype=torch.float16)
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(model, data)
                body_vel = object_velocities(model, data, body_ids)
                outputs["body_pos_w"][frame] = torch.as_tensor(data.xpos[body_ids], dtype=torch.float16)
                outputs["body_quat_w"][frame] = torch.as_tensor(data.xquat[body_ids], dtype=torch.float16)
                outputs["body_ang_vel_w"][frame] = torch.as_tensor(body_vel[:, :3], dtype=torch.float16)
                outputs["body_lin_vel_w"][frame] = torch.as_tensor(body_vel[:, 3:], dtype=torch.float16)
                outputs["root_ang_vel_w"][frame] = torch.as_tensor(body_vel[0, :3], dtype=torch.float16)
                outputs["root_lin_vel_w"][frame] = torch.as_tensor(body_vel[0, 3:], dtype=torch.float16)

                if frame + 1 < end:
                    qpos = qpos_next

            pbar.update(end - start)

    write_meta(dataset, body_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", nargs="*")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = [Path(name) for name in args.dataset] if args.dataset else sorted(
        path for path in DATASET_DIR.iterdir() if (path / "_tensordict").exists()
    )

    for path in paths:
        if not path.exists():
            path = DATASET_DIR / path
        build_kinematics(load_source_dataset(path), args.model, args.fps, args.overwrite)


if __name__ == "__main__":
    main()
