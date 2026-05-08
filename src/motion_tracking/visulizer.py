import argparse
import time
from typing import Sequence
from pathlib import Path

import numpy as np
import torch

import mujoco
from mujoco import viewer

try:
    from motion_tracking.dataset import DATASET_DIR, MotionDataset, list_datasets, load_dataset
except ModuleNotFoundError:
    from dataset import DATASET_DIR, MotionDataset, list_datasets, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "active-adaptation" / "active_adaptation" / "assets" / "G1" / "mjcf" / "g1.xml"
FPS = 50.0

# MuJoCo viewer uses GLFW key codes for non-printable keys.
KEY_SPACE = 32
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265


class MujocoRobot:
    """Load a MuJoCo robot and write dataset frames into qpos."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        joint_names: Sequence[str] = (),
    ):
        self.model = self._load_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_addr = self._build_joint_qpos_addr(joint_names)

    def _load_model(self, model_path: Path | str) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(model_path))
        self._add_flat_ground(spec)
        return spec.compile()

    def _add_flat_ground(self, spec: mujoco.MjSpec) -> None:
        spec.add_texture(
            name="groundplane",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=300,
            height=300,
            rgb1=[0.20, 0.30, 0.40],
            rgb2=[0.10, 0.20, 0.30],
            mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[0.80, 0.80, 0.80],
        )
        spec.add_material(
            name="groundplane",
            textures=["groundplane"],
            texuniform=1,
            texrepeat=[5.0, 5.0],
            reflectance=0.20,
        )
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            friction=[0.75, 0.1, 0.1],
            material="groundplane",
        )

    def _build_joint_qpos_addr(self, joint_names: Sequence[str]) -> np.ndarray:
        qpos_addr = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint not found in MuJoCo model: {name}")
            qpos_addr.append(self.model.jnt_qposadr[joint_id])
        return np.asarray(qpos_addr, dtype=np.int32)

    def set_pose(
        self,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        """Write one motion frame into the MuJoCo state."""
        self.data.qpos[:3] = root_pos_w.detach().cpu().numpy()
        self.data.qpos[3:7] = root_quat_w.detach().cpu().numpy()
        self.data.qpos[self.joint_qpos_addr] = joint_pos.detach().cpu().numpy()
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)


class MotionPlayer:
    def __init__(
        self,
        dataset: MotionDataset,
        motion_idx: int = 0,
        fps: float = FPS,
        paused: bool = False,
    ):
        self.dataset = dataset
        self.robot = MujocoRobot(MODEL_PATH, dataset.joint_names)
        self.motion_idx = motion_idx % self.num_motions
        self.frame = 0
        self.fps = fps
        self.paused = paused
        self.speed = 1.0
        self._last_time = time.perf_counter()
        self._frame_accumulator = 0.0
        self.apply_current_frame()

    @property
    def num_motions(self) -> int:
        return self.dataset.num_motions

    @property
    def motion_start(self) -> int:
        return int(self.dataset.starts[self.motion_idx].item())

    @property
    def motion_end(self) -> int:
        return int(self.dataset.ends[self.motion_idx].item())

    @property
    def motion_length(self) -> int:
        return self.motion_end - self.motion_start

    @property
    def global_frame(self) -> int:
        return self.motion_start + self.frame

    def apply_current_frame(self) -> None:
        frame = self.dataset.get_slice(self.motion_idx, self.frame)
        self.robot.set_pose(
            frame.root_pos_w[0, 0],
            frame.root_quat_w[0, 0],
            frame.joint_pos[0, 0],
        )

    def step(self, frames: int = 1) -> None:
        self.frame = (self.frame + frames) % self.motion_length
        self.apply_current_frame()

    def select_motion(self, motion_idx: int) -> None:
        self.motion_idx = motion_idx % self.num_motions
        self.frame = 0
        self._frame_accumulator = 0.0
        self._last_time = time.perf_counter()
        self.apply_current_frame()

    def next_motion(self) -> None:
        self.select_motion(self.motion_idx + 1)

    def prev_motion(self) -> None:
        self.select_motion(self.motion_idx - 1)

    def restart(self) -> None:
        self.frame = 0
        self._frame_accumulator = 0.0
        self._last_time = time.perf_counter()
        self.apply_current_frame()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._last_time = time.perf_counter()

    def change_speed(self, delta: float) -> None:
        self.speed = min(5.0, max(0.1, self.speed + delta))

    def update(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now
        if self.paused:
            return

        self._frame_accumulator += elapsed * self.fps * self.speed
        steps = int(self._frame_accumulator)
        if steps:
            self._frame_accumulator -= steps
            self.step(steps)

    def key_callback(self, keycode: int) -> None:
        if keycode == KEY_SPACE:
            self.toggle_pause()
        elif keycode == KEY_RIGHT:
            self.paused = True
            self.step(1)
        elif keycode == KEY_LEFT:
            self.paused = True
            self.step(-1)
        elif keycode == KEY_DOWN:
            self.next_motion()
        elif keycode == KEY_UP:
            self.prev_motion()
        elif keycode in (ord("="), ord("+")):
            self.change_speed(0.25)
        elif keycode == ord("-"):
            self.change_speed(-0.25)
        elif keycode in (ord("r"), ord("R")):
            self.restart()
        elif keycode == ord("0"):
            self.speed = 1.0

    def overlay_text(self) -> str:
        state = "PAUSED" if self.paused else "PLAYING"
        return (
            f"[{state}] {self.dataset.name} "
            f"motion {self.motion_idx + 1}/{self.num_motions} "
            f"frame {self.frame}/{self.motion_length} "
            f"speed {self.speed:.2f}x"
        )

    def run(self) -> None:
        with viewer.launch_passive(
            self.robot.model,
            self.robot.data,
            key_callback=self.key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as handle:
            while handle.is_running():
                self.update()
                handle.set_texts((None, None, "Motion Player", self.overlay_text()))
                handle.sync()
                time.sleep(0.001)


def main() -> None:
    datasets = list_datasets()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=datasets, default=datasets[0] if datasets else None)
    parser.add_argument("-m", "--motion", type=int, default=0)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--paused", action="store_true")
    args = parser.parse_args()

    if args.dataset is None:
        raise RuntimeError(f"No dataset found under {DATASET_DIR}")

    player = MotionPlayer(
        load_dataset(args.dataset),
        motion_idx=args.motion,
        fps=args.fps,
        paused=args.paused,
    )
    player.run()


if __name__ == "__main__":
    main()
