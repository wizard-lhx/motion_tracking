from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

import torch

from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from motion_tracking.reward import _desired_pos, _desired_quat


_METRIC_SUM_KEYS = (
    "global_body_pos_error_mean_m",
    "anchor_body_pos_error_mean_m",
    "body_ori_error_mean_rad",
    "joint_pos_error_mean_rad",
    "root_pos_error_mean_m",
    "root_ori_error_mean_rad",
)


def _unwrap_base_env(env):
    base = env
    while hasattr(base, "base_env"):
        base = base.base_env
    return base


def _cfg_get(cfg: Any, path: tuple[str, ...], default=None):
    cur = cfg
    for key in path:
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_") or "run"


class PlayTrackingMetrics:
    """Behavior-neutral motion-tracking metrics for play/eval rollouts."""

    def __init__(self, env, run_name: str, output_dir: str | Path = "outputs/play_metrics"):
        self.env = _unwrap_base_env(env)
        self.asset = self.env.scene.articulations["robot"]
        self.command = self.env.command_manager
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.step_dt = float(self.env.step_dt)
        self.run_name = _safe_run_name(run_name)
        self.output_dir = Path(output_dir)

        body_names = _cfg_get(
            self.env.cfg,
            ("reward", "tracking", "body_pos_tracking", "body_names"),
            ".*",
        )
        if not isinstance(body_names, str):
            body_names = list(body_names)
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self._sum = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in _METRIC_SUM_KEYS
        }
        self._global_body_pos_max = torch.zeros(self.num_envs, device=self.device)
        self._anchor_body_pos_max = torch.zeros(self.num_envs, device=self.device)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._action_smooth_sum = torch.zeros(self.num_envs, device=self.device)
        self._action_smooth_count = torch.zeros(self.num_envs, device=self.device)
        self._prev_action: torch.Tensor | None = None
        self._prev_action_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.episodes: list[dict[str, Any]] = []

    @torch.inference_mode()
    def record_step(self, action: torch.Tensor | None) -> None:
        metrics = self._compute_step_metrics()
        for key in _METRIC_SUM_KEYS:
            self._sum[key].add_(metrics[key])
        self._global_body_pos_max.copy_(
            torch.maximum(
                self._global_body_pos_max,
                metrics["global_body_pos_error_max_m"],
            )
        )
        self._anchor_body_pos_max.copy_(
            torch.maximum(
                self._anchor_body_pos_max,
                metrics["anchor_body_pos_error_max_m"],
            )
        )
        self._steps.add_(1)

        if action is not None:
            action = action.detach()
            if self._prev_action is not None:
                valid = self._prev_action_valid
                smooth = (action - self._prev_action).norm(dim=-1)
                self._action_smooth_sum[valid] += smooth[valid]
                self._action_smooth_count[valid] += 1.0
            self._prev_action = action.clone()
            self._prev_action_valid[:] = True

    @torch.inference_mode()
    def finalize_done(
        self,
        done: torch.Tensor,
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
    ) -> None:
        done = done.reshape(self.num_envs).bool()
        if not done.any():
            return

        terminated = (
            torch.zeros_like(done)
            if terminated is None
            else terminated.reshape(self.num_envs).bool()
        )
        truncated = (
            torch.zeros_like(done)
            if truncated is None
            else truncated.reshape(self.num_envs).bool()
        )

        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            steps = int(self._steps[env_id].item())
            denom = max(steps, 1)
            episode = {
                "env_id": env_id,
                "steps": steps,
                "seconds": steps * self.step_dt,
                "terminated": bool(terminated[env_id].item()),
                "truncated": bool(truncated[env_id].item()),
            }
            episode["fall"] = episode["terminated"]
            episode["success"] = episode["truncated"] and not episode["terminated"]

            for key in _METRIC_SUM_KEYS:
                episode[key] = float((self._sum[key][env_id] / denom).item())
            episode["global_body_pos_error_max_m"] = float(
                self._global_body_pos_max[env_id].item()
            )
            episode["anchor_body_pos_error_max_m"] = float(
                self._anchor_body_pos_max[env_id].item()
            )
            smooth_count = float(self._action_smooth_count[env_id].item())
            episode["action_smoothness_mean_norm"] = (
                float((self._action_smooth_sum[env_id] / smooth_count).item())
                if smooth_count > 0
                else 0.0
            )
            self.episodes.append(episode)

        self._reset_env_accumulators(done)

    def summary(self) -> dict[str, Any]:
        if not self.episodes:
            return {"run_name": self.run_name, "episode_count": 0}

        keys = [
            *_METRIC_SUM_KEYS,
            "global_body_pos_error_max_m",
            "anchor_body_pos_error_max_m",
            "action_smoothness_mean_norm",
            "steps",
            "seconds",
        ]
        summary: dict[str, Any] = {
            "run_name": self.run_name,
            "episode_count": len(self.episodes),
            "success_rate": _mean([float(ep["success"]) for ep in self.episodes]),
            "fall_rate": _mean([float(ep["fall"]) for ep in self.episodes]),
            "average_survival_seconds": _mean([ep["seconds"] for ep in self.episodes]),
        }
        for key in keys:
            values = [float(ep[key]) for ep in self.episodes]
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_std"] = _std(values)
        return summary

    def print_summary(self) -> None:
        summary = self.summary()
        if summary["episode_count"] == 0:
            print("[tracking_metrics] no completed episodes")
            return

        print("\n=== Tracking Metrics Summary ===")
        print(f"run_name: {self.run_name}")
        print(f"episodes: {summary['episode_count']}")
        print(f"success_rate: {summary['success_rate']:.3f}")
        print(f"fall_rate: {summary['fall_rate']:.3f}")
        print(f"average_survival_seconds: {summary['average_survival_seconds']:.3f} s")
        rows = [
            ("global body pos mean", "global_body_pos_error_mean_m", "m"),
            ("global body pos max", "global_body_pos_error_max_m", "m"),
            ("anchor body pos mean", "anchor_body_pos_error_mean_m", "m"),
            ("anchor body pos max", "anchor_body_pos_error_max_m", "m"),
            ("body orientation mean", "body_ori_error_mean_rad", "rad"),
            ("joint angle mean", "joint_pos_error_mean_rad", "rad"),
            ("root position mean", "root_pos_error_mean_m", "m"),
            ("root orientation mean", "root_ori_error_mean_rad", "rad"),
            ("action smoothness", "action_smoothness_mean_norm", "norm"),
            ("episode steps", "steps", "steps"),
            ("episode seconds", "seconds", "s"),
        ]
        print(f"{'metric':<28} {'mean':>12} {'std':>12} unit")
        for label, key, unit in rows:
            mean = summary[f"{key}_mean"]
            std = summary[f"{key}_std"]
            print(f"{label:<28} {mean:12.6f} {std:12.6f} {unit}")

    def save(self) -> tuple[Path, Path]:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = self.output_dir / self.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"tracking_metrics-{timestamp}.json"
        csv_path = out_dir / f"tracking_metrics-{timestamp}.csv"

        payload = {
            "summary": self.summary(),
            "episodes": self.episodes,
            "body_names": self.body_names,
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        fieldnames = sorted({key for ep in self.episodes for key in ep.keys()})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episodes)
        return json_path, csv_path

    def _compute_step_metrics(self) -> dict[str, torch.Tensor]:
        current_body_pos = self.asset.data.body_link_pos_w[:, self.body_ids]
        current_body_quat = self.asset.data.body_link_quat_w[:, self.body_ids]
        target_body_pos = self.command.target_body_pos_w[:, 0, self.body_ids]
        target_body_quat = self.command.target_body_quat_w[:, 0, self.body_ids]

        global_body_pos = (target_body_pos - current_body_pos).norm(dim=-1)

        anchor_pos = self.asset.data.body_link_pos_w[:, 0]
        anchor_quat = self.asset.data.body_link_quat_w[:, 0]
        ref_anchor_pos = self.command.target_body_pos_w[:, 0, 0]
        ref_anchor_quat = self.command.target_body_quat_w[:, 0, 0]

        anchor_body_pos_target = _desired_pos(
            anchor_pos,
            anchor_quat,
            ref_anchor_pos,
            ref_anchor_quat,
            target_body_pos,
        )
        anchor_body_pos = (anchor_body_pos_target - current_body_pos).norm(dim=-1)

        anchor_body_quat_target = _desired_quat(
            anchor_quat,
            ref_anchor_quat,
            target_body_quat,
        )
        body_ori = axis_angle_from_quat(
            quat_mul(anchor_body_quat_target, quat_conjugate(current_body_quat))
        ).norm(dim=-1)

        joint_pos = (self.command.target_joint_pos - self.asset.data.joint_pos).abs()

        root_pos = (ref_anchor_pos - anchor_pos).norm(dim=-1)
        root_ori = axis_angle_from_quat(
            quat_mul(ref_anchor_quat, quat_conjugate(anchor_quat))
        ).norm(dim=-1)

        return {
            "global_body_pos_error_mean_m": global_body_pos.mean(dim=-1),
            "global_body_pos_error_max_m": global_body_pos.max(dim=-1).values,
            "anchor_body_pos_error_mean_m": anchor_body_pos.mean(dim=-1),
            "anchor_body_pos_error_max_m": anchor_body_pos.max(dim=-1).values,
            "body_ori_error_mean_rad": body_ori.mean(dim=-1),
            "joint_pos_error_mean_rad": joint_pos.mean(dim=-1),
            "root_pos_error_mean_m": root_pos,
            "root_ori_error_mean_rad": root_ori,
        }

    def _reset_env_accumulators(self, done: torch.Tensor) -> None:
        for value in self._sum.values():
            value[done] = 0.0
        self._global_body_pos_max[done] = 0.0
        self._anchor_body_pos_max[done] = 0.0
        self._steps[done] = 0
        self._action_smooth_sum[done] = 0.0
        self._action_smooth_count[done] = 0.0
        self._prev_action_valid[done] = False
        if self._prev_action is not None:
            self._prev_action[done] = 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(var**0.5)
