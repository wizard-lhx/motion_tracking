from __future__ import annotations

import gc
import itertools
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint


CONFIG_PATH = Path(aa.__file__).resolve().parents[1] / "cfg"


def _apply_metric_launch_defaults(cfg: DictConfig) -> None:
    user_cfg = cfg.get("tracking_metrics", {})
    if bool(user_cfg.get("headless", True)):
        cfg.headless = True
        cfg.app.headless = True


def _metric_cfg(cfg: DictConfig):
    default = {
        "episodes": max(int(cfg.task.num_envs), 1),
        "max_steps": int(cfg.task.max_episode_length) * max(int(cfg.task.num_envs), 1) * 2,
        "run_name": cfg.get("run_name", None) or f"{cfg.task.name}-play-metrics",
        "output_dir": "outputs/play_metrics",
    }
    user_cfg = cfg.get("tracking_metrics", {})
    return OmegaConf.merge(default, user_cfg)


def _close_isaac_app() -> None:
    try:
        if aa.get_backend() != "isaac":
            return
    except RuntimeError:
        return

    try:
        from isaaclab.app import AppLauncher
    except Exception:
        return

    for obj in gc.get_objects():
        try:
            if isinstance(obj, AppLauncher):
                obj.app.close(wait_for_replicator=False)
                return
        except Exception:
            continue


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    _apply_metric_launch_defaults(cfg)

    aa.init(cfg, auto_rank=True)

    OmegaConf.resolve(cfg)

    from active_adaptation.helpers import make_env_policy
    from motion_tracking.metrics import PlayTrackingMetrics

    checkpoint = parse_checkpoint(cfg.checkpoint_path)
    env = None
    try:
        env, policy = make_env_policy(cfg, checkpoint)

        metric_cfg = _metric_cfg(cfg)
        target_episodes = int(metric_cfg.episodes)
        max_steps = int(metric_cfg.max_steps)

        rollout_policy = policy.get_rollout_policy("eval")
        env.base_env.eval()
        carry = env.reset()
        metrics = PlayTrackingMetrics(
            env,
            run_name=str(metric_cfg.run_name),
            output_dir=metric_cfg.output_dir,
        )

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for step in itertools.count():
                if len(metrics.episodes) >= target_episodes:
                    break
                if step >= max_steps:
                    print(
                        f"[tracking_metrics] reached max_steps={max_steps} "
                        f"with completed_episodes={len(metrics.episodes)}"
                    )
                    break

                carry = rollout_policy(carry)
                action = carry.get("action", None)
                metrics.record_step(action)
                tensordict, carry = env.step_and_maybe_reset(carry)
                next_td = tensordict["next"]
                terminated = next_td["terminated"] if "terminated" in next_td.keys() else None
                truncated = next_td["truncated"] if "truncated" in next_td.keys() else None
                metrics.finalize_done(
                    next_td["done"],
                    terminated=terminated,
                    truncated=truncated,
                )

        metrics.print_summary()
        json_path, csv_path = metrics.save()
        print(f"[tracking_metrics] wrote JSON: {json_path}")
        print(f"[tracking_metrics] wrote CSV: {csv_path}")
    finally:
        if env is not None:
            env.close()
        _close_isaac_app()


if __name__ == "__main__":
    main()
