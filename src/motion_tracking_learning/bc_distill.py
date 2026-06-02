from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as distr
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules import IndependentNormal, VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY, Actor, make_batch, make_mlp
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.wandb import parse_checkpoint


TEACHER_KEY = "teacher"
STUDENT_KEY = "student"
TEACHER_ACTION_KEY = "_teacher_action"


@dataclass
class BCDistillConfig:
    _target_: str = "motion_tracking_learning.bc_distill.BCDistillPolicy"
    name: str = "bc_distill"
    train_every: int = 32
    bc_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    teacher_checkpoint_path: str | None = None
    load_teacher: bool = True
    rollout_policy: str = "teacher"
    activation: str = "Mish"
    max_grad_norm: float = 1.0
    use_ddp: bool = True
    in_keys: Tuple[str, ...] = (TEACHER_KEY, STUDENT_KEY)
    store_transitions: bool = True


ConfigStore.instance().store("bc_distill", node=BCDistillConfig, group="algo")


class FrozenVecNorm(VecNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with VecNorm.freeze():
            return super().forward(x)


class BCDistillPolicy(PPOBase):
    def __init__(
        self,
        cfg: BCDistillConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = BCDistillConfig(**cfg)
        if self.cfg.rollout_policy not in {"teacher", "student"}:
            raise ValueError("rollout_policy must be `teacher` or `student`.")
        self.device = device
        self.max_grad_norm = float(self.cfg.max_grad_norm)
        self.world_size = aa.get_world_size()
        self.should_reduce_grads = aa.is_distributed() and not self.cfg.use_ddp
        self.load_teacher = bool(self.cfg.load_teacher)

        if self.load_teacher and TEACHER_KEY not in observation_spec.keys(True, True):
            raise ValueError("bc_distill requires `teacher` observations.")
        if STUDENT_KEY not in observation_spec.keys(True, True):
            raise ValueError("bc_distill requires `student` observations.")

        self.action_dim = env.action_manager.action_dim
        activation = getattr(nn, self.cfg.activation)
        fake_input = observation_spec.zero()

        if self.load_teacher:
            self.teacher_vecnorm = Seq(
                Mod(
                    FrozenVecNorm(
                        input_shape=observation_spec[TEACHER_KEY].shape[-1:],
                        stats_shape=observation_spec[TEACHER_KEY].shape[-1:],
                        decay=1.0,
                    ),
                    [TEACHER_KEY],
                    ["_teacher_obs_normed"],
                )
            ).to(self.device)
            teacher_module = Seq(
                Mod(
                    make_mlp([256, 256, 256], activation=activation),
                    ["_teacher_obs_normed"],
                    ["_teacher_feature"],
                ),
                Mod(Actor(self.action_dim), ["_teacher_feature"], ["loc", "scale"]),
            )
            self.teacher_actor = ProbabilisticActor(
                module=teacher_module,
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(self.device)

        self.actor_vecnorm = Seq(
            Mod(
                VecNorm(
                    input_shape=observation_spec[STUDENT_KEY].shape[-1:],
                    stats_shape=observation_spec[STUDENT_KEY].shape[-1:],
                    decay=1.0,
                ),
                [STUDENT_KEY],
                ["_actor_obs_normed"],
            )
        ).to(self.device)
        student_module = Seq(
            Mod(
                make_mlp([256, 256, 256], activation=activation),
                ["_actor_obs_normed"],
                ["_actor_feature"],
            ),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
        )
        self.actor = ProbabilisticActor(
            module=student_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        if self.load_teacher:
            self.teacher_vecnorm(fake_input)
            self.teacher_actor(fake_input)
        self.actor_vecnorm(fake_input)
        self.actor(fake_input)

        self._init_actor(self.actor)
        if self.load_teacher:
            self._load_teacher()
            self._freeze(self.teacher_vecnorm)
            self._freeze(self.teacher_actor)

        if aa.is_distributed():
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor, device_ids=[aa.get_local_rank()])
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)

        self.opt = torch.optim.AdamW(
            [
                {"params": self.actor_vecnorm.parameters()},
                {"params": self.actor.parameters()},
            ],
            lr=self.cfg.lr,
            weight_decay=0.01,
        )

    def _init_actor(self, module: nn.Module) -> None:
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.orthogonal_(child.weight, 0.01)
                nn.init.constant_(child.bias, 0.0)

    def _freeze(self, module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _load_teacher(self) -> None:
        if self.cfg.teacher_checkpoint_path is None:
            raise ValueError("bc_distill requires `algo.teacher_checkpoint_path`.")

        checkpoint = parse_checkpoint(self.cfg.teacher_checkpoint_path)
        checkpoint.update()
        checkpoint_path = checkpoint.get_path()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        policy_state = state_dict.get("policy", state_dict)

        vecnorm_state = policy_state.get("actor_vecnorm", policy_state.get("vecnorm"))
        actor_state = policy_state.get("actor")
        if vecnorm_state is None or actor_state is None:
            raise KeyError("Teacher checkpoint must contain actor vecnorm and actor.")

        self.teacher_vecnorm.load_state_dict(vecnorm_state)
        self.teacher_actor.load_state_dict(actor_state)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        if mode == "train" and self.cfg.rollout_policy == "teacher":
            if not self.load_teacher:
                raise ValueError("Teacher rollout requires `algo.load_teacher=true`.")
            return Seq(
                self.teacher_vecnorm,
                self.teacher_actor,
                Mod(nn.Identity(), ["loc"], [ACTION_KEY]),
            )
        return Seq(self.actor_vecnorm, self.actor)

    def compute_value(self, tensordict):
        raise NotImplementedError("bc_distill does not use a critic.")

    def train_op(self, tensordict: TensorDict):
        if not self.load_teacher:
            raise RuntimeError("BC distillation training requires `algo.load_teacher=true`.")

        tensordict = tensordict.exclude("stats")
        valid = ~tensordict["is_init"]
        valid_cnt = valid.sum().clamp_min(1)

        with torch.no_grad():
            teacher_td = tensordict.select(TEACHER_KEY).clone()
            self.teacher_vecnorm(teacher_td)
            self.teacher_actor(teacher_td)
            tensordict[TEACHER_ACTION_KEY] = teacher_td["loc"].detach()

        self.actor_vecnorm(tensordict)

        infos = []
        for _ in range(self.cfg.bc_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update(minibatch))

        with torch.no_grad():
            self.actor(tensordict)
            target_action = tensordict[TEACHER_ACTION_KEY]
            error = (tensordict["loc"] - target_action).square().sum(dim=-1, keepdim=True)
            mse = (error * valid).sum() / valid_cnt

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["distill/action_mse"] = mse.item()
        infos["distill/target_action_std"] = target_action.std().item()
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        return dict(sorted(infos.items()))

    def _update(self, tensordict: TensorDict):
        target_action = tensordict[TEACHER_ACTION_KEY]
        valid = ~tensordict["is_init"]
        valid_cnt = valid.sum().clamp_min(1)

        self.actor(tensordict)
        action_error = (tensordict["loc"] - target_action).square().sum(dim=-1, keepdim=True)
        loss = (action_error * valid).sum() / valid_cnt

        self.opt.zero_grad()
        loss.backward()
        if self.should_reduce_grads:
            for param in self.actor.parameters():
                if param.grad is not None:
                    distr.all_reduce(param.grad.data, op=distr.ReduceOp.SUM)
                    param.grad.data /= self.world_size

        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt.step()

        return {
            "distill/bc_loss": loss.detach(),
            "actor/grad_norm": grad_norm,
            "actor/noise_std": tensordict["scale"].mean().detach(),
        }
