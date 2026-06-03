from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as distr
import torch.nn as nn
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
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    GAE,
    OBS_PRIV_KEY,
    REWARD_KEY,
    Actor,
    CatTensors,
    Critic,
    make_batch,
    make_mlp,
    normalize,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.wandb import parse_checkpoint


TEACHER_KEY = "teacher"
STUDENT_KEY = "student"
TEACHER_ACTION_KEY = "_teacher_action"


@dataclass
class PPODAggerConfig:
    _target_: str = "motion_tracking_learning.ppo_dagger.PPODAggerPolicy"
    name: str = "ppo_dagger"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    desired_kl: float | None = None
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    dagger_coef: float = 1.0
    dagger_coef_final: float = 0.05
    teacher_checkpoint_path: str | None = None
    load_teacher: bool = True
    activation: str = "Mish"
    max_grad_norm: float = 1.0
    compile: bool = False
    use_ddp: bool = True
    in_keys: Tuple[str, ...] = (TEACHER_KEY, STUDENT_KEY, OBS_PRIV_KEY)
    store_transitions: bool = True


ConfigStore.instance().store("ppo_dagger", node=PPODAggerConfig, group="algo")


class FrozenVecNorm(VecNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with VecNorm.freeze():
            return super().forward(x)


class PPODAggerPolicy(PPOBase):
    def __init__(
        self,
        cfg: PPODAggerConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPODAggerConfig(**cfg)
        self.device = device
        self.action_dim = env.action_manager.action_dim
        self.max_grad_norm = float(self.cfg.max_grad_norm)
        self.current_dagger_coef = float(self.cfg.dagger_coef)
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        self.world_size = aa.get_world_size()
        self.should_reduce_grads = aa.is_distributed() and not self.cfg.use_ddp
        self.load_teacher = bool(self.cfg.load_teacher)

        keys = observation_spec.keys(True, True)
        if STUDENT_KEY not in keys:
            raise ValueError("ppo_dagger requires `student` observations.")
        if self.load_teacher and TEACHER_KEY not in keys:
            raise ValueError("ppo_dagger training requires `teacher` observations.")
        self.use_critic = OBS_PRIV_KEY in keys
        if self.load_teacher and not self.use_critic:
            raise ValueError("ppo_dagger training requires `priv` observations.")

        fake_input = observation_spec.zero()
        activation = getattr(nn, self.cfg.activation)

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

        if self.use_critic:
            critic_dim = observation_spec[STUDENT_KEY].shape[-1] + observation_spec[OBS_PRIV_KEY].shape[-1]
            self.critic_vecnorm = Seq(
                CatTensors([STUDENT_KEY, OBS_PRIV_KEY], "_critic_input", del_keys=False, sort=False),
                Mod(
                    VecNorm(
                        input_shape=(critic_dim,),
                        stats_shape=(critic_dim,),
                        decay=1.0,
                    ),
                    ["_critic_input"],
                    ["_critic_obs_normed"],
                ),
            ).to(self.device)

        actor_module = Seq(
            Mod(
                make_mlp([256, 256, 256], activation=activation),
                ["_actor_obs_normed"],
                ["_actor_feature"],
            ),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
        )
        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)
        if self.use_critic:
            self.critic = Seq(
                Mod(
                    make_mlp([256, 256, 256], activation=activation),
                    ["_critic_obs_normed"],
                    ["_critic_feature"],
                ),
                Mod(Critic(1), ["_critic_feature"], ["state_value"]),
            ).to(self.device)

        if self.load_teacher:
            self.teacher_vecnorm(fake_input)
            self.teacher_actor(fake_input)
        self.actor_vecnorm(fake_input)
        if self.use_critic:
            self.critic_vecnorm(fake_input)
        self.actor(fake_input)
        if self.use_critic:
            self.critic(fake_input)

        self._init_linear(self.actor)
        if self.use_critic:
            self._init_linear(self.critic)
        if self.load_teacher:
            self._load_teacher()
            self._freeze(self.teacher_vecnorm)
            self._freeze(self.teacher_actor)

        if aa.is_distributed():
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor, device_ids=[aa.get_local_rank()])
                if self.use_critic:
                    self.critic = DDP(self.critic, device_ids=[aa.get_local_rank()])
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                if self.use_critic:
                    for param in self.critic.parameters():
                        distr.broadcast(param, src=0)

        params = [{"params": self.actor.parameters()}]
        if self.use_critic:
            params.append({"params": self.critic.parameters()})
        self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.01)
        self.update = self._update
        if self.cfg.compile and not aa.is_distributed():
            self.update = torch.compile(self.update)

    def _init_linear(self, module: nn.Module) -> None:
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
            raise ValueError("ppo_dagger requires `algo.teacher_checkpoint_path`.")

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

    def step_schedule(self, progress: float):
        start = float(self.cfg.dagger_coef)
        end = float(self.cfg.dagger_coef_final)
        self.current_dagger_coef = start + (end - start) * float(progress)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        if critic and self.use_critic:
            return Seq(self.actor_vecnorm, self.critic_vecnorm, self.critic, self.actor)
        return Seq(self.actor_vecnorm, self.actor)

    def compute_value(self, tensordict: TensorDict):
        if not self.use_critic:
            raise RuntimeError("ppo_dagger was created without a critic.")
        self.critic_vecnorm(tensordict)
        return self.critic(tensordict)

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        if not self.load_teacher or not self.use_critic:
            raise RuntimeError("ppo_dagger training requires teacher and critic modules.")

        tensordict = tensordict.exclude("stats")
        valid_ratio = (~tensordict["is_init"]).sum() / tensordict.numel()

        with torch.no_grad():
            teacher_td = tensordict.select(TEACHER_KEY).clone()
            self.teacher_vecnorm(teacher_td)
            self.teacher_actor(teacher_td)
            tensordict[TEACHER_ACTION_KEY] = teacher_td["loc"].detach()

        infos = []
        self.actor_vecnorm(tensordict)
        self.critic_vecnorm(tensordict)
        next_keys = tensordict["next"].keys(True, True)
        if STUDENT_KEY in next_keys and OBS_PRIV_KEY in next_keys:
            self.critic_vecnorm(tensordict["next"])
        self.compute_advantage(tensordict, self.critic, "adv", "ret")

        action = tensordict[ACTION_KEY]
        adv_unnormalized = tensordict["adv"]
        log_probs_before = tensordict["action_log_prob"]
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self.update(minibatch))
                if self.cfg.desired_kl is not None:
                    kl = infos[-1]["actor/approx_kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.cfg.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif 0.0 < kl < self.cfg.desired_kl / 2.0:
                        actor_lr = min(1e-3, actor_lr * 1.5)
                    self.opt.param_groups[0]["lr"] = actor_lr

        with torch.no_grad():
            td_after = self.actor(tensordict.copy())
            dist = IndependentNormal(td_after["loc"], td_after["scale"])
            log_probs_after = dist.log_prob(action)
            pg_loss_after = log_probs_after.reshape_as(adv_unnormalized) * adv_unnormalized
            pg_loss_before = log_probs_before.reshape_as(adv_unnormalized) * adv_unnormalized

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/pg_loss_raw_after"] = pg_loss_after.mean().item()
        infos["actor/pg_loss_raw_before"] = pg_loss_before.mean().item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.0).float().mean().item()
        infos["critic/valid_ratio"] = valid_ratio.item()
        infos["dagger/coef"] = self.current_dagger_coef
        return dict(sorted(infos.items()))

    def _update(self, tensordict: TensorDict):
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        teacher_action = tensordict[TEACHER_ACTION_KEY]

        valid = ~tensordict["is_init"]
        valid_cnt = valid.sum().clamp_min(1)

        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
        policy_loss = -(torch.min(surr1, surr2).reshape_as(valid) * valid).sum() / valid_cnt
        entropy_loss = -self.cfg.entropy_coef * entropy

        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(tensordict["ret"], values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        dagger_error = (tensordict["loc"] - teacher_action).square().mean(dim=-1, keepdim=True)
        dagger_loss = (dagger_error * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss + self.current_dagger_coef * dagger_loss
        self.opt.zero_grad()
        loss.backward()

        if self.should_reduce_grads:
            for module in (self.actor, self.critic):
                for param in module.parameters():
                    if param.grad is not None:
                        distr.all_reduce(param.grad.data, op=distr.ReduceOp.SUM)
                        param.grad.data /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/noise_std": tensordict["scale"].mean().detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "dagger/loss": dagger_loss.detach(),
            "dagger/action_mse": (dagger_error * valid).sum().detach() / valid_cnt,
        }
        with torch.no_grad():
            ret_var = tensordict["ret"][valid].var().clamp_min(1e-8)
            info["critic/explained_var"] = 1 - value_loss / ret_var
            info["actor/clamp_ratio"] = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
            info["actor/approx_kl"] = ((ratio - 1.0) - log_ratio).mean()
        return info
