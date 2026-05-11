import torch

from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import matrix_from_quat
from active_adaptation.envs.mdp.rewards.base import Reward


class root_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.25, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = self.command_manager.target_body_pos_w[:, 0, 0] - self.asset.data.body_link_pos_w[:, 0]
        return torch.exp(-error.square().sum(dim=-1, keepdim=True) / self.sigma)


class root_quat_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.5, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_rotmat_w[:, 0, 0]
        current = matrix_from_quat(self.asset.data.body_link_quat_w[:, 0])
        error = (target - current).square().sum(dim=(-2, -1)).unsqueeze(-1)
        return torch.exp(-error / self.sigma)


class joint_pos_tracking(Reward):
    def __init__(self, env, weight: float, sigma: float = 0.5, track_var: bool = False):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        error = self.command_manager.target_joint_pos - self.asset.data.joint_pos
        return torch.exp(-error.square() / self.sigma).mean(dim=-1, keepdim=True)


class body_pos_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 0.25,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_pos_w[:, 0, self.body_ids]
        current = self.asset.data.body_link_pos_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1)
        return torch.exp(-error / self.sigma).mean(dim=-1, keepdim=True)

    def debug_draw(self) -> None:
        target = self.command_manager.target_body_pos_w[:, 0, self.body_ids]
        self.env.debug_draw.point(
            target.reshape(-1, 3),
            color=(1.0, 0.0, 0.0, 1.0),
        )


class body_rotmat_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 0.5,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_rotmat_w[:, 0, self.body_ids]
        current = matrix_from_quat(self.asset.data.body_link_quat_w[:, self.body_ids])
        error = (target - current).square().sum(dim=(-2, -1))
        return torch.exp(-error / self.sigma).mean(dim=-1, keepdim=True)


class body_lin_vel_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 1.0,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_lin_vel_w[:, 0, self.body_ids]
        current = self.asset.data.body_lin_vel_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1)
        return torch.exp(-error / self.sigma).mean(dim=-1, keepdim=True)


class body_ang_vel_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        body_names: str | list[str],
        sigma: float = 1.0,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.sigma = sigma

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.target_body_ang_vel_w[:, 0, self.body_ids]
        current = self.asset.data.body_ang_vel_w[:, self.body_ids]
        error = (target - current).square().sum(dim=-1)
        return torch.exp(-error / self.sigma).mean(dim=-1, keepdim=True)
