import torch

from active_adaptation.envs.mdp.randomizations.base import Randomization


class randomize_default_joint_pos(Randomization):
    namespace = "motion_tracking"

    def __init__(self, env, low: float = -0.01, high: float = 0.01):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.action_manager = self.env.action_manager
        self.low = float(low)
        self.high = float(high)
        self.nominal = self.asset.data.default_joint_pos.clone()

    def startup(self) -> None:
        randomized = self.nominal + torch.empty_like(self.nominal).uniform_(
            self.low,
            self.high,
        )
        self.asset.data.default_joint_pos.copy_(randomized)

        action_defaults = randomized[:, self.action_manager.joint_ids]
        self.action_manager.default_joint_pos.copy_(action_defaults)
        self.action_manager.offset.zero_()
