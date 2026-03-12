from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class HighLevelPolicyConfig:
    num_drones: int = 4
    num_skills: int = 7
    hidden_dim: int = 256


class JointSkillActorCritic(nn.Module):
    """Centralized high-level policy over joint skill actions.

    Input:
      global_obs: (B, D_global)
    Output:
      logits: (B, N, Z), value: (B,)
    """

    def __init__(self, global_obs_dim: int, cfg: HighLevelPolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or HighLevelPolicyConfig()

        self.backbone = nn.Sequential(
            nn.Linear(global_obs_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(self.cfg.hidden_dim, self.cfg.num_drones * self.cfg.num_skills)
        self.value_head = nn.Linear(self.cfg.hidden_dim, 1)

    def forward(self, global_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(global_obs)
        logits = self.policy_head(h).view(-1, self.cfg.num_drones, self.cfg.num_skills)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(
        self,
        global_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logits, value = self.forward(global_obs)
        dist = Categorical(logits=logits)
        if deterministic:
            skill = torch.argmax(logits, dim=-1)
        else:
            skill = dist.sample()
        logp = dist.log_prob(skill)  # (B, N)
        entropy = dist.entropy().mean(dim=-1)  # (B,)
        return {
            "skill": skill,
            "logp": logp,
            "logp_joint": logp.sum(dim=-1),
            "value": value,
            "entropy": entropy,
            "logits": logits,
        }
