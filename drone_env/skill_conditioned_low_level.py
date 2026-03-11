from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class SkillConditionedPolicyConfig:
    num_skills: int = 7
    hidden_dim: int = 256
    init_log_std: float = -0.7


def skills_to_onehot(skill_idx: torch.Tensor, num_skills: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(skill_idx.long(), num_classes=num_skills).to(dtype=torch.float32)


class SkillConditionedActorCritic(nn.Module):
    """Per-drone shared actor-critic.

    Input per drone:
      [obs_local, onehot(skill), tau]
    Output per drone:
      nominal velocity distribution params and state value.
    """

    def __init__(self, obs_local_dim: int, cfg: SkillConditionedPolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or SkillConditionedPolicyConfig()
        in_dim = int(obs_local_dim + self.cfg.num_skills + 1)

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.value_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((3,), self.cfg.init_log_std))

    def _build_features(self, obs_local: torch.Tensor, skill_idx: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # obs_local: (B, N, D), skill_idx: (B, N), tau: (B, 1) or (B, N, 1)
        b, n, _ = obs_local.shape
        if tau.ndim == 2:
            tau = tau[:, None, :].expand(b, n, 1)
        elif tau.ndim == 3 and tau.shape[1] == 1:
            tau = tau.expand(b, n, 1)
        elif tau.ndim == 3 and tau.shape[1] == n:
            pass
        else:
            raise ValueError(f"Unexpected tau shape {tuple(tau.shape)}")

        onehot = skills_to_onehot(skill_idx, self.cfg.num_skills)
        return torch.cat([obs_local, onehot, tau], dim=-1)

    def forward(self, obs_local: torch.Tensor, skill_idx: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._build_features(obs_local, skill_idx, tau)
        h = self.backbone(feat)
        mean = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        std = torch.exp(self.log_std).view(1, 1, 3).expand_as(mean)
        return mean, std, value

    def act(
        self,
        obs_local: torch.Tensor,
        skill_idx: torch.Tensor,
        tau: torch.Tensor,
        cbf_state: torch.Tensor,
        qp_solver=None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Single-step rollout call, with batch size B=1 expected.
        mean, std, value = self.forward(obs_local, skill_idx, tau)
        dist = Normal(mean, std)
        if deterministic:
            u_nom = mean
        else:
            u_nom = dist.rsample()
        logp = dist.log_prob(u_nom).sum(dim=-1)  # (B, N)

        u_safe = u_nom
        slack = torch.zeros((u_nom.shape[0], 1), dtype=u_nom.dtype, device=u_nom.device)
        if qp_solver is not None:
            safe_list = []
            slack_list = []
            qp_fail_count = 0
            for b in range(u_nom.shape[0]):
                try:
                    u_b, s_b = qp_solver.solve_torch(cbf_state[b], u_nom[b])
                except Exception:
                    # Fallback to nominal action for this sample to keep training loop alive.
                    u_b = u_nom[b]
                    s_b = torch.zeros((1,), dtype=u_nom.dtype, device=u_nom.device)
                    qp_fail_count += 1
                safe_list.append(u_b)
                # s_b shape: (num_constraints,)
                slack_list.append(torch.mean(s_b**2, dim=0, keepdim=True))
            u_safe = torch.stack(safe_list, dim=0)
            slack = torch.stack(slack_list, dim=0)
        else:
            qp_fail_count = 0

        return {
            "u_nom": u_nom,
            "u_safe": u_safe,
            "logp": logp,
            "logp_joint": logp.sum(dim=-1),  # (B,)
            "value_agent": value,  # (B, N)
            "value_joint": value.mean(dim=-1),  # (B,)
            "entropy_joint": dist.entropy().sum(dim=-1).mean(dim=-1),  # (B,)
            "slack_aux": slack.squeeze(-1),  # (B,)
            "qp_fail_count": torch.as_tensor([qp_fail_count], dtype=torch.int32, device=obs_local.device),
            "mean": mean,
            "std": std,
        }


def skill_velocity_alignment_reward(v_safe: np.ndarray, skill_idx: np.ndarray) -> float:
    """Simple intrinsic reward encouraging velocity alignment with skill command."""
    pref = np.array(
        [
            [0.0, 0.0, 0.0],   # keep
            [0.0, -1.0, 0.0],  # left
            [0.0, 1.0, 0.0],   # right
            [0.0, 0.0, 1.0],   # up
            [0.0, 0.0, -1.0],  # down
            [1.0, 0.0, 0.0],   # forward
            [-1.0, 0.0, 0.0],  # backward
        ],
        dtype=np.float32,
    )
    v = np.asarray(v_safe, dtype=np.float32)
    z = np.asarray(skill_idx, dtype=np.int64).reshape(-1)
    eps = 1e-6
    vals = []
    for i in range(v.shape[0]):
        p = pref[int(z[i])]
        if int(z[i]) == 0:
            vals.append(float(np.exp(-np.linalg.norm(v[i]) ** 2)))
        else:
            vals.append(float(np.dot(v[i], p) / (np.linalg.norm(v[i]) + eps)))
    return float(np.mean(vals))
