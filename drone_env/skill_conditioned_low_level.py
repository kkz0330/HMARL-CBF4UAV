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
    action_scale_xy: float = 1.0
    action_scale_z: float = 1.0
    log_std_min: float = -4.0
    log_std_max: float = 0.5
    use_parametric_qp: bool = True
    qp_h_base: float = 2.0
    qp_h_min: float = 1e-3
    qp_f_scale: float = 0.5
    qp_only: bool = True


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
        self.register_buffer(
            "action_scale",
            torch.tensor(
                [self.cfg.action_scale_xy, self.cfg.action_scale_xy, self.cfg.action_scale_z],
                dtype=torch.float32,
            ).view(1, 1, 3),
        )

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.value_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.qp_h_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.qp_f_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.log_std = nn.Parameter(torch.full((3,), self.cfg.init_log_std))
        self._softplus = nn.Softplus()

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
        mean = torch.tanh(self.policy_head(h)) * self.action_scale
        value = self.value_head(h).squeeze(-1)
        log_std = torch.clamp(self.log_std, min=self.cfg.log_std_min, max=self.cfg.log_std_max)
        std = torch.exp(log_std).view(1, 1, 3).expand_as(mean)
        return mean, std, value

    def qp_objective_params(
        self,
        obs_local: torch.Tensor,
        skill_idx: torch.Tensor,
        tau: torch.Tensor,
        u_nom: torch.Tensor | None,
        nominal_anchor_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute parametric QP objective terms from phi = Gamma_nu(o|z)."""
        anchor_w = 0.0 if bool(self.cfg.qp_only) else float(nominal_anchor_weight)
        if u_nom is None:
            u_nom_term = torch.zeros(
                (obs_local.shape[0], obs_local.shape[1], 3),
                dtype=obs_local.dtype,
                device=obs_local.device,
            )
        else:
            u_nom_term = u_nom
        if not self.cfg.use_parametric_qp:
            h_diag = torch.full_like(u_nom_term, float(self.cfg.qp_h_base))
            f_lin = -anchor_w * h_diag * u_nom_term
            return h_diag, f_lin

        feat = self._build_features(obs_local, skill_idx, tau)
        h = self.backbone(feat)
        h_delta = self._softplus(self.qp_h_head(h))
        h_diag = float(self.cfg.qp_h_base) + h_delta + float(self.cfg.qp_h_min)
        f_delta = self.qp_f_head(h)
        # Base term keeps unconstrained optimum near u_nom, delta term learns task shaping.
        f_lin = -anchor_w * h_diag * u_nom_term + float(self.cfg.qp_f_scale) * f_delta
        return h_diag, f_lin

    def act(
        self,
        obs_local: torch.Tensor,
        skill_idx: torch.Tensor,
        tau: torch.Tensor,
        cbf_state: torch.Tensor,
        qp_target_pos: torch.Tensor | None = None,
        qp_target_vel: torch.Tensor | None = None,
        qp_skill_idx: torch.Tensor | None = None,
        qp_obstacle_A: torch.Tensor | None = None,
        qp_obstacle_b: torch.Tensor | None = None,
        qp_solver=None,
        qp_nominal_anchor_weight: float = 1.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Single-step rollout call, with batch size B=1 expected.
        mean, std, value = self.forward(obs_local, skill_idx, tau)
        if bool(self.cfg.qp_only):
            u_nom = torch.zeros_like(mean)
            logp = torch.zeros((mean.shape[0], mean.shape[1]), dtype=mean.dtype, device=mean.device)
            entropy_joint = torch.zeros((mean.shape[0],), dtype=mean.dtype, device=mean.device)
            qp_v_des = torch.zeros_like(mean)
            u_nom_for_obj = None
        else:
            dist = Normal(mean, std)
            if deterministic:
                u_nom = mean
            else:
                u_nom = dist.rsample()
            logp = dist.log_prob(u_nom).sum(dim=-1)  # (B, N)
            entropy_joint = dist.entropy().sum(dim=-1).mean(dim=-1)  # (B,)
            qp_v_des = u_nom
            u_nom_for_obj = u_nom

        qp_h_diag, qp_f = self.qp_objective_params(
            obs_local,
            skill_idx,
            tau,
            u_nom_for_obj,
            nominal_anchor_weight=qp_nominal_anchor_weight,
        )

        u_safe = qp_v_des
        slack = torch.zeros((u_nom.shape[0], 1), dtype=u_nom.dtype, device=u_nom.device)
        if qp_solver is not None:
            safe_list = []
            slack_list = []
            qp_fail_count = 0
            for b in range(u_nom.shape[0]):
                try:
                    target_b = None if qp_target_pos is None else qp_target_pos[b]
                    target_vel_b = None if qp_target_vel is None else qp_target_vel[b]
                    skill_b = None if qp_skill_idx is None else qp_skill_idx[b]
                    obs_A_b = None if qp_obstacle_A is None else qp_obstacle_A[b]
                    obs_b_b = None if qp_obstacle_b is None else qp_obstacle_b[b]
                    u_b, s_b = qp_solver.solve_torch(
                        cbf_state[b],
                        qp_v_des[b],
                        target_pos_t=target_b,
                        target_vel_t=target_vel_b,
                        skill_idx_t=skill_b,
                        obstacle_A_t=obs_A_b,
                        obstacle_b_t=obs_b_b,
                        h_diag_t=qp_h_diag[b],
                        f_t=qp_f[b],
                    )
                except Exception:
                    # Keep the loop alive while avoiding unsafe nominal-action bypass.
                    u_b = torch.zeros_like(u_nom[b])
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
            "entropy_joint": entropy_joint,  # (B,)
            "slack_aux": slack.squeeze(-1),  # (B,)
            "qp_fail_count": torch.as_tensor([qp_fail_count], dtype=torch.int32, device=obs_local.device),
            "mean": mean,
            "std": std,
            "qp_h_diag": qp_h_diag,
            "qp_f": qp_f,
        }


def skill_velocity_alignment_reward(
    v_safe: np.ndarray,
    skill_idx: np.ndarray,
    cruise_speed: float = 0.5,
    accelerate_speed: float = 0.9,
    decelerate_speed: float = 0.2,
) -> float:
    """Intrinsic reward for skill execution in velocity space.

    Skill mapping:
      0 cruise, 1 left, 2 right, 3 up, 4 down, 5 accelerate, 6 decelerate
    """
    pref = np.array(
        [
            [1.0, 0.0, 0.0],   # cruise (forward)
            [0.0, -1.0, 0.0],  # left
            [0.0, 1.0, 0.0],   # right
            [0.0, 0.0, 1.0],   # up
            [0.0, 0.0, -1.0],  # down
            [1.0, 0.0, 0.0],   # accelerate (forward faster)
            [1.0, 0.0, 0.0],   # decelerate (forward slower)
        ],
        dtype=np.float32,
    )
    v = np.asarray(v_safe, dtype=np.float32)
    z = np.asarray(skill_idx, dtype=np.int64).reshape(-1)
    eps = 1e-6
    speed_sigma = 0.25
    vals = []
    for i in range(v.shape[0]):
        zi = int(z[i])
        p = pref[zi]
        if zi in (0, 5, 6):
            if zi == 0:
                speed_ref = float(cruise_speed)
            elif zi == 5:
                speed_ref = float(accelerate_speed)
            else:
                speed_ref = float(decelerate_speed)
            v_forward = float(v[i, 0])
            v_lat = float(np.linalg.norm(v[i, 1:3]))
            speed_term = float(np.exp(-((v_forward - speed_ref) ** 2) / (2.0 * speed_sigma**2)))
            lat_pen = 0.2 * v_lat
            vals.append(float(np.clip(speed_term - lat_pen, -1.0, 1.0)))
        else:
            vals.append(float(np.dot(v[i], p) / (np.linalg.norm(v[i]) + eps)))
    return float(np.mean(vals))
