"""Task-agnostic latent state/delta manifold CEM planner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.lewm import aligned_rollout_latents
from src.manifold import LatentManifold


@dataclass(frozen=True)
class PlannerConfig:
    history_size: int
    action_block: int = 1
    plan_blocks: int = 16
    samples: int = 256
    topk: int = 32
    iters: int = 4
    init_std: float = 0.015
    min_std: float = 0.003
    max_std: float | None = None
    momentum: float = 0.15
    cost_scale_min: float = 1e-4
    # Prior scaffolding knobs (ablation surface). With a well-trained
    # action-sensitive predictor the prior should be a soft regulariser
    # rather than a structural cheat. Defaults mirror the "prior wins by
    # construction" behaviour; the ablation figures flip these.
    prior_in_mean: bool = True
    prior_in_samples: bool = True
    prior_tie_abs: float = 0.0

    @property
    def horizon(self) -> int:
        return self.action_block * self.plan_blocks


@dataclass(frozen=True)
class PlannerWeights:
    state: float = 0.3
    delta: float = 4.0
    near: float = 0.01
    action_prior: float = 0.01
    smooth: float = 0.08
    energy: float = 0.002

    def as_dict(self) -> dict[str, float]:
        return {
            "W_STATE": self.state,
            "W_DELTA": self.delta,
            "W_NEAR": self.near,
            "W_ACTION_PRIOR": self.action_prior,
            "W_SMOOTH": self.smooth,
            "W_ENERGY": self.energy,
        }


def block_actions_from_step_actions(actions: torch.Tensor, cfg: PlannerConfig) -> torch.Tensor:
    blocks = actions[: cfg.horizon].reshape(cfg.plan_blocks, cfg.action_block, -1)
    return blocks.mean(dim=1)


def weighted_horizon_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights[: values.size(1)].to(values)
    return (values * w.view(1, -1)).sum(dim=1) / w.sum().clamp_min(1e-8)


def nearest_point_sequence_mse(
    sequence: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    samples, horizon, dim = sequence.shape
    if points.size(0) == 0:
        return sequence.new_zeros((samples,))
    flat = sequence.float().reshape(samples * horizon, dim)
    nearest = torch.cdist(flat, points.float()).pow(2).min(dim=1).values.div(dim)
    nearest = nearest.view(samples, horizon)
    return weighted_horizon_mean(nearest, weights)


class LatentCEMPlanner:
    def __init__(self, model, cfg: PlannerConfig, weights: PlannerWeights):
        self.model = model
        self.cfg = cfg
        self.weights = weights

    @torch.no_grad()
    def plan(
        self,
        *,
        history_pixels: torch.Tensor,
        history_actions: torch.Tensor,
        current_latent: torch.Tensor,
        target_latents: torch.Tensor,
        manifold: LatentManifold,
        prior_actions: torch.Tensor,
        previous_action: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        horizon_weights: torch.Tensor,
        rng: torch.Generator,
        init_std: float | None = None,
        action_prior_weight: float | None = None,
        smooth_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_dim = low.numel()
        prior_blocks = block_actions_from_step_actions(prior_actions, self.cfg).to(low.device)
        base_mean = prior_blocks if self.cfg.prior_in_mean else torch.zeros_like(prior_blocks)
        mean = torch.clamp(base_mean, low, high)
        std_value = float(self.cfg.init_std if init_std is None else init_std)
        if self.cfg.max_std is not None:
            std_value = min(std_value, float(self.cfg.max_std))
        std = torch.full_like(mean, std_value)
        action_prior_weight_value = (
            self.weights.action_prior if action_prior_weight is None else float(action_prior_weight)
        )
        smooth_weight_value = self.weights.smooth if smooth_weight is None else float(smooth_weight)

        history_actions = history_actions.to(low.device)
        current_latent = current_latent.to(low.device).float()
        target_latents = target_latents.to(low.device).float()
        state_points = manifold.state_latents.to(low.device).float()
        delta_points = manifold.delta_latents.to(low.device).float()
        state_scale = max(float(manifold.state_scale), self.cfg.cost_scale_min)
        delta_scale = max(float(manifold.delta_scale), self.cfg.cost_scale_min)

        best_plan: torch.Tensor | None = None
        best_info: dict[str, float] | None = None
        best_cost = float("inf")
        prior_plan: torch.Tensor | None = None
        prior_info: dict[str, float] | None = None
        prior_cost_value = float("inf")

        for _ in range(self.cfg.iters):
            eps = torch.randn(
                (self.cfg.samples, self.cfg.plan_blocks, action_dim),
                generator=rng,
                device=low.device,
            )
            block_samples = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * eps, low, high)
            if self.cfg.prior_in_samples:
                block_samples[0] = prior_blocks
            if self.cfg.samples > 1:
                block_samples[1] = mean
            step_samples = block_samples.repeat_interleave(self.cfg.action_block, dim=1)

            pred = aligned_rollout_latents(
                self.model,
                history_pixels,
                history_actions,
                step_samples.unsqueeze(0),
                history_size=self.cfg.history_size,
            ).squeeze(0)

            samples, _, dim = pred.shape
            current = current_latent.view(1, 1, dim).expand(samples, 1, dim)
            pred_with_current = torch.cat([current, pred.float()], dim=1)
            pred_deltas = pred_with_current[:, 1:] - pred_with_current[:, :-1]
            goal_state_cost = (
                nearest_point_sequence_mse(pred.float(), state_points, horizon_weights) / state_scale
            )
            goal_delta_cost = (
                nearest_point_sequence_mse(pred_deltas, delta_points, horizon_weights) / delta_scale
            )

            goal = target_latents.unsqueeze(0).expand_as(pred)
            near_raw = (pred.float() - goal).pow(2).mean(dim=-1)
            near_cost = weighted_horizon_mean(near_raw, horizon_weights) / state_scale

            # Prior cost is measured in CEM-std units so a "low" weight is still
            # numerically visible to the optimiser even when actions are O(1).
            prior_scale = max(std_value**2, 1e-8)
            prior_cost = (
                (block_samples - prior_blocks.unsqueeze(0)).pow(2).mean(dim=(1, 2))
                / prior_scale
            )
            prev = torch.cat(
                [
                    previous_action.view(1, 1, -1).expand(self.cfg.samples, 1, -1),
                    block_samples[:, :-1],
                ],
                dim=1,
            )
            smooth_cost = (block_samples - prev).pow(2).mean(dim=(1, 2))
            energy_cost = block_samples.pow(2).mean(dim=(1, 2))

            weighted_state = self.weights.state * goal_state_cost
            weighted_delta = self.weights.delta * goal_delta_cost
            weighted_near = self.weights.near * near_cost
            weighted_prior = action_prior_weight_value * prior_cost
            weighted_smooth = smooth_weight_value * smooth_cost
            weighted_energy = self.weights.energy * energy_cost
            cost = (
                weighted_state
                + weighted_delta
                + weighted_near
                + weighted_prior
                + weighted_smooth
                + weighted_energy
            )

            def make_info(i: int) -> dict[str, float]:
                denom = cost[i].abs().clamp_min(1e-8)
                return {
                    "state_cost": float(goal_state_cost[i].item()),
                    "delta_cost": float(goal_delta_cost[i].item()),
                    "near_cost": float(near_cost[i].item()),
                    "prior_cost": float(prior_cost[i].item()),
                    "smooth_cost": float(smooth_cost[i].item()),
                    "energy_cost": float(energy_cost[i].item()),
                    "planner_init_std": float(std_value),
                    "effective_action_prior_weight": float(action_prior_weight_value),
                    "effective_smooth_weight": float(smooth_weight_value),
                    "weighted_state_cost": float(weighted_state[i].item()),
                    "weighted_delta_cost": float(weighted_delta[i].item()),
                    "weighted_near_cost": float(weighted_near[i].item()),
                    "weighted_prior_cost": float(weighted_prior[i].item()),
                    "weighted_smooth_cost": float(weighted_smooth[i].item()),
                    "weighted_energy_cost": float(weighted_energy[i].item()),
                    "total_cost": float(cost[i].item()),
                    "state_cost_share": float((weighted_state[i] / denom).item()),
                    "delta_cost_share": float((weighted_delta[i] / denom).item()),
                    "near_cost_share": float((weighted_near[i] / denom).item()),
                    "prior_cost_share": float((weighted_prior[i] / denom).item()),
                    "smooth_cost_share": float((weighted_smooth[i] / denom).item()),
                    "energy_cost_share": float((weighted_energy[i] / denom).item()),
                }

            if float(cost[0].item()) < prior_cost_value:
                prior_cost_value = float(cost[0].item())
                prior_plan = block_samples[0].clone()
                prior_info = make_info(0)

            topk = min(self.cfg.topk, self.cfg.samples)
            elite_idx = torch.topk(cost, topk, largest=False).indices
            elites = block_samples[elite_idx]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(self.cfg.min_std)
            mean = (1.0 - self.cfg.momentum) * elite_mean + self.cfg.momentum * mean
            std = elite_std

            i_best = int(torch.argmin(cost).item())
            if float(cost[i_best]) < best_cost:
                best_cost = float(cost[i_best])
                best_plan = block_samples[i_best].clone()
                best_info = make_info(i_best)

        assert best_plan is not None and best_info is not None
        assert prior_plan is not None and prior_info is not None
        improvement = prior_cost_value - best_cost
        margin = float(self.cfg.prior_tie_abs)
        if margin > 0.0 and improvement <= margin:
            prior_info.update({
                "used_prior_guard": 1.0,
                "prior_total_cost": prior_cost_value,
                "cem_best_total_cost": best_cost,
                "prior_cost_improvement": float(improvement),
                "prior_guard_margin": float(margin),
            })
            return prior_plan, prior_info
        best_info.update({
            "used_prior_guard": 0.0,
            "prior_total_cost": prior_cost_value,
            "cem_best_total_cost": best_cost,
            "prior_cost_improvement": float(improvement),
            "prior_guard_margin": float(margin),
        })
        return best_plan, best_info
