"""Hopper hop task: one-button train + eval + video render.

CLI:
    python -m simulation.lewm.hopper_hop \
        --train-steps 50000 --device cuda

The architecture (Adaptive CEM + score-grounded memory):
- LeWM learns physics from real (s, a, s') transitions only.
- AdaptiveCEMPlanner proposes actions via three sources: colored noise +
  motor primitives + scored elite memory; rolls out futures through LeWM;
  scores via HopperStateCostModel.
- At episode end, the real environment's `task_score` is written back
  into both the replay buffer (per-transition) and the elite memory
  (per-window). Future planning is biased toward what *actually* worked,
  not toward what the model thought would work.

After train_steps, the same script runs a final evaluation pass with full
metric printout and saves a video of the best episode.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from simulation.lewm.cem import AdaptiveCEMConfig, AdaptiveCEMPlanner, MotorPrimitive
from simulation.lewm.common import (
    ScoredTransitionReplay,
    StrategyBundle,
    find_repo_root,
    flatten_obs,
    get_stablewm_home,
    make_dm_env,
    physics_scalar,
    rollout_episode,
    save_video,
)
from simulation.lewm.lewm import (
    JEPA, LeWMTrainConfig, SIGReg, build_lewm_model, dynamics_loss_for_batch,
)


# -----------------------------------------------------------------------------
# Hopper observation layout (dm_control hopper.hop: 15-dim obs)
#   position: 6, velocity: 7, touch: 2
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HopperObsLayout:
    position: slice = slice(0, 6)
    height: int = 0
    angle: int = 1
    velocity: slice = slice(6, 13)
    touch: slice = slice(13, 15)


HOPPER_OBS = HopperObsLayout()


def _hopper_upright_from_obs(obs_vec: np.ndarray) -> float:
    angle = float(np.asarray(obs_vec, dtype=np.float32)[HOPPER_OBS.angle])
    return float(np.cos(angle))


# -----------------------------------------------------------------------------
# Cost
# -----------------------------------------------------------------------------

@dataclass
class HopperCostConfig:
    history_size: int = 3
    progress_weight: float = 7.0
    speed_weight: float = 1.0
    height_weight: float = 2.5
    upright_weight: float = 2.5
    alive_weight: float = 2.0
    action_energy_weight: float = 0.02
    action_jerk_weight: float = 0.05
    target_speed: float = 2.0
    stand_height: float = 0.6
    alive_height: float = 0.35


class HopperStateCostModel(torch.nn.Module):
    """CEM cost over LeWM-predicted future states.

    Score = w_p·forward + w_s·speed_match + w_h·height + w_u·upright + w_a·alive
    Cost  = w_e·energy + w_j·jerk − Score

    Alive is sigmoid-smoothed across the height/upright thresholds so CEM has
    a non-zero gradient signal even when every candidate is "fallen".
    """

    def __init__(self, model: JEPA, cfg: HopperCostConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.history_size = int(cfg.history_size)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info_dict.items()}
        action = torch.clamp(action_candidates.to(device), -1.0, 1.0)
        rollout = self.model.rollout_states(info, action, history_size=self.history_size)
        future = rollout["predicted_state"][..., self.history_size :, :]

        angle = future[..., HOPPER_OBS.angle]
        upright = torch.clamp(torch.cos(angle), 0.0, 1.0)
        height = future[..., HOPPER_OBS.height]
        speed = future[..., HOPPER_OBS.velocity.start]
        target_speed = max(float(self.cfg.target_speed), 1e-6)

        forward = torch.clamp(speed, -target_speed, target_speed * 2.0).mean(dim=-1) / target_speed
        speed_match = torch.clamp(1.0 - (speed - target_speed).abs() / target_speed, 0.0, 1.0).mean(dim=-1)
        height_score = torch.clamp(height / self.cfg.stand_height, 0.0, 1.0).mean(dim=-1)
        upright_score = upright.mean(dim=-1)
        alive_h = torch.sigmoid(8.0 * (height - self.cfg.alive_height))
        alive_u = torch.sigmoid(8.0 * (upright - 0.25))
        alive = (alive_h * alive_u).mean(dim=-1)

        score = (
            self.cfg.progress_weight * forward
            + self.cfg.speed_weight * speed_match
            + self.cfg.height_weight * height_score
            + self.cfg.upright_weight * upright_score
            + self.cfg.alive_weight * alive
        )
        energy = action.square().mean(dim=(-1, -2))
        jerk = (
            torch.diff(action, dim=2).square().mean(dim=(-1, -2))
            if action.shape[2] > 1 else torch.zeros_like(energy)
        )
        return self.cfg.action_energy_weight * energy + self.cfg.action_jerk_weight * jerk - score


# -----------------------------------------------------------------------------
# Hopper motor primitives
# -----------------------------------------------------------------------------

class ExtensionPulse(MotorPrimitive):
    name = "extension_pulse"

    def sample_params(self, rng):
        return {
            "hip": float(rng.uniform(-0.4, 0.2)),
            "knee": float(rng.uniform(-1.0, -0.25)),
            "ankle": float(rng.uniform(-0.8, -0.15)),
            "duration": int(rng.integers(3, 10)),
        }

    def generate(self, params, horizon, action_dim, device):
        actions = torch.zeros((horizon, action_dim), dtype=torch.float32, device=device)
        d = min(int(params["duration"]), horizon)
        if action_dim >= 4:
            actions[:d, 0] = params["hip"]
            actions[:d, 1] = params["knee"]
            actions[:d, 2] = params["ankle"]
            actions[:d, 3] = 0.5 * params["ankle"]
        else:
            actions[:d] = params["knee"]
        return actions


class ForwardLean(MotorPrimitive):
    name = "forward_lean"

    def sample_params(self, rng):
        return {
            "lean": float(rng.uniform(0.15, 0.6)),
            "knee": float(rng.uniform(-0.25, 0.25)),
            "period": float(rng.uniform(18.0, 45.0)),
            "phase": float(rng.uniform(0.0, 2.0 * float(np.pi))),
        }

    def generate(self, params, horizon, action_dim, device):
        t = torch.arange(horizon, dtype=torch.float32, device=device)
        omega = 2.0 * float(np.pi) / max(float(params["period"]), 1.0)
        wave = 0.25 * torch.sin(omega * t + float(params["phase"]))
        actions = torch.zeros((horizon, action_dim), dtype=torch.float32, device=device)
        if action_dim >= 4:
            actions[:, 0] = params["lean"] + wave
            actions[:, 1] = params["knee"] - wave
            actions[:, 2] = -0.2 * params["lean"]
        else:
            actions[:] = params["lean"]
        return actions


class HopCycle(MotorPrimitive):
    name = "hop_cycle"

    def sample_params(self, rng):
        return {
            "amp": float(rng.uniform(0.2, 0.6)),
            "period": float(rng.uniform(16.0, 40.0)),
            "phase": float(rng.uniform(0.0, 2.0 * float(np.pi))),
        }

    def generate(self, params, horizon, action_dim, device):
        t = torch.arange(horizon, dtype=torch.float32, device=device)
        omega = 2.0 * float(np.pi) / max(float(params["period"]), 1.0)
        wave = float(params["amp"]) * torch.sin(omega * t + float(params["phase"]))
        actions = torch.zeros((horizon, action_dim), dtype=torch.float32, device=device)
        if action_dim >= 4:
            actions[:, 0] = wave
            actions[:, 1] = -wave.abs()
            actions[:, 2] = -0.7 * wave.abs()
            actions[:, 3] = -0.4 * wave.abs()
        else:
            actions[:] = wave.unsqueeze(-1)
        return actions


class BodyRocking(MotorPrimitive):
    name = "body_rocking"

    def sample_params(self, rng):
        return {
            "amp": float(rng.uniform(0.1, 0.4)),
            "period": float(rng.uniform(20.0, 50.0)),
            "phase": float(rng.uniform(0.0, 2.0 * float(np.pi))),
            "offset": float(rng.uniform(-0.2, 0.2)),
        }

    def generate(self, params, horizon, action_dim, device):
        t = torch.arange(horizon, dtype=torch.float32, device=device)
        omega = 2.0 * float(np.pi) / max(float(params["period"]), 1.0)
        wave = float(params["amp"]) * torch.sin(omega * t + float(params["phase"]))
        return (float(params["offset"]) + wave).unsqueeze(-1).expand(-1, action_dim).clone()


def hopper_primitives() -> list[MotorPrimitive]:
    return [ExtensionPulse(), ForwardLean(), HopCycle(), BodyRocking()]


# -----------------------------------------------------------------------------
# Episode quality metric (used as score for replay + elite buffer)
# -----------------------------------------------------------------------------

def compute_hopper_metrics(
    rewards: np.ndarray, actions: np.ndarray, heights: np.ndarray,
    uprights: np.ndarray, speeds: np.ndarray, xpos: np.ndarray,
    x_start: float, max_steps: int,
) -> dict[str, Any]:
    steps = len(rewards)
    if steps == 0:
        return {
            "episode_return": 0.0, "official_score": 0.0, "task_score": 0.0,
            "progress_m": 0.0, "avg_speed_mps": 0.0, "mean_upright_01": 0.0,
            "stand_fraction": 0.0, "stable_fraction": 0.0, "action_smoothness": 0.0,
            "first_collapse_step": None, "grade_label": "0档：还没跳起来",
            "num_steps": 0,
        }
    upright01 = np.clip((uprights + 1.0) * 0.5, 0.0, 1.0)
    stand_fraction = float(np.mean(heights >= 0.6))
    stable_mask = (heights >= 0.45) & (upright01 >= 0.5)
    stable_fraction = float(np.mean(stable_mask))
    collapse_mask = (heights < 0.30) | (upright01 < 0.25)
    collapse_indices = np.where(collapse_mask)[0]
    first_collapse_step = int(collapse_indices[0] + 1) if len(collapse_indices) > 0 else None
    progress_m = float(xpos[-1] - x_start) if len(xpos) > 0 else 0.0
    avg_speed_mps = float(np.mean(np.maximum(speeds, 0.0))) if len(speeds) > 0 else 0.0
    mean_upright_01 = float(np.mean(upright01))
    if len(actions) > 1:
        delta = np.diff(actions, axis=0)
        mean_action_delta_sq = float(np.mean(np.sum(delta ** 2, axis=1)))
    else:
        mean_action_delta_sq = 10.0
    action_smoothness = float(1.0 / (1.0 + 4.0 * mean_action_delta_sq))
    episode_return = float(np.sum(rewards))

    locomotion_quality = np.clip(progress_m / 10.0, 0.0, 1.0)
    speed_quality = np.clip(avg_speed_mps / 2.0, 0.0, 1.0)
    posture_quality = np.clip(0.5 * mean_upright_01 + 0.5 * stand_fraction, 0.0, 1.0)
    stability_quality = np.clip(stable_fraction, 0.0, 1.0)
    smoothness_quality = np.clip(action_smoothness, 0.0, 1.0)
    task_score = 1000.0 * (
        0.30 * locomotion_quality + 0.20 * speed_quality + 0.20 * stability_quality
        + 0.20 * posture_quality + 0.10 * smoothness_quality
    )

    if progress_m < 0.5 and stand_fraction < 0.2:
        grade = "0档：基本没跳起来"
    elif first_collapse_step is not None and first_collapse_step < 120:
        grade = "1档：跳了几步就倒"
    elif task_score < 450:
        grade = "2档：能往前跳，但歪歪扭扭"
    elif task_score < 750:
        grade = "3档：能比较稳定地跳完"
    else:
        grade = "4档：跳得稳、顺，而且比较像样"

    return {
        "episode_return": episode_return,
        "official_score": episode_return,
        "task_score": float(task_score),
        "progress_m": progress_m,
        "avg_speed_mps": avg_speed_mps,
        "mean_upright_01": mean_upright_01,
        "stand_fraction": stand_fraction,
        "stable_fraction": stable_fraction,
        "action_smoothness": action_smoothness,
        "first_collapse_step": first_collapse_step,
        "grade_label": grade,
        "num_steps": steps,
    }


def summarize_episodes(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    official = np.asarray([r["official_score"] for r in results], dtype=np.float32)
    task = np.asarray([r["task_score"] for r in results], dtype=np.float32)
    progresses = np.asarray([r["progress_m"] for r in results], dtype=np.float32)
    speeds = np.asarray([r["avg_speed_mps"] for r in results], dtype=np.float32)
    return {
        "official_score_mean": float(official.mean()),
        "official_score_std": float(official.std()),
        "task_score_mean": float(task.mean()),
        "task_score_std": float(task.std()),
        "progress_mean": float(progresses.mean()),
        "speed_mean": float(speeds.mean()),
    }


# -----------------------------------------------------------------------------
# Hopper policy: state/action history bookkeeping + Adaptive CEM
# -----------------------------------------------------------------------------

class HopperLeWMPolicy:
    """dm_control-compatible policy wrapping AdaptiveCEMPlanner."""

    def __init__(
        self,
        model: JEPA,
        cost_model: HopperStateCostModel,
        cem_cfg: AdaptiveCEMConfig,
        primitives: list[MotorPrimitive],
        rng: np.random.Generator,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.cost_model = cost_model.to(device).eval()
        self.history_size = int(cost_model.history_size)
        self.plan_every = int(cem_cfg.plan_every)
        self._planner = AdaptiveCEMPlanner(
            cost_model=self.cost_model, rng=rng, cfg=cem_cfg,
            primitives=primitives, history_size=self.history_size,
        )
        self.base_action_dim: int | None = None
        self._state_history: deque[torch.Tensor] = deque(maxlen=self.history_size)
        self._action_history: deque[torch.Tensor] = deque(maxlen=self.history_size)
        self._action_queue: deque[np.ndarray] = deque()
        self._has_planned = False
        self._last_plan_was_forced = False
        self._forced_action_count = 0
        self.last_plan_score: float | None = None

    # ----- env interface --------------------------------------------------------

    def bind_env(self, env) -> None:
        action_spec = env.action_spec()
        self.base_action_dim = int(np.prod(action_spec.shape))
        self._planner.configure(action_spec)

    def reset(self) -> None:
        self._state_history.clear()
        self._action_history.clear()
        self._action_queue.clear()
        self._has_planned = False
        self._last_plan_was_forced = False
        self._planner.reset()
        self.last_plan_score = None

    def set_step(self, step: int) -> None:
        self._planner.set_step(step)

    def record_action(self, action: np.ndarray) -> None:
        assert self.base_action_dim is not None
        self._action_history.append(
            torch.as_tensor(action, dtype=torch.float32).reshape(self.base_action_dim)
        )

    # ----- proxies for diagnostics ---------------------------------------------

    def primitive_exec_counts(self) -> dict[str, int]:
        return self._planner.primitive_exec_counts()

    def forced_action_count(self) -> int:
        return self._forced_action_count

    def primitive_mixture_weight(self) -> float:
        return self._planner.primitive_mixture_weight()

    def forced_exploration_prob(self) -> float:
        return self._planner.forced_exploration_prob()

    def elite_summary(self) -> dict[str, float]:
        return self._planner.elite_buffer.score_summary()

    def add_episode_to_elite(self, actions: np.ndarray, score: float) -> int:
        return self._planner.add_episode_to_elite(actions, score)

    # ----- internal -------------------------------------------------------------

    def _update_history(self, obs_vec) -> None:
        state = torch.as_tensor(obs_vec, dtype=torch.float32)
        if not self._state_history:
            for _ in range(self.history_size):
                self._state_history.append(state)
        else:
            self._state_history.append(state)

    def _build_info(self) -> dict[str, torch.Tensor]:
        assert self.base_action_dim is not None
        missing = self.history_size - len(self._action_history)
        action_history = [
            torch.zeros(self.base_action_dim, dtype=torch.float32) for _ in range(missing)
        ]
        action_history.extend(self._action_history)
        return {
            "observation": torch.stack(list(self._state_history), dim=0).unsqueeze(0).unsqueeze(0),
            "action": torch.stack(action_history, dim=0).unsqueeze(0).unsqueeze(0),
        }

    def _maybe_pick_forced_primitive(self):
        p = self._planner.forced_exploration_prob()
        if p <= 0.0:
            return None
        if float(self._planner.rng.random()) >= p:
            return None
        return self._planner.pick_forced_primitive()

    # ----- act ------------------------------------------------------------------

    def act(self, obs_vec, normalizer, action_spec) -> np.ndarray:
        self._update_history(obs_vec)
        if not self._action_queue:
            if self._has_planned:
                self._planner.shift_warm_start(self.plan_every)

            forced = self._maybe_pick_forced_primitive()
            if forced is not None:
                plan = self._planner.generate_primitive_plan(forced)
                self._last_plan_was_forced = True
                self.last_plan_score = None
            else:
                self.model.eval()
                self.cost_model.eval()
                with torch.inference_mode():
                    outputs = self._planner.solve(self._build_info())
                plan = outputs["actions"][0]
                costs = outputs.get("costs")
                self.last_plan_score = (
                    -float(costs.detach().cpu().float().mean())
                    if torch.is_tensor(costs) else None
                )
                self._last_plan_was_forced = False

            n_executed = min(self.plan_every, plan.shape[0])
            for i in range(n_executed):
                self._action_queue.append(
                    plan[i].detach().cpu().numpy().astype(np.float32)
                )
            if forced is not None:
                self._planner.record_primitive_executed(forced, n_executed)
                self._forced_action_count += n_executed
            self._has_planned = True

        action = self._action_queue.popleft()
        return np.clip(action, action_spec.minimum, action_spec.maximum).astype(np.float32)

    def exploration_action(self, action_spec, obs_vec=None) -> np.ndarray:
        if obs_vec is not None:
            self._update_history(obs_vec)
        action = self._planner.exploration_action()
        return np.clip(action, action_spec.minimum, action_spec.maximum).astype(np.float32)


# -----------------------------------------------------------------------------
# Top-level config
# -----------------------------------------------------------------------------

@dataclass
class HopperTaskConfig:
    # I/O
    dataset_name: str = "hopper_hop"
    run_name: str = "hopper/hop"
    seed: int = 42
    device: str = "cpu"
    # Training schedule
    train_steps: int = 50_000
    random_steps: int = 100
    updates_per_step: int = 1
    batch_size: int = 256
    sequence_length: int = 16
    replay_capacity: int = 1_000_000
    max_episode_steps: int = 500
    fall_reset_height: float = 0.30
    fall_reset_grace: int = 8
    eval_every: int = 1_000
    eval_episodes: int = 1
    eval_episode_steps: int = 200
    checkpoint_every: int = 10_000
    log_every: int = 1
    score_weight_alpha: float = 0.5
    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 1e-3
    precision: str = "fp32"
    grad_clip_norm: float = 1.0
    normalizer_update_every: int = 1000
    exploration_noise: float = 0.10
    # Final eval / video
    final_eval_episodes: int = 5
    final_eval_episode_steps: int = 1000
    video_path: str = "hopper_best_episode.mp4"
    video_fps: int = 40
    render_width: int = 480
    render_height: int = 368
    camera_id: int = 0
    # Sub-configs (flat for ease of CLI)
    state_dim: int = 15
    history_size: int = 3
    state_hidden_dim: int = 256
    embed_dim: int = 192
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    sigreg_weight: float = 0.09
    state_loss_weight: float = 1.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    # Cost
    progress_weight: float = 7.0
    speed_weight: float = 1.0
    height_weight: float = 2.5
    upright_weight: float = 2.5
    alive_weight: float = 2.0
    action_energy_weight: float = 0.02
    action_jerk_weight: float = 0.05
    target_speed: float = 2.0
    stand_height: float = 0.6
    alive_height: float = 0.35
    # CEM
    plan_horizon: int = 32
    cem_num_samples: int = 256
    cem_n_steps: int = 4
    cem_topk: int = 32
    cem_var_scale: float = 1.0
    cem_init_std: float = 0.5
    cem_min_std: float = 0.05
    cem_noise_beta: float = 2.0
    plan_every: int = 4
    elite_buffer_size: int = 64
    elite_seed_fraction: float = 0.2
    elite_score_temperature: float = 100.0
    primitive_mixture_initial: float = 0.5
    primitive_mixture_final: float = 0.05
    primitive_mixture_anneal_steps: int = 5000
    forced_exploration_initial: float = 0.4
    forced_exploration_baseline: float = 0.05
    forced_primitive_quota: int = 500
    primitive_elite_min_fraction: float = 0.2
    exploration_std_scale: float = 1.5


def _train_cfg(c: HopperTaskConfig) -> LeWMTrainConfig:
    return LeWMTrainConfig(
        seed=c.seed, device=c.device, batch_size=c.batch_size, lr=c.lr,
        weight_decay=c.weight_decay, history_size=c.history_size,
        state_dim=c.state_dim, state_hidden_dim=c.state_hidden_dim,
        embed_dim=c.embed_dim, predictor_depth=c.predictor_depth,
        predictor_heads=c.predictor_heads, predictor_mlp_dim=c.predictor_mlp_dim,
        predictor_dim_head=c.predictor_dim_head, predictor_dropout=c.predictor_dropout,
        predictor_emb_dropout=c.predictor_emb_dropout,
        sigreg_weight=c.sigreg_weight, state_loss_weight=c.state_loss_weight,
        sigreg_knots=c.sigreg_knots, sigreg_num_proj=c.sigreg_num_proj,
        precision=c.precision,
    )


def _cost_cfg(c: HopperTaskConfig) -> HopperCostConfig:
    return HopperCostConfig(
        history_size=c.history_size,
        progress_weight=c.progress_weight, speed_weight=c.speed_weight,
        height_weight=c.height_weight, upright_weight=c.upright_weight,
        alive_weight=c.alive_weight,
        action_energy_weight=c.action_energy_weight, action_jerk_weight=c.action_jerk_weight,
        target_speed=c.target_speed, stand_height=c.stand_height, alive_height=c.alive_height,
    )


def _cem_cfg(c: HopperTaskConfig) -> AdaptiveCEMConfig:
    return AdaptiveCEMConfig(
        plan_horizon=c.plan_horizon, cem_num_samples=c.cem_num_samples,
        cem_n_steps=c.cem_n_steps, cem_topk=min(c.cem_topk, c.cem_num_samples),
        cem_init_std=c.cem_init_std, cem_min_std=c.cem_min_std,
        cem_noise_beta=c.cem_noise_beta, cem_var_scale=c.cem_var_scale,
        plan_every=c.plan_every,
        elite_buffer_size=c.elite_buffer_size,
        elite_seed_fraction=c.elite_seed_fraction,
        elite_score_temperature=c.elite_score_temperature,
        primitive_mixture_initial=c.primitive_mixture_initial,
        primitive_mixture_final=c.primitive_mixture_final,
        primitive_mixture_anneal_steps=c.primitive_mixture_anneal_steps,
        forced_exploration_initial=c.forced_exploration_initial,
        forced_exploration_baseline=c.forced_exploration_baseline,
        forced_primitive_quota=c.forced_primitive_quota,
        primitive_elite_min_fraction=c.primitive_elite_min_fraction,
        exploration_std_scale=c.exploration_std_scale,
    )


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def _update_stats(model: JEPA, replay: ScoredTransitionReplay) -> None:
    mean, std = replay.state_stats()
    device = model.state_mean.device
    with torch.no_grad():
        model.state_mean.copy_(torch.as_tensor(mean, device=device).view(1, 1, -1))
        model.state_std.copy_(torch.as_tensor(std, device=device).view(1, 1, -1))


def _tensor_batch(batch: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


def _model_update(
    model: JEPA, sigreg: SIGReg, optimizer, replay: ScoredTransitionReplay,
    train_cfg: LeWMTrainConfig, cfg: HopperTaskConfig, scaler,
) -> dict[str, float]:
    model.train()
    batch = _tensor_batch(
        replay.sample(cfg.batch_size, cfg.sequence_length, score_weight_alpha=cfg.score_weight_alpha),
        cfg.device,
    )
    use_amp = cfg.device.startswith("cuda") and cfg.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext():
        losses = dynamics_loss_for_batch(model, sigreg, batch, train_cfg)
    if scaler.is_enabled():
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def _save_checkpoint(
    path: Path, model: JEPA, optimizer, train_cfg: LeWMTrainConfig,
    cfg: HopperTaskConfig, action_dim: int, step: int, best_score: float,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "train_config": asdict(train_cfg),
            "task_config": asdict(cfg),
            "action_dim": action_dim,
            "state_dim": cfg.state_dim,
            "model_type": "lewm_hopper_hop_v2",
            "global_step": step,
            "best_score": best_score,
            "history": history,
        },
        path,
    )


def _latest_checkpoint(target: str | Path) -> Path | None:
    path = Path(target).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("lewm_best.pt", "lewm_latest.pt"):
            cand = path / name
            if cand.exists():
                return cand
        cands = sorted(path.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None
    exact = path.parent / f"{path.name}.pt"
    return exact if exact.exists() else None


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[JEPA, dict[str, Any]]:
    checkpoint = _latest_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint under {checkpoint_path}")
    payload = torch.load(checkpoint, map_location=device)
    train_cfg = LeWMTrainConfig(**payload["train_config"])
    model = build_lewm_model(
        train_cfg, int(payload["action_dim"]),
        state_dim=int(payload.get("state_dim", train_cfg.state_dim)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, {**payload, "checkpoint_file": str(checkpoint)}


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------

def _eval_single_episode(
    policy: HopperLeWMPolicy, env, max_steps: int, render: bool,
    width: int, height: int, camera_id: int, progress_desc: str | None,
) -> dict[str, Any]:
    out = rollout_episode(
        env=env, policy=policy, act_normalizer=None,
        max_steps=max_steps, stats_normalizer=None, render=render,
        width=width, height=height, camera_id=camera_id,
        progress_desc=progress_desc,
    )
    metrics = compute_hopper_metrics(
        rewards=out["rewards"], actions=out["actions"], heights=out["heights"],
        uprights=out["uprights"], speeds=out["speeds"], xpos=out["xpos"],
        x_start=out["x_start"], max_steps=max_steps,
    )
    metrics["frames"] = out["frames"]
    return metrics


def _build_eval_policy(model: JEPA, cfg: HopperTaskConfig, seed: int) -> HopperLeWMPolicy:
    cost_model = HopperStateCostModel(model, _cost_cfg(cfg))
    return HopperLeWMPolicy(
        model=model, cost_model=cost_model, cem_cfg=_cem_cfg(cfg),
        primitives=hopper_primitives(),
        rng=np.random.default_rng(seed),
        device=cfg.device,
    )


def _evaluate_model(
    model: JEPA, cfg: HopperTaskConfig, seed: int, *,
    replay: ScoredTransitionReplay | None = None,
    show_progress: bool = True, n_episodes: int | None = None,
    episode_steps: int | None = None,
) -> list[dict[str, Any]]:
    if replay is not None and replay.size >= 2:
        _update_stats(model, replay)
    policy = _build_eval_policy(model, cfg, seed)
    n = n_episodes if n_episodes is not None else cfg.eval_episodes
    steps = episode_steps if episode_steps is not None else cfg.eval_episode_steps
    env = make_dm_env("hopper", "hop", seed=seed)
    results = []
    for ep in range(n):
        results.append(_eval_single_episode(
            policy=policy, env=env, max_steps=steps, render=False,
            width=224, height=224, camera_id=0,
            progress_desc=f"eval episode {ep + 1}/{n}" if show_progress else None,
        ))
    return results


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def run_training(cfg: HopperTaskConfig) -> dict[str, Any]:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if cfg.sequence_length < cfg.history_size:
        raise ValueError("sequence_length must be >= history_size")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    stable_home = get_stablewm_home()
    run_dir = stable_home / "checkpoints" / cfg.run_name
    deploy_dir = stable_home / "checkpoints" / "hopper" / "lewm"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"

    env = make_dm_env("hopper", "hop", seed=cfg.seed)
    action_spec = env.action_spec()
    action_dim = int(np.prod(action_spec.shape))
    low = np.asarray(action_spec.minimum, dtype=np.float32)
    high = np.asarray(action_spec.maximum, dtype=np.float32)

    train_cfg = _train_cfg(cfg)
    model = build_lewm_model(train_cfg, action_dim, state_dim=cfg.state_dim).to(cfg.device)
    sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_num_proj).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.device.startswith("cuda") and cfg.precision == "fp16",
    )
    replay = ScoredTransitionReplay(
        cfg.replay_capacity, cfg.state_dim, action_dim, rng,
    )

    cost_model = HopperStateCostModel(model, _cost_cfg(cfg))
    planner = HopperLeWMPolicy(
        model=model, cost_model=cost_model, cem_cfg=_cem_cfg(cfg),
        primitives=hopper_primitives(),
        rng=np.random.default_rng(cfg.seed + 1),
        device=cfg.device,
    )
    planner.bind_env(env)

    time_step = env.reset()
    x_start = float(env.physics.named.data.xpos["torso", "x"])
    episode_step = 0
    consecutive_low_height = 0
    rewards: list[float] = []
    actions: list[np.ndarray] = []
    heights: list[float] = []
    uprights: list[float] = []
    speeds: list[float] = []
    xpos: list[float] = []
    recent_returns: deque[float] = deque(maxlen=20)
    recent_progress: deque[float] = deque(maxlen=20)
    recent_episode_lens: deque[int] = deque(maxlen=20)
    recent_fall_resets: deque[int] = deque(maxlen=20)
    recent_task_scores: deque[float] = deque(maxlen=20)
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    update_count = 0
    last_losses: dict[str, float] = {}
    start_time = time.time()

    pbar = tqdm(
        range(1, cfg.train_steps + 1),
        total=cfg.train_steps,
        desc="LeWM-Hopper", unit="step", dynamic_ncols=True,
    )
    for step in pbar:
        obs = flatten_obs(time_step.observation)
        planner.set_step(step)

        use_mpc = (
            step > cfg.random_steps
            and replay.can_sample(cfg.sequence_length)
            and episode_step >= cfg.history_size
        )
        if use_mpc:
            action = planner.act(obs, None, action_spec)
            if cfg.exploration_noise > 0:
                action = action + rng.normal(0.0, cfg.exploration_noise, size=action.shape).astype(np.float32)
        else:
            action = planner.exploration_action(action_spec, obs_vec=obs)
        action = np.clip(action, low, high).astype(np.float32)

        next_ts = env.step(action)
        planner.record_action(action)
        next_obs = flatten_obs(next_ts.observation)
        cur_height = physics_scalar(env.physics, ("height", "torso_height"), default=0.0)

        if cur_height < cfg.fall_reset_height:
            consecutive_low_height += 1
        else:
            consecutive_low_height = 0
        fall_reset = (
            cfg.fall_reset_grace > 0 and consecutive_low_height >= cfg.fall_reset_grace
        )
        done = bool(
            next_ts.last() or episode_step + 1 >= cfg.max_episode_steps or fall_reset
        )
        replay.append(obs, action, next_obs, done)

        rewards.append(float(next_ts.reward or 0.0))
        actions.append(action.copy())
        heights.append(cur_height)
        uprights.append(_hopper_upright_from_obs(next_obs))
        speeds.append(physics_scalar(env.physics, ("speed", "horizontal_velocity"), default=0.0))
        xpos.append(float(env.physics.named.data.xpos["torso", "x"]))
        episode_step += 1

        if replay.size >= 2 and cfg.normalizer_update_every and step % cfg.normalizer_update_every == 0:
            _update_stats(model, replay)
        if replay.can_sample(cfg.sequence_length):
            for _ in range(cfg.updates_per_step):
                last_losses = _model_update(model, sigreg, optimizer, replay, train_cfg, cfg, scaler)
                update_count += 1

        if done:
            m = compute_hopper_metrics(
                rewards=np.asarray(rewards, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                heights=np.asarray(heights, dtype=np.float32),
                uprights=np.asarray(uprights, dtype=np.float32),
                speeds=np.asarray(speeds, dtype=np.float32),
                xpos=np.asarray(xpos, dtype=np.float32),
                x_start=x_start, max_steps=cfg.max_episode_steps,
            )
            # Score-feedback to BOTH the replay buffer (per-transition) and
            # the elite memory (per-window). The score is the real-env
            # task_score, so the model never gets to vote on what's
            # "good enough" to remember.
            ep_score = float(m["task_score"])
            replay.assign_recent_score(episode_step, ep_score)
            if len(actions) >= cfg.plan_horizon:
                planner.add_episode_to_elite(
                    np.asarray(actions, dtype=np.float32), ep_score,
                )
            recent_returns.append(float(m["official_score"]))
            recent_progress.append(float(m["progress_m"]))
            recent_episode_lens.append(int(episode_step))
            recent_fall_resets.append(1 if fall_reset else 0)
            recent_task_scores.append(ep_score)

            time_step = env.reset()
            planner.reset()
            x_start = float(env.physics.named.data.xpos["torso", "x"])
            episode_step = 0
            consecutive_low_height = 0
            rewards.clear(); actions.clear(); heights.clear()
            uprights.clear(); speeds.clear(); xpos.clear()
        else:
            time_step = next_ts

        if step % max(cfg.log_every, 1) == 0:
            elite = planner.elite_summary()
            pbar.set_postfix(
                mode="mpc" if use_mpc else "rand",
                replay=replay.size,
                upd=update_count,
                loss=f"{last_losses.get('loss', float('nan')):.3f}",
                eplen=f"{np.mean(recent_episode_lens) if recent_episode_lens else 0.0:.0f}",
                fall=f"{np.mean(recent_fall_resets) if recent_fall_resets else 0.0:.2f}",
                task=f"{np.mean(recent_task_scores) if recent_task_scores else 0.0:.0f}",
                pmix=f"{planner.primitive_mixture_weight():.2f}",
                pforce=f"{planner.forced_exploration_prob():.2f}",
                forced=planner.forced_action_count(),
                elite=f"{elite.get('count', 0)}/{cfg.elite_buffer_size}",
            )

        if (cfg.eval_every > 0 and step % cfg.eval_every == 0) or step == cfg.train_steps:
            pbar.clear()
            try:
                eval_results = _evaluate_model(
                    model, cfg, cfg.seed + 10_000 + step,
                    replay=replay, show_progress=True,
                )
                summary = summarize_episodes(eval_results)
            finally:
                pbar.refresh()
            improved = summary["official_score_mean"] > best_score
            best_score = max(best_score, summary["official_score_mean"])
            elite = planner.elite_summary()
            row = {
                "step": step, "updates": update_count, "replay_size": replay.size,
                "loss": last_losses.get("loss", np.nan),
                "latent_loss": last_losses.get("latent_loss", np.nan),
                "state_loss": last_losses.get("state_loss", np.nan),
                "sigreg_loss": last_losses.get("sigreg_loss", np.nan),
                "episode_return_recent": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "episode_progress_recent": float(np.mean(recent_progress)) if recent_progress else 0.0,
                "episode_len_recent": float(np.mean(recent_episode_lens)) if recent_episode_lens else 0.0,
                "fall_reset_rate_recent": float(np.mean(recent_fall_resets)) if recent_fall_resets else 0.0,
                "task_score_recent": float(np.mean(recent_task_scores)) if recent_task_scores else 0.0,
                "primitive_mixture_weight": float(planner.primitive_mixture_weight()),
                "forced_exploration_prob": float(planner.forced_exploration_prob()),
                "forced_action_count": int(planner.forced_action_count()),
                "elite_buffer_size": int(elite.get("count", 0)),
                "elite_mean_score": float(elite.get("mean_score", 0.0)),
                "elite_max_score": float(elite.get("max_score", 0.0)),
                **{f"primitive_count_{name}": int(c) for name, c in planner.primitive_exec_counts().items()},
                **summary,
                "elapsed_sec": time.time() - start_time,
                "best": improved,
            }
            rows.append(row)
            with metrics_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            latest = run_dir / "lewm_latest.pt"
            _save_checkpoint(latest, model, optimizer, train_cfg, cfg, action_dim, step, best_score, rows)
            if improved:
                best = run_dir / "lewm_best.pt"
                _save_checkpoint(best, model, optimizer, train_cfg, cfg, action_dim, step, best_score, rows)
                deploy_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best, deploy_dir / "lewm_best.pt")
            tqdm.write(
                f"[eval step {step}] score={summary['official_score_mean']:.2f} "
                f"progress={summary['progress_mean']:.2f} speed={summary['speed_mean']:.2f} "
                f"state_loss={last_losses.get('state_loss', np.nan):.4f} best={best_score:.2f} "
                f"elite={elite.get('count', 0)}/{cfg.elite_buffer_size}"
            )
        elif cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0:
            _save_checkpoint(
                run_dir / f"lewm_step_{step:08d}.pt",
                model, optimizer, train_cfg, cfg, action_dim, step, best_score, rows,
            )

    best_path = run_dir / "lewm_best.pt"
    latest_path = run_dir / "lewm_latest.pt"
    if not best_path.exists() and latest_path.exists():
        shutil.copy2(latest_path, best_path)
        deploy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, deploy_dir / "lewm_best.pt")

    return {
        "metrics_path": str(metrics_path),
        "best_checkpoint": str(best_path),
        "deployed_checkpoint": str(deploy_dir / "lewm_best.pt"),
        "best_score": best_score,
        "replay_size": replay.size,
        "updates": update_count,
        "repo_root": str(find_repo_root()),
        "model": model,
        "rows": rows,
    }


# -----------------------------------------------------------------------------
# Final eval + video
# -----------------------------------------------------------------------------

def run_final_eval(cfg: HopperTaskConfig, model: JEPA) -> dict[str, Any]:
    """Run final eval pass with multiple episodes; render the best episode."""
    print(f"\n[final eval] {cfg.final_eval_episodes} episodes × "
          f"{cfg.final_eval_episode_steps} steps")
    eval_results = _evaluate_model(
        model, cfg, cfg.seed + 999_999,
        show_progress=True,
        n_episodes=cfg.final_eval_episodes,
        episode_steps=cfg.final_eval_episode_steps,
    )
    summary = summarize_episodes(eval_results)
    print(f"summary={summary}")
    for i, r in enumerate(eval_results):
        print(f"  ep {i}: official={r['official_score']:.2f} task={r['task_score']:.1f} "
              f"prog={r['progress_m']:.2f}m grade={r['grade_label']}")

    best_idx = int(np.argmax([r["official_score"] for r in eval_results]))
    print(f"rendering best episode (idx={best_idx})...")
    policy = _build_eval_policy(model, cfg, cfg.seed + 999_999)
    env = make_dm_env("hopper", "hop", seed=cfg.seed + 999_999)
    rendered = _eval_single_episode(
        policy=policy, env=env,
        max_steps=cfg.final_eval_episode_steps, render=True,
        width=cfg.render_width, height=cfg.render_height, camera_id=cfg.camera_id,
        progress_desc=f"render best (idx {best_idx})",
    )
    print(
        f"rendered_metrics: official_score={rendered['official_score']:.2f} "
        f"task_score={rendered['task_score']:.1f} "
        f"progress_m={rendered['progress_m']:.2f}m grade={rendered['grade_label']}"
    )
    save_video(rendered["frames"], cfg.video_path, fps=cfg.video_fps)
    return {
        "summary": summary,
        "eval_results": [{k: v for k, v in r.items() if k != "frames"} for r in eval_results],
        "rendered_summary": {k: v for k, v in rendered.items() if k != "frames"},
        "video_path": str(cfg.video_path),
    }


# -----------------------------------------------------------------------------
# Strategy bundle (for downstream tools that want to load and run)
# -----------------------------------------------------------------------------

def build_strategy(config: dict[str, Any]) -> StrategyBundle:
    repo_root = find_repo_root()
    cfg_dict = config["strategies"]["lewm"]
    target = Path(cfg_dict["checkpoint_path"]).expanduser()
    if not target.is_absolute():
        target = repo_root / target
    try:
        model, payload = load_checkpoint(target, cfg_dict.get("device", "cpu"))
    except (FileNotFoundError, ValueError) as exc:
        return StrategyBundle("lewm", None, None, None, {"status": "not_ready", "message": str(exc)})
    task_payload = payload.get("task_config") or {}
    base = HopperTaskConfig()
    valid_keys = set(asdict(base).keys())
    overrides = {k: v for k, v in task_payload.items() if k in valid_keys}
    overrides["device"] = cfg_dict.get("device", "cpu")
    cfg = HopperTaskConfig(**{**asdict(base), **overrides})
    policy = _build_eval_policy(model, cfg, config.get("seed", 42))
    return StrategyBundle(
        "lewm", policy, None, payload.get("history"),
        {
            "status": "ready", "checkpoint_file": payload["checkpoint_file"],
            "model_type": payload.get("model_type"), "global_step": payload.get("global_step"),
        },
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + eval + render hopper.hop via LeWM-MPC.")
    p.add_argument("--dataset-name", default="hopper_hop")
    p.add_argument("--run-name", default="hopper/hop")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    # Training schedule
    p.add_argument("--train-steps", type=int, default=50_000)
    p.add_argument("--random-steps", type=int, default=100)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--replay-capacity", type=int, default=1_000_000)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--fall-reset-height", type=float, default=0.30)
    p.add_argument("--fall-reset-grace", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--eval-episode-steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--score-weight-alpha", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--normalizer-update-every", type=int, default=1000)
    p.add_argument("--exploration-noise", type=float, default=0.10)
    # Final eval / video
    p.add_argument("--final-eval-episodes", type=int, default=5)
    p.add_argument("--final-eval-episode-steps", type=int, default=1000)
    p.add_argument("--video-path", type=str, default="hopper_best_episode.mp4")
    p.add_argument("--video-fps", type=int, default=40)
    p.add_argument("--render-width", type=int, default=480)
    p.add_argument("--render-height", type=int, default=368)
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--no-final-eval", action="store_true",
                   help="Skip the post-training evaluation+video pass.")
    # Model architecture
    p.add_argument("--state-dim", type=int, default=15)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--state-hidden-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-dropout", type=float, default=0.1)
    p.add_argument("--predictor-emb-dropout", type=float, default=0.0)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--state-loss-weight", type=float, default=1.0)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)
    # Cost
    p.add_argument("--progress-weight", type=float, default=7.0)
    p.add_argument("--speed-weight", type=float, default=1.0)
    p.add_argument("--height-weight", type=float, default=2.5)
    p.add_argument("--upright-weight", type=float, default=2.5)
    p.add_argument("--alive-weight", type=float, default=2.0)
    p.add_argument("--action-energy-weight", type=float, default=0.02)
    p.add_argument("--action-jerk-weight", type=float, default=0.05)
    p.add_argument("--target-speed", type=float, default=2.0)
    p.add_argument("--stand-height", type=float, default=0.6)
    p.add_argument("--alive-height", type=float, default=0.35)
    # CEM
    p.add_argument("--plan-horizon", type=int, default=32)
    p.add_argument("--cem-num-samples", type=int, default=256)
    p.add_argument("--cem-n-steps", type=int, default=4)
    p.add_argument("--cem-topk", type=int, default=32)
    p.add_argument("--cem-var-scale", type=float, default=1.0)
    p.add_argument("--cem-init-std", type=float, default=0.5)
    p.add_argument("--cem-min-std", type=float, default=0.05)
    p.add_argument("--cem-noise-beta", type=float, default=2.0)
    p.add_argument("--plan-every", type=int, default=4)
    p.add_argument("--elite-buffer-size", type=int, default=64)
    p.add_argument("--elite-seed-fraction", type=float, default=0.2)
    p.add_argument("--elite-score-temperature", type=float, default=100.0)
    p.add_argument("--primitive-mixture-initial", type=float, default=0.5)
    p.add_argument("--primitive-mixture-final", type=float, default=0.05)
    p.add_argument("--primitive-mixture-anneal-steps", type=int, default=5000)
    p.add_argument("--forced-exploration-initial", type=float, default=0.4)
    p.add_argument("--forced-exploration-baseline", type=float, default=0.05)
    p.add_argument("--forced-primitive-quota", type=int, default=500)
    p.add_argument("--primitive-elite-min-fraction", type=float, default=0.2)
    p.add_argument("--exploration-std-scale", type=float, default=1.5)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    skip_eval = bool(args.no_final_eval)
    cli_kwargs = {k: v for k, v in vars(args).items() if k != "no_final_eval"}
    cfg = HopperTaskConfig(**cli_kwargs)

    print(f"[run] {cfg.run_name} on {cfg.device} for {cfg.train_steps} steps")
    result = run_training(cfg)
    print(f"\n[train done] best_score={result['best_score']:.2f} "
          f"replay={result['replay_size']} updates={result['updates']}")
    print(f"  metrics: {result['metrics_path']}")
    print(f"  best ckpt: {result['best_checkpoint']}")

    if not skip_eval:
        eval_payload = run_final_eval(cfg, result["model"])
        print(f"\n[final eval done] video at {eval_payload['video_path']}")


if __name__ == "__main__":
    main()
