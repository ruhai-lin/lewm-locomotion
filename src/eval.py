"""Task-agnostic CEM evaluation for trained LeWM models.

The eval loop:

1. Builds the goal manifold from the dataset (state + delta + repulsion).
2. Resets the env and bootstraps ``bootstrap_steps`` env steps using the
   first actions of the goal trajectory, so the agent enters the eval at
   a state that overlaps the goal manifold.
3. Runs ``CEM`` over LeWM rollouts for the remaining steps. At each step,
   ``phase`` walks along the goal trajectory (one frame per env step,
   wrapped); the planner's ``prior_actions`` come from the goal at the
   current phase; the planner's ``target_latents`` come from the goal
   latents at the next H frames. Whatever the planner returns is executed
   in the real env.
4. Writes per-step ``eval_trace.jsonl`` (with per-step task metrics from
   ``record_fn`` passed through verbatim) and a summary ``metrics.json``.

Nothing in this file mentions walker / cheetah / hopper. Tasks plug in via
callables: ``env_fn``, ``render_video_fn``, ``render_dataset_fn``,
``action_bounds_fn``, ``record_fn``, ``preprocess_pixels_fn``,
and ``encode_frames_fn`` for manifold building.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from src import lewm as lewm_module
from src.manifold import LatentManifold, build_goal_manifold
from src.planner import LatentCEMPlanner, PlannerConfig, PlannerWeights


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    eval_steps: int
    eval_seed: int
    bootstrap_steps: int
    history_size: int
    plan_horizon: int
    action_block: int
    plan_blocks: int
    samples: int
    topk: int
    iters: int
    init_std: float
    min_std: float
    max_std: float
    momentum: float
    prior_tie_abs: float = 0.0
    prior_in_mean: bool = True
    prior_in_samples: bool = True
    cost_scale_min: float = 1e-4
    traj_discount: float = 0.92
    video_fps: int = 40


# ---------------------------------------------------------------------------
# Phase tracking against the (looped) goal trajectory.
# ---------------------------------------------------------------------------


def _frame_index(phase: int, goal_len: int) -> int:
    if goal_len <= 0:
        return 0
    return int(phase % goal_len)


def _action_index(phase: int, num_actions: int) -> int:
    if num_actions <= 0:
        return 0
    return int(phase % num_actions)


def _goal_window(
    goal_latents: torch.Tensor,
    goal_actions: torch.Tensor,
    phase: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_ids = [_frame_index(phase + i + 1, goal_latents.size(0)) for i in range(horizon)]
    action_ids = [_action_index(phase + i, goal_actions.size(0)) for i in range(horizon)]
    frame_idx = torch.tensor(frame_ids, device=goal_latents.device, dtype=torch.long)
    action_idx = torch.tensor(action_ids, device=goal_actions.device, dtype=torch.long)
    return goal_latents.index_select(0, frame_idx), goal_actions.index_select(0, action_idx)


@torch.no_grad()
def _nearest_goal_phase(
    current_latent: torch.Tensor, goal_latents: torch.Tensor
) -> tuple[int, float]:
    candidates = goal_latents.float()
    dist = (candidates - current_latent.view(1, -1).float()).pow(2).mean(dim=-1)
    idx = int(torch.argmin(dist).item())
    return idx, float(dist[idx].item())


@torch.no_grad()
def _nearest_point_mse(point: torch.Tensor, points: torch.Tensor) -> float:
    if points.size(0) == 0:
        return float("nan")
    point = point.view(1, -1).float()
    dist = (points.float() - point).pow(2).mean(dim=-1)
    return float(dist.min().item())


# ---------------------------------------------------------------------------
# Main entrypoint.
# ---------------------------------------------------------------------------


PreprocessPixelsFn = Callable[[torch.Tensor], torch.Tensor]
EncodeFramesFn = Callable[[object, np.ndarray, torch.device], torch.Tensor]


@torch.no_grad()
def evaluate_lewm_cem(
    *,
    ckpt_path: Path,
    dataset_dir: Path,
    out_dir: Path,
    model_cfg: lewm_module.LeWMModelConfig,
    eval_cfg: EvalConfig,
    weights: PlannerWeights,
    env_fn: Callable[[int], object],
    action_bounds_fn: Callable[[object], tuple[np.ndarray, np.ndarray]],
    render_dataset_fn: Callable[[object], np.ndarray],
    render_video_fn: Callable[[object], np.ndarray],
    record_fn: Callable[[object], dict[str, float]],
    preprocess_pixels_fn: PreprocessPixelsFn,
    manifold_max_points: int = 4096,
    manifold_max_segments: int = 4096,
    extra_metrics: dict | None = None,
    domain: str = "",
    task: str = "",
) -> dict:
    """Run a full CEM eval and return the summary metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "eval_trace.jsonl"
    diagnostics_path = out_dir / "diagnostics.json"
    metrics_path = out_dir / "metrics.json"

    model, device = lewm_module.load_lewm(ckpt_path, model_cfg)

    with np.load(dataset_dir / "goal_trajectory.npz") as goal_npz:
        goal_ep = {k: goal_npz[k] for k in goal_npz.files}
    goal_pixels = goal_ep["pixels"]
    goal_actions_np = goal_ep["actions"].astype(np.float32)

    def _encode(pixels: np.ndarray) -> torch.Tensor:
        return lewm_module.encode_frames(
            model, pixels, device, image_size=model_cfg.image_size, chunk=128
        )

    goal_manifold = build_goal_manifold(
        goal_pixels=goal_pixels,
        encode_frames=_encode,
        plan_horizon=eval_cfg.plan_horizon,
        max_state_points=manifold_max_points,
        max_delta_points=manifold_max_points,
        max_segments=manifold_max_segments,
        cost_scale_min=eval_cfg.cost_scale_min,
        device=device,
    )

    goal_latents = _encode(goal_pixels)
    goal_actions = torch.from_numpy(goal_actions_np).to(device)
    imageio.mimsave(out_dir / "goal_trajectory.mp4", list(goal_pixels), fps=eval_cfg.video_fps)

    diagnostics_payload = {
        "goal_manifold": goal_manifold.diagnostics,
        "planner_weights": weights.as_dict(),
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2))

    if trace_path.exists():
        trace_path.unlink()

    def _append_trace(record: dict) -> None:
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    env = env_fn(eval_cfg.eval_seed)
    env.reset()
    low_np, high_np = action_bounds_fn(env)
    action_dim = int(low_np.size)
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    zero = np.zeros(action_dim, dtype=np.float32)

    initial_frame = render_dataset_fn(env)
    history_frames: list[np.ndarray] = [initial_frame.copy()]
    history_actions: list[np.ndarray] = [zero.copy()]
    last_obs_latent = _encode(initial_frame[None])[0]
    previous_action = torch.zeros(action_dim, device=device)
    horizon_weights = torch.tensor(
        [eval_cfg.traj_discount**i for i in range(eval_cfg.plan_horizon)],
        device=device,
        dtype=torch.float32,
    )
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    rng = torch.Generator(device=gen_device).manual_seed(eval_cfg.eval_seed + 30_000)

    planner = LatentCEMPlanner(
        model,
        PlannerConfig(
            history_size=eval_cfg.history_size,
            action_block=eval_cfg.action_block,
            plan_blocks=eval_cfg.plan_blocks,
            samples=eval_cfg.samples,
            topk=eval_cfg.topk,
            iters=eval_cfg.iters,
            init_std=eval_cfg.init_std,
            max_std=eval_cfg.max_std,
            min_std=eval_cfg.min_std,
            momentum=eval_cfg.momentum,
            prior_tie_abs=eval_cfg.prior_tie_abs,
            cost_scale_min=eval_cfg.cost_scale_min,
            prior_in_mean=eval_cfg.prior_in_mean,
            prior_in_samples=eval_cfg.prior_in_samples,
        ),
        weights,
    )

    video_frames: list[np.ndarray] = []
    rewards: list[float] = []
    actions_out: list[np.ndarray] = []
    diag_log: dict[str, list[float]] = {}
    tracking_log: dict[str, list[float]] = {}
    cost_infos: list[dict[str, float]] = []

    phase = 0
    env_steps = 0
    bar = tqdm(total=eval_cfg.eval_steps, desc="eval LeWM-CEM", dynamic_ncols=True)

    def _tracking_record(
        *,
        next_frame: np.ndarray,
        prev_latent: torch.Tensor,
        phase_value: int,
        selected_action: np.ndarray,
        prior_action: np.ndarray,
    ) -> tuple[dict[str, float], torch.Tensor]:
        next_latent = _encode(next_frame[None])[0]
        obs_delta = next_latent - prev_latent
        goal_idx = _frame_index(phase_value, goal_latents.size(0))
        goal_latent = goal_latents[goal_idx]
        action_diff = selected_action.astype(np.float32) - prior_action.astype(np.float32)
        state_mse = _nearest_point_mse(next_latent, goal_manifold.state_latents)
        delta_mse = _nearest_point_mse(obs_delta, goal_manifold.delta_latents)
        nearest_goal_mse = _nearest_point_mse(next_latent, goal_latents)
        phase_goal_mse = float((next_latent.float() - goal_latent.float()).pow(2).mean().item())
        record = {
            "obs_state_mse": state_mse,
            "obs_delta_mse": delta_mse,
            "obs_state_dist_norm": state_mse / max(goal_manifold.state_scale, 1e-8),
            "obs_delta_dist_norm": delta_mse / max(goal_manifold.delta_scale, 1e-8),
            "obs_goal_traj_mse": nearest_goal_mse,
            "obs_goal_traj_dist_norm": nearest_goal_mse / max(goal_manifold.state_scale, 1e-8),
            "obs_phase_goal_mse": phase_goal_mse,
            "obs_phase_goal_dist_norm": phase_goal_mse / max(goal_manifold.state_scale, 1e-8),
            "selected_prior_action_l2": float(np.linalg.norm(action_diff)),
            "selected_prior_action_mse": float(np.mean(action_diff * action_diff)),
            "selected_prior_action_mean_abs": float(np.mean(np.abs(action_diff))),
        }
        for key, value in record.items():
            tracking_log.setdefault(key, []).append(float(value))
        return record, next_latent

    bootstrap_steps = min(eval_cfg.bootstrap_steps, eval_cfg.eval_steps, len(goal_actions_np))
    for boot in range(bootstrap_steps):
        action = goal_actions_np[boot].astype(np.float32)
        video_frames.append(render_video_fn(env))
        ts = env.step(action)
        diag = record_fn(env)
        rewards.append(float(ts.reward or 0.0))
        actions_out.append(action.copy())
        for key, value in diag.items():
            diag_log.setdefault(key, []).append(float(value))
        next_frame = render_dataset_fn(env)
        tracking, last_obs_latent = _tracking_record(
            next_frame=next_frame,
            prev_latent=last_obs_latent,
            phase_value=boot + 1,
            selected_action=action,
            prior_action=action,
        )
        history_frames.append(next_frame)
        history_actions.append(action.copy())
        previous_action = torch.from_numpy(action).to(device)
        env_steps += 1
        phase = boot + 1
        bar.update(1)
        _append_trace(
            {
                "step": env_steps,
                "mode": "bootstrap",
                "reward": rewards[-1],
                "return": float(np.sum(rewards)),
                "phase": boot,
                "action_abs": float(np.mean(np.abs(action))),
                **tracking,
                **{k: float(v) for k, v in diag.items()},
            }
        )
        if ts.last():
            env_steps = eval_cfg.eval_steps
            break

    while env_steps < eval_cfg.eval_steps:
        current_u8 = (
            torch.from_numpy(np.stack(history_frames[-1:], axis=0))
            .unsqueeze(0)
            .to(device)
        )
        current_latent = (
            model.encode({"pixels": preprocess_pixels_fn(current_u8)})["emb"]
            .squeeze(0)
            .squeeze(0)
        )
        # Advance phase by one per env step, wrapping around the (looped)
        # goal trajectory. We also peek at the nearest-goal phase for the
        # trace but no longer relock to it; under the E7 config the simple
        # monotonic phase walks is just as effective and removes a knob.
        global_phase, current_dist = _nearest_goal_phase(current_latent, goal_latents)
        target_latents, prior_actions = _goal_window(
            goal_latents, goal_actions, phase, eval_cfg.plan_horizon
        )

        hist_pix = torch.from_numpy(
            np.stack(history_frames[-eval_cfg.history_size:], axis=0)
        ).unsqueeze(0)
        hist_act = torch.from_numpy(
            np.stack(history_actions[-eval_cfg.history_size:], axis=0)
        ).unsqueeze(0)
        plan, plan_info = planner.plan(
            history_pixels=preprocess_pixels_fn(hist_pix.to(device)),
            history_actions=hist_act,
            current_latent=current_latent,
            target_latents=target_latents,
            manifold=goal_manifold,
            prior_actions=prior_actions,
            previous_action=previous_action,
            low=low,
            high=high,
            horizon_weights=horizon_weights,
            rng=rng,
        )

        action = plan[0].detach().cpu().numpy().astype(np.float32)
        for _ in range(eval_cfg.action_block):
            if env_steps >= eval_cfg.eval_steps:
                break
            step_phase = phase
            video_frames.append(render_video_fn(env))
            ts = env.step(action)
            diag = record_fn(env)
            rewards.append(float(ts.reward or 0.0))
            actions_out.append(action.copy())
            for key, value in diag.items():
                diag_log.setdefault(key, []).append(float(value))
            cost_infos.append(plan_info)
            next_frame = render_dataset_fn(env)
            prior_step_action = (
                goal_actions[_action_index(step_phase, goal_actions.size(0))]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            tracking, last_obs_latent = _tracking_record(
                next_frame=next_frame,
                prev_latent=last_obs_latent,
                phase_value=step_phase + 1,
                selected_action=action,
                prior_action=prior_step_action,
            )
            history_frames.append(next_frame)
            history_actions.append(action.copy())
            previous_action = torch.from_numpy(action).to(device)
            env_steps += 1
            bar.update(1)
            _append_trace(
                {
                    "step": env_steps,
                    "mode": "cem",
                    "reward": rewards[-1],
                    "return": float(np.sum(rewards)),
                    "distance": float(current_dist),
                    "phase": int(step_phase),
                    "global_phase": int(global_phase),
                    "action_abs": float(np.mean(np.abs(action))),
                    "previous_action_abs": float(previous_action.abs().mean().detach().cpu()),
                    **tracking,
                    **{k: float(v) for k, v in diag.items()},
                    **plan_info,
                }
            )
            method_share = (
                plan_info.get("state_cost_share", 0.0)
                + plan_info.get("delta_cost_share", 0.0)
            )
            postfix = {
                "ret": f"{sum(rewards):.1f}",
                "method": f"{method_share:.2f}",
                "phase": step_phase,
            }
            for k, v in diag.items():
                postfix[k[:8]] = f"{float(v):.2f}"
            bar.set_postfix(postfix)
            phase += 1
            if ts.last():
                env_steps = eval_cfg.eval_steps
                break
    bar.close()
    env.close()

    video_path = out_dir / "eval_cam.mp4"
    imageio.mimsave(video_path, video_frames, fps=eval_cfg.video_fps)

    def mean_cost_info(key: str) -> float:
        if not cost_infos:
            return float("nan")
        return float(np.mean([info.get(key, 0.0) for info in cost_infos]))

    mean_method_share = mean_cost_info("state_cost_share") + mean_cost_info("delta_cost_share")
    mean_near_prior_share = mean_cost_info("near_cost_share") + mean_cost_info("prior_cost_share")
    total_return = float(np.sum(rewards)) if rewards else 0.0
    metrics = {
        "domain": domain,
        "task": task,
        "policy": "pure_lewm_cem_latent_delta_manifold",
        "steps": len(rewards),
        "total_return": total_return,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_abs_action": float(np.mean(np.abs(actions_out))) if actions_out else float("nan"),
        "planner_weights": weights.as_dict(),
        "mean_state_cost": mean_cost_info("state_cost"),
        "mean_delta_cost": mean_cost_info("delta_cost"),
        "mean_near_cost": mean_cost_info("near_cost"),
        "mean_prior_cost": mean_cost_info("prior_cost"),
        "mean_smooth_cost": mean_cost_info("smooth_cost"),
        "mean_energy_cost": mean_cost_info("energy_cost"),
        "mean_total_cost": mean_cost_info("total_cost"),
        "mean_state_cost_share": mean_cost_info("state_cost_share"),
        "mean_delta_cost_share": mean_cost_info("delta_cost_share"),
        "mean_near_cost_share": mean_cost_info("near_cost_share"),
        "mean_prior_cost_share": mean_cost_info("prior_cost_share"),
        "prior_guard_used_rate": mean_cost_info("used_prior_guard"),
        "mean_prior_cost_improvement": mean_cost_info("prior_cost_improvement"),
        "mean_prior_guard_margin": mean_cost_info("prior_guard_margin"),
        "mean_method_cost_share": mean_method_share,
        "mean_near_prior_cost_share": mean_near_prior_share,
        "method_cost_dominant": bool(mean_method_share > mean_near_prior_share),
        "sanity_return_ok": bool(total_return >= 100.0),
        "goal_manifold": {
            k: goal_manifold.diagnostics[k]
            for k in (
                "state_points",
                "delta_points",
                "state_segments",
                "delta_segments",
                "state_scale",
                "delta_scale",
            )
        },
        "checkpoint": str(ckpt_path),
        "dataset_dir": str(dataset_dir),
        "eval_trace": str(trace_path),
        "diagnostics": str(diagnostics_path),
        "goal_video": str(out_dir / "goal_trajectory.mp4"),
        "eval_video": str(video_path),
    }
    # Task-specific aggregates: mean / min / max of every diag field recorded
    # by ``record_fn``. We use ``mean_<key>`` naming so existing dashboards
    # that look for ``mean_torso_height`` keep working on walker.
    for key, values in diag_log.items():
        if values:
            metrics[f"mean_{key}"] = float(np.nanmean(values))
            metrics[f"min_{key}"] = float(np.nanmin(values))
            metrics[f"max_{key}"] = float(np.nanmax(values))
    for key, values in tracking_log.items():
        if values:
            metrics[f"mean_{key}"] = float(np.nanmean(values))
            metrics[f"min_{key}"] = float(np.nanmin(values))
            metrics[f"max_{key}"] = float(np.nanmax(values))

    if extra_metrics is not None:
        metrics.update(extra_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"[eval] return={total_return:.1f} "
        f"method_share={mean_method_share:.2f} "
        f"guard_rate={metrics['prior_guard_used_rate']:.3f} "
        f"-> {video_path}"
    )
    return metrics
