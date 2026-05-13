"""Generic latent state/delta manifold construction.

Task adapters decide which frames belong to successful behavior. This module
only consumes masks over pixels and builds latent geometry for planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


EncodeFramesFn = Callable[[np.ndarray], torch.Tensor]


@dataclass(frozen=True)
class ManifoldConfig:
    horizon: int
    max_state_points: int = 4096
    max_delta_points: int = 4096
    max_segments: int = 4096
    cost_scale_min: float = 1e-4
    nn_stats_points: int = 1024


@dataclass(frozen=True)
class SuccessSegment:
    name: str
    pixels: np.ndarray
    mask: np.ndarray
    role: str = "goal"
    metadata: dict | None = None


@dataclass(frozen=True)
class LatentManifold:
    state_latents: torch.Tensor
    delta_latents: torch.Tensor
    state_segments: torch.Tensor
    delta_segments: torch.Tensor
    support_state_latents: torch.Tensor
    support_delta_latents: torch.Tensor
    support_state_segments: torch.Tensor
    support_delta_segments: torch.Tensor
    state_scale: float
    delta_scale: float
    support_state_scale: float
    support_delta_scale: float
    diagnostics: dict


def subsample_rows(values: torch.Tensor, max_rows: int) -> torch.Tensor:
    if values.size(0) <= max_rows:
        return values
    idx = torch.linspace(
        0,
        values.size(0) - 1,
        max_rows,
        device=values.device,
        dtype=torch.float32,
    ).round().long()
    return values.index_select(0, idx)


def tensor_stats(values: torch.Tensor) -> dict:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {}
    return {
        "min": float(values.min().item()),
        "median": float(values.median().item()),
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


@torch.no_grad()
def nearest_neighbor_mse_stats(values: torch.Tensor, max_points: int = 1024) -> dict:
    values = subsample_rows(values.float(), max_points)
    if values.size(0) < 2:
        return {}
    dist = torch.cdist(values, values).pow(2).div(values.size(-1))
    dist.fill_diagonal_(float("inf"))
    return tensor_stats(dist.min(dim=1).values)


def contiguous_motion_segments(
    latents: torch.Tensor,
    mask: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_segments = []
    delta_segments = []
    max_start = latents.size(0) - horizon - 1
    for start in range(max_start + 1):
        if bool(mask[start : start + horizon + 1].all()):
            state_segments.append(latents[start + 1 : start + horizon + 1])
            delta_segments.append(latents[start + 1 : start + horizon + 1] - latents[start : start + horizon])
    if not state_segments:
        empty = latents.new_zeros((0, horizon, latents.size(-1)))
        return empty, empty
    return torch.stack(state_segments, dim=0), torch.stack(delta_segments, dim=0)


def _empty_like(parts: list[torch.Tensor], horizon: int, device: torch.device) -> torch.Tensor:
    if parts:
        dim = parts[0].size(-1)
    else:
        dim = 1
    return torch.zeros((0, horizon, dim), device=device)


def _empty_points_like(parts: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if parts:
        dim = parts[0].size(-1)
    else:
        dim = 1
    return torch.zeros((0, dim), device=device)


def _robust_scale(values: torch.Tensor, min_value: float, max_points: int) -> tuple[float, dict]:
    stats = nearest_neighbor_mse_stats(values, max_points)
    if not stats:
        return min_value, stats
    return max(float(stats.get("p95", stats.get("median", min_value))), min_value), stats


@torch.no_grad()
def build_latent_manifold(
    segments: list[SuccessSegment],
    encode_frames: EncodeFramesFn,
    config: ManifoldConfig,
    device: torch.device,
) -> LatentManifold:
    goal_state_parts: list[torch.Tensor] = []
    goal_delta_parts: list[torch.Tensor] = []
    goal_state_segment_parts: list[torch.Tensor] = []
    goal_delta_segment_parts: list[torch.Tensor] = []
    support_state_parts: list[torch.Tensor] = []
    support_delta_parts: list[torch.Tensor] = []
    support_state_segment_parts: list[torch.Tensor] = []
    support_delta_segment_parts: list[torch.Tensor] = []
    sources: list[dict] = []

    for source in segments:
        if source.role not in {"goal", "support"}:
            raise ValueError(f"unknown success segment role: {source.role}")
        latents = encode_frames(source.pixels).float()
        mask_np = np.asarray(source.mask, dtype=bool)
        if len(mask_np) != len(source.pixels):
            raise ValueError(f"mask length mismatch for {source.name}")
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        pair_mask = mask[:-1] & mask[1:]

        states = latents[mask]
        deltas = latents[1:][pair_mask] - latents[:-1][pair_mask]
        state_segments, delta_segments = contiguous_motion_segments(latents, mask, config.horizon)

        if source.role == "goal":
            state_parts = goal_state_parts
            delta_parts = goal_delta_parts
            state_segment_parts = goal_state_segment_parts
            delta_segment_parts = goal_delta_segment_parts
        else:
            state_parts = support_state_parts
            delta_parts = support_delta_parts
            state_segment_parts = support_state_segment_parts
            delta_segment_parts = support_delta_segment_parts

        if states.numel() > 0:
            state_parts.append(states)
        if deltas.numel() > 0:
            delta_parts.append(deltas)
        if state_segments.numel() > 0:
            state_segment_parts.append(state_segments)
        if delta_segments.numel() > 0:
            delta_segment_parts.append(delta_segments)

        sources.append(
            {
                "name": source.name,
                "role": source.role,
                "frames": int(len(source.pixels)),
                "state_points": int(states.size(0)),
                "delta_points": int(deltas.size(0)),
                "state_segments": int(state_segments.size(0)),
                "delta_segments": int(delta_segments.size(0)),
                "metadata": source.metadata or {},
            }
        )

    if not goal_state_parts or not goal_delta_parts or not goal_state_segment_parts:
        raise RuntimeError("Could not build a non-empty goal latent manifold.")

    state_latents = subsample_rows(torch.cat(goal_state_parts, dim=0), config.max_state_points)
    delta_latents = subsample_rows(torch.cat(goal_delta_parts, dim=0), config.max_delta_points)
    state_segments = subsample_rows(torch.cat(goal_state_segment_parts, dim=0), config.max_segments)
    delta_segments = subsample_rows(torch.cat(goal_delta_segment_parts, dim=0), config.max_segments)

    support_state_latents = (
        subsample_rows(torch.cat(support_state_parts, dim=0), config.max_state_points)
        if support_state_parts else _empty_points_like(goal_state_parts, device)
    )
    support_delta_latents = (
        subsample_rows(torch.cat(support_delta_parts, dim=0), config.max_delta_points)
        if support_delta_parts else _empty_points_like(goal_delta_parts, device)
    )
    support_state_segments = (
        subsample_rows(torch.cat(support_state_segment_parts, dim=0), config.max_segments)
        if support_state_segment_parts else _empty_like(goal_state_parts, config.horizon, device)
    )
    support_delta_segments = (
        subsample_rows(torch.cat(support_delta_segment_parts, dim=0), config.max_segments)
        if support_delta_segment_parts else _empty_like(goal_delta_parts, config.horizon, device)
    )

    state_scale, state_nn = _robust_scale(state_latents, config.cost_scale_min, config.nn_stats_points)
    delta_scale, delta_nn = _robust_scale(delta_latents, config.cost_scale_min, config.nn_stats_points)
    support_state_scale, support_state_nn = (
        _robust_scale(support_state_latents, config.cost_scale_min, config.nn_stats_points)
        if support_state_latents.size(0) >= 2 else (state_scale, {})
    )
    support_delta_scale, support_delta_nn = (
        _robust_scale(support_delta_latents, config.cost_scale_min, config.nn_stats_points)
        if support_delta_latents.size(0) >= 2 else (delta_scale, {})
    )

    delta_norm = delta_latents.pow(2).mean(dim=-1)
    support_delta_norm = (
        support_delta_latents.pow(2).mean(dim=-1)
        if support_delta_latents.size(0) else torch.zeros(0, device=device)
    )
    diagnostics = {
        "sources": sources,
        "state_points": int(state_latents.size(0)),
        "delta_points": int(delta_latents.size(0)),
        "state_segments": int(state_segments.size(0)),
        "delta_segments": int(delta_segments.size(0)),
        "support_state_points": int(support_state_latents.size(0)),
        "support_delta_points": int(support_delta_latents.size(0)),
        "support_state_segments": int(support_state_segments.size(0)),
        "support_delta_segments": int(support_delta_segments.size(0)),
        "state_scale": state_scale,
        "delta_scale": delta_scale,
        "support_state_scale": support_state_scale,
        "support_delta_scale": support_delta_scale,
        "state_nearest_mse": state_nn,
        "delta_nearest_mse": delta_nn,
        "support_state_nearest_mse": support_state_nn,
        "support_delta_nearest_mse": support_delta_nn,
        "delta_norm_mse": tensor_stats(delta_norm),
        "support_delta_norm_mse": tensor_stats(support_delta_norm),
    }
    return LatentManifold(
        state_latents=state_latents,
        delta_latents=delta_latents,
        state_segments=state_segments,
        delta_segments=delta_segments,
        support_state_latents=support_state_latents,
        support_delta_latents=support_delta_latents,
        support_state_segments=support_state_segments,
        support_delta_segments=support_delta_segments,
        state_scale=state_scale,
        delta_scale=delta_scale,
        support_state_scale=support_state_scale,
        support_delta_scale=support_delta_scale,
        diagnostics=diagnostics,
    )

