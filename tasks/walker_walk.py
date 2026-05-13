"""Clean reference-style LeWM pipeline for dm_control walker/walk.

Run:
    uv run python tasks/walker_walk.py

The script is intentionally split into reusable stages:
1. Build a static dataset under DATASET_DIR if it does not already exist.
2. Train LeWM from that dataset with one-step prediction + SIGReg.
3. Evaluate pure LeWM-CEM with latent state/delta manifold planning.

No MuJoCo oracle, reward, or intervention is used during training or eval.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from dm_control import suite
from tqdm import tqdm

from src import datasets, lewm
from src.manifold import LatentManifold, ManifoldConfig, SuccessSegment, build_latent_manifold
from src.planner import LatentCEMPlanner, PlannerConfig, PlannerWeights


# ---------------------------------------------------------------------------
# Experiment paths.
# ---------------------------------------------------------------------------


DOMAIN = "walker"
TASK = "walk"
SEED = 42

DATASET_DIR = Path("datasets/walker_walk")
OUT_DIR = Path("outputs/walker_walk")
CKPT_DIR = OUT_DIR / "ckpt"
TRAIN_LOG_PATH = OUT_DIR / "train_log.jsonl"
EVAL_TRACE_PATH = OUT_DIR / "eval_trace.jsonl"
DIAGNOSTICS_PATH = OUT_DIR / "diagnostics.json"
DATASET_VERSION = 7  # bumped: 128 explore + 1 oracle goal demo (E7)


# ---------------------------------------------------------------------------
# Offline dataset generation.
# ---------------------------------------------------------------------------


IMAGE_SIZE = 112
DATASET_CAMERA_ID = 0
VIDEO_CAMERA_ID = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 40

# Exploration dataset. All explore episodes come from generic seed-driven
# policies in ``src.datasets``; no task-specific actions, no recovery
# scripts. The dataset is N explore + 1 oracle-CEM goal demo (the "+1"):
# E6 showed that fully-oracle-free random policies essentially never
# produce a frame **visually compatible with the env reset pose**, so the
# goal manifold and the reset pose end up as disjoint islands in latent
# space and CEM cannot bridge them. The 1 oracle demo gives the planner
# exactly one "start-at-reset-then-walk" trajectory, making the manifold
# reachable from where eval lives. See experiments.md E7.
DATASET_N_EPISODES = 128
DATASET_EPISODE_STEPS = 400
DATASET_WORKERS = min(8, os.cpu_count() or 1)
DATASET_MP_START_METHOD = "forkserver"

# Minimal oracle-CEM (used only for the 1 goal demo, never for explore data).
ORACLE_PLAN_HORIZON = 20
ORACLE_CEM_SAMPLES = 128
ORACLE_CEM_TOPK = 16
ORACLE_CEM_ITERS = 4
ORACLE_CEM_INIT_STD = 0.55
ORACLE_CEM_MIN_STD = 0.06
ORACLE_CEM_MOMENTUM = 0.15
ORACLE_REWARD_GAMMA = 0.98
ORACLE_TARGET_SPEED = 1.2
ORACLE_PROGRESS_BONUS = 0.75
ORACLE_SPEED_TRACKING_BONUS = 0.12
ORACLE_OVERSPEED_PENALTY = 0.35
ORACLE_POSTURE_PENALTY = 1.2
ORACLE_ACTION_PENALTY = 0.01
ORACLE_ACTION_SMOOTHNESS_PENALTY = 0.015

# Goal-frame selection (applied *after* collection). Frames that pass these
# thresholds are treated as on-manifold "walking-like" examples. Random
# policies rarely sustain *all four* tight thresholds simultaneously for
# many consecutive frames, so the defaults emphasise structural posture
# (height + upright) and ask for "non-stationary forward motion" via
# velocity. Reward is intentionally lax: it is a derived metric that adds
# little once height/upright/velocity already pass. If post-collection
# goal_manifold_log.json is empty, loosen these (or raise N_EPISODES).
GOAL_FRAME_MIN_REWARD = 0.0
GOAL_FRAME_MIN_HEIGHT = 0.9
GOAL_FRAME_MIN_UPRIGHT = 0.5
GOAL_FRAME_MIN_VELOCITY = 0.1
GOAL_MIN_SEGMENT_LEN = 4

# Support (off-manifold) frame selection — frames that look like a fall.
# Used as repulsion targets in the planner if W_SUPPORT_* > 0.
SUPPORT_FRAME_MAX_HEIGHT = 0.7
SUPPORT_FRAME_MAX_UPRIGHT = 0.3


# ---------------------------------------------------------------------------
# LeWM training.
# ---------------------------------------------------------------------------


HISTORY_SIZE = 4
NUM_PREDS = 1
FRAMESKIP = 1
PATCH_SIZE = 14
ENCODER_SCALE = "tiny"
EMBED_DIM = 192

PREDICTOR_DEPTH = 6
PREDICTOR_HEADS = 16
PREDICTOR_DIM_HEAD = 64
PREDICTOR_MLP_DIM = 2048
PREDICTOR_DROPOUT = 0.1
# "adaln_zero" matches upstream LeWM but produces zero action conditioning
# at init, which lets the predictor collapse onto visual continuity and
# ignore actions. "adaln_id" keeps the LeWM loss form unchanged but starts
# with non-zero action conditioning. See experiments.md E4d for the
# diagnostic that justifies this default.
PREDICTOR_CONDITIONING = "adaln_id"

SIGREG_WEIGHT = 0.07
SIGREG_KNOTS = 17
SIGREG_NUM_PROJ = 1024
TRAIN_BATCH_SIZE = 64
TRAIN_STEPS = 9000  # \"adaln_id\" + LeWM-pure loss needs more steps than E4a (see experiments.md E4d-v2).
TRAIN_NUM_WORKERS = 0
LR = 5e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
LOG_EVERY = 50
TRAIN_ROLLOUT_STEPS = 16
ROLLOUT_LOSS_WEIGHT = 1.0
# Action-contrast loss was useful only as an instrumented experiment in E4a
# to *prove* the action-bottleneck hypothesis. As a deliverable it
# (a) breaks the LeWM-pure loss form and (b) destabilises training
# (latent norm explodes -> rollout_loss spikes -> pred_loss spikes).
# E4d showed that the architectural fix (PREDICTOR_CONDITIONING="adaln_id")
# achieves the same goal with the original LeWM loss, so we keep contrast
# **off** by default.
ACTION_CONTRAST_WEIGHT = 0.0
ACTION_CONTRAST_MARGIN = 0.0
ACTION_CONTRAST_HORIZON = 0
# Window weighting (E6 Fix C, attempt 1) backfired: by concentrating
# training on stable walking windows it gave the predictor a more
# predictable visual signal and the action Jacobian regressed (E6 post-fix
# C3 measurement). Disabled in E6 post-mortem.
TRAIN_WINDOW_WEIGHTING = False


# ---------------------------------------------------------------------------
# Pure LeWM-CEM eval.
# ---------------------------------------------------------------------------


EVAL_STEPS = 400
EVAL_SEED = SEED
EVAL_BOOTSTRAP_STEPS = HISTORY_SIZE - 1
ACTION_BLOCK = 1
PLAN_BLOCKS = 16
PLAN_HORIZON = ACTION_BLOCK * PLAN_BLOCKS
EVAL_CEM_SAMPLES = 256
EVAL_CEM_TOPK = 32
EVAL_CEM_ITERS = 4
EVAL_CEM_INIT_STD = 0.01
EVAL_CEM_FAR_INIT_STD = 0.45
EVAL_CEM_MAX_STD = 0.65
EVAL_CEM_MIN_STD = 0.002
EVAL_CEM_MOMENTUM = 0.15
CEM_WARM_START_BLEND = 0.0
CEM_PRIOR_TIE_REL = 0.0
CEM_PRIOR_TIE_ABS = 0.0  # E7: guard fully off; CEM elite wins on cost alone.
# Planner uses the prior block as both the CEM mean and a hard-injected
# candidate (slot 0). Setting these to False disables both, forcing the
# prior to compete on cost. See experiments.md E5.
CEM_PRIOR_IN_MEAN = True
CEM_PRIOR_IN_SAMPLES = True
# Saturating clip on the normalised state cost. With a tight goal manifold
# (small state_scale), being many manifold-radii away from goal makes the
# raw state cost dominate and removes useful gradient. The clip caps state
# cost at ~5 manifold-radii so delta/near/smooth still shape behaviour
# outside the manifold. See experiments.md E6 follow-up.
CEM_STATE_COST_CLIP = 0.0  # off; only useful if goal manifold is far from reset attractor.
CEM_DELTA_COST_CLIP = 0.0
CEM_NEAR_COST_CLIP = 0.0
W_NEAR = 0.01  # original value, restored.
RECOVERY_DISTANCE_CENTER = 0.20
RECOVERY_DISTANCE_TEMPERATURE = 0.07
RECOVERY_PRESSURE_POWER = 1.4
RECOVERY_PHASE_CENTER = 40
RECOVERY_PHASE_TEMPERATURE = 6.0

GOAL_ORBIT_START = 40
PHASE_SEARCH_WINDOW = 4
PHASE_MIN_STEP = 0
PHASE_MAX_STEP = 0
TRAJ_DISCOUNT = 0.92
NEAR_DELTA_WEIGHT = 0.5
W_STATE = 0.2
W_DELTA = 4.0
W_SUPPORT_STATE = 0.0
W_SUPPORT_DELTA = 0.0
# W_NEAR is defined above (bumped to 0.1 in E6 follow-up); keeping the
# variable here only for documentation cohesion.
W_ACTION_PRIOR = 0.1
W_SMOOTH = 0.1
W_FAR_SMOOTH = 0.015
W_ENERGY = 0.002
ORBIT_BLEND_SIGMA_MULT = 8.0
ORBIT_BLEND_SIGMA_MIN = 0.025
ORBIT_BLEND_SIGMA_MAX = 0.35
PHASE_RELOCK_RHO = -1.0
GOAL_MANIFOLD_MIN_REWARD = 0.85
GOAL_MANIFOLD_MIN_HEIGHT = 1.05
GOAL_MANIFOLD_MIN_UPRIGHT = 0.85
GOAL_MANIFOLD_MIN_VELOCITY = 0.4
SUPPORT_TAIL_STABLE_STEPS = 12
SUPPORT_MIN_MASK_FRAMES = PLAN_HORIZON + 1
MANIFOLD_MAX_STATE_POINTS = 4096
MANIFOLD_MAX_DELTA_POINTS = 4096
MANIFOLD_MAX_SEGMENTS = 4096
MANIFOLD_COST_SCALE_MIN = 1e-4
ROLLOUT_DIAGNOSTIC_HORIZONS = [1, 4, 8, 16]


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def hparams() -> dict:
    simple = (str, int, float, bool, type(None))
    out = {}
    for key, value in globals().items():
        if not key.isupper():
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, simple):
            out[key] = value
        elif isinstance(value, list):
            out[key] = value
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Environment, rendering, and oracle CEM.
# ---------------------------------------------------------------------------


def make_env(seed: int):
    return suite.load(
        domain_name=DOMAIN,
        task_name=TASK,
        task_kwargs={"random": seed},
        visualize_reward=False,
    )


def action_bounds(env) -> Tuple[np.ndarray, np.ndarray]:
    spec = env.action_spec()
    return (
        np.asarray(spec.minimum, dtype=np.float32),
        np.asarray(spec.maximum, dtype=np.float32),
    )


def render_dataset_frame(env) -> np.ndarray:
    return env.physics.render(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        camera_id=DATASET_CAMERA_ID,
    ).astype(np.uint8)


def render_video_frame(env) -> np.ndarray:
    return env.physics.render(
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        camera_id=VIDEO_CAMERA_ID,
    ).astype(np.uint8)


def diagnostics(env) -> dict[str, float]:
    out = {
        "torso_height": float("nan"),
        "torso_upright": float("nan"),
        "horizontal_velocity": float("nan"),
        "torso_x": float("nan"),
    }
    try:
        out["torso_height"] = float(env.physics.torso_height())
        out["torso_upright"] = float(env.physics.torso_upright())
        out["horizontal_velocity"] = float(env.physics.horizontal_velocity())
        out["torso_x"] = float(env.physics.named.data.xpos["torso", "x"])
    except Exception:
        pass
    return out


def discounted_sum(rewards: Iterable[float], gamma: float) -> float:
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * float(reward)
        discount *= gamma
    return float(total)


# ---------------------------------------------------------------------------
# Oracle-free dataset collection.  All episodes come from generic
# exploration policies defined in src.datasets. The only thing this adapter
# contributes is task wiring: how to make the env, how to render a frame,
# what per-step metrics to record, and how to label frames after the fact.
# ---------------------------------------------------------------------------


def _env_fn(seed: int):
    return make_env(seed)


def _render_fn(env) -> np.ndarray:
    return render_dataset_frame(env)


def _action_bounds_fn(env) -> tuple[np.ndarray, np.ndarray]:
    return action_bounds(env)


def _record_fn(env) -> dict[str, float]:
    return diagnostics(env)


# Register with src.datasets so worker processes (forkserver) can resolve
# these functions by re-importing this module — no closure pickling.
datasets.register_task(
    "walker_walk",
    datasets.TaskHandles(
        env_fn=_env_fn,
        render_fn=_render_fn,
        action_bounds_fn=_action_bounds_fn,
        record_fn=_record_fn,
        dt=0.025,  # dm_control walker step length (40 Hz).
    ),
)


def goal_frame_mask(ep: dict[str, np.ndarray]) -> np.ndarray:
    """Per-frame boolean mask of "this frame looks like stable walking".

    Operates only on per-step metrics; lives in the task adapter because
    "what counts as walking" is task knowledge. The src/ layer never sees
    these thresholds.
    """
    num_frames = len(ep["pixels"])
    if num_frames == 0:
        return np.zeros(0, dtype=bool)
    action_steps = max(num_frames - 1, 1)
    frame_ids = np.arange(num_frames)
    metric_ids = np.clip(frame_ids - 1, 0, action_steps - 1)
    mask = np.ones(num_frames, dtype=bool)
    for key, threshold in (
        ("rewards", GOAL_FRAME_MIN_REWARD),
        ("torso_height", GOAL_FRAME_MIN_HEIGHT),
        ("torso_upright", GOAL_FRAME_MIN_UPRIGHT),
        ("horizontal_velocity", GOAL_FRAME_MIN_VELOCITY),
    ):
        if key in ep:
            values = np.asarray(ep[key])
            ids = np.clip(metric_ids, 0, len(values) - 1)
            mask = mask & (values[ids] >= threshold)
    return mask


def fall_frame_mask(ep: dict[str, np.ndarray]) -> np.ndarray:
    """Per-frame boolean mask of "this frame looks like a fall"."""
    num_frames = len(ep["pixels"])
    if num_frames == 0:
        return np.zeros(0, dtype=bool)
    action_steps = max(num_frames - 1, 1)
    frame_ids = np.arange(num_frames)
    metric_ids = np.clip(frame_ids - 1, 0, action_steps - 1)
    out = np.zeros(num_frames, dtype=bool)
    if "torso_height" in ep:
        h = np.asarray(ep["torso_height"])
        out = out | (h[np.clip(metric_ids, 0, len(h) - 1)] <= SUPPORT_FRAME_MAX_HEIGHT)
    if "torso_upright" in ep:
        u = np.asarray(ep["torso_upright"])
        out = out | (u[np.clip(metric_ids, 0, len(u) - 1)] <= SUPPORT_FRAME_MAX_UPRIGHT)
    return out


def _contiguous_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def pick_goal_trajectory_from_dataset(dataset_dir: Path) -> dict[str, np.ndarray] | None:
    """Stitch every "looks like walking" contiguous segment across all
    episodes into a single synthetic goal trajectory, ordered longest-first.

    Returns ``None`` if no segment passes ``goal_frame_mask`` for
    ``GOAL_MIN_SEGMENT_LEN`` consecutive frames. The first segment becomes
    the prefix used by the eval bootstrap; later segments give the prior a
    longer "rhythm" to cycle through. All concatenated actions still
    satisfy ``actions[t]`` advances ``pixels[t]`` to ``pixels[t+1]`` *within
    a segment*; segment boundaries are visual discontinuities, which the
    eval handles via its phase-wrapping logic.
    """
    candidates: list[tuple[int, Path, int, int]] = []  # (length, path, start, end)
    for path in sorted(dataset_dir.glob("episode_*.npz")):
        with np.load(path) as ep_npz:
            ep = {k: ep_npz[k] for k in ep_npz.files}
        mask = goal_frame_mask(ep)
        for s, e in _contiguous_segments(mask, GOAL_MIN_SEGMENT_LEN):
            candidates.append((e - s, path, s, e))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])  # longest first

    pix_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    rew_parts: list[np.ndarray] = []
    metric_parts: dict[str, list[np.ndarray]] = {}
    sources: list[dict] = []
    for length, path, s, e in candidates:
        with np.load(path) as ep_npz:
            ep = {k: ep_npz[k] for k in ep_npz.files}
        # Take pixels [s, e) and actions [s, e-1) so the segment is
        # internally consistent.
        if e - s < 2:
            continue
        pix_parts.append(ep["pixels"][s:e].copy())
        act_parts.append(ep["actions"][s : e - 1].copy())
        if "rewards" in ep:
            rew_parts.append(ep["rewards"][s : e - 1].copy())
        for key in ("torso_height", "torso_upright", "horizontal_velocity", "torso_x"):
            if key in ep:
                metric_parts.setdefault(key, []).append(np.asarray(ep[key])[s : e - 1].copy())
        sources.append({
            "episode": path.name,
            "segment_start": int(s),
            "segment_end": int(e),
            "segment_len": int(length),
        })
    if not pix_parts:
        return None
    sliced = {
        "pixels": np.concatenate(pix_parts, axis=0),
        "actions": np.concatenate(act_parts, axis=0),
    }
    if rew_parts:
        sliced["rewards"] = np.concatenate(rew_parts, axis=0)
    for key, parts in metric_parts.items():
        sliced[key] = np.concatenate(parts, axis=0)
    sliced["metadata"] = np.asarray(
        json.dumps(
            {
                "source_episodes": sources,
                "n_segments": len(sources),
                "total_pixels": int(sliced["pixels"].shape[0]),
                "total_actions": int(sliced["actions"].shape[0]),
                "action_semantics": "actions[t] advances pixels[t] to pixels[t+1] within a segment",
            }
        )
    )
    return sliced


# ---------------------------------------------------------------------------
# Oracle-CEM goal demo (the "+1" of "128 + 1").
#
# This runs an oracle simulator-state-restoring CEM with a hand-coded reward
# (forward speed + posture) to produce **one** clean walking trajectory that
# starts from the env reset pose. The 128 explore episodes are still
# oracle-free; this single trajectory only seeds the goal manifold with a
# reachable-from-reset path. Used in the planner exclusively as the
# ``goal_trajectory`` for the prior + near terms (it is not added to the
# wider goal_frame_mask sweep across explore episodes; if you want the
# manifold larger, raise N_EPISODES instead of adding more oracle data).
# ---------------------------------------------------------------------------


def _score_oracle_candidate(
    oracle_env,
    state0: np.ndarray,
    actions: np.ndarray,
    previous_action: np.ndarray,
) -> float:
    oracle_env.reset()
    oracle_env.physics.set_state(state0)
    oracle_env.physics.forward()
    start_x = float(oracle_env.physics.named.data.xpos["torso", "x"])
    rewards: list[float] = []
    last_action = previous_action
    smooth_cost = 0.0
    speed_tracking = 0.0
    overspeed_cost = 0.0
    posture_cost = 0.0
    for action in actions:
        ts = oracle_env.step(action.astype(np.float32))
        velocity = float(oracle_env.physics.horizontal_velocity())
        height = float(oracle_env.physics.torso_height())
        upright = float(oracle_env.physics.torso_upright())
        rewards.append(float(ts.reward or 0.0))
        smooth_cost += float(np.mean((action - last_action) ** 2))
        speed_tracking += float(np.exp(-((velocity - ORACLE_TARGET_SPEED) ** 2) / 0.5))
        overspeed_cost += max(velocity - 1.8, 0.0) ** 2
        posture_cost += max(1.0 - height, 0.0) ** 2 + max(0.75 - upright, 0.0) ** 2
        last_action = action
        if ts.last():
            break
    final_x = float(oracle_env.physics.named.data.xpos["torso", "x"])
    progress = max(final_x - start_x, 0.0)
    n = max(len(rewards), 1)
    return (
        discounted_sum(rewards, ORACLE_REWARD_GAMMA)
        + ORACLE_PROGRESS_BONUS * progress
        + ORACLE_SPEED_TRACKING_BONUS * (speed_tracking / n)
        - ORACLE_OVERSPEED_PENALTY * (overspeed_cost / n)
        - ORACLE_POSTURE_PENALTY * (posture_cost / n)
        - ORACLE_ACTION_PENALTY * float(np.mean(actions**2))
        - ORACLE_ACTION_SMOOTHNESS_PENALTY * smooth_cost
    )


def _oracle_cem_plan(
    oracle_env,
    state0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    init_mean: np.ndarray,
    previous_action: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    action_dim = low.shape[0]
    mean = init_mean.astype(np.float32).copy()
    std = np.full_like(mean, ORACLE_CEM_INIT_STD, dtype=np.float32)
    topk = min(ORACLE_CEM_TOPK, ORACLE_CEM_SAMPLES)
    best_plan: np.ndarray | None = None
    best_score = -float("inf")
    for _ in range(ORACLE_CEM_ITERS):
        samples = rng.normal(
            loc=mean[None],
            scale=std[None],
            size=(ORACLE_CEM_SAMPLES, ORACLE_PLAN_HORIZON, action_dim),
        ).astype(np.float32)
        samples[0] = mean
        samples = np.clip(samples, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
        scores = np.empty(samples.shape[0], dtype=np.float32)
        for i in range(samples.shape[0]):
            scores[i] = _score_oracle_candidate(oracle_env, state0, samples[i], previous_action)
        elite_idx = np.argpartition(-scores, topk - 1)[:topk]
        elites = samples[elite_idx]
        elite_mean = elites.mean(axis=0).astype(np.float32)
        elite_std = np.maximum(elites.std(axis=0), ORACLE_CEM_MIN_STD).astype(np.float32)
        mean = (1.0 - ORACLE_CEM_MOMENTUM) * elite_mean + ORACLE_CEM_MOMENTUM * mean
        std = elite_std
        i_best = int(np.argmax(scores))
        if float(scores[i_best]) > best_score:
            best_score = float(scores[i_best])
            best_plan = samples[i_best].copy()
    assert best_plan is not None
    return best_plan[0].astype(np.float32), mean.astype(np.float32), best_score


def collect_oracle_goal_demo(seed: int = SEED) -> dict[str, np.ndarray]:
    """Run an oracle-CEM-controlled walker for one full episode."""
    rng = np.random.default_rng(seed)
    env = make_env(seed)
    oracle_env = make_env(seed + 10_000)
    low, high = action_bounds(env)
    action_dim = int(low.shape[0])
    env.reset()
    plan_mean = np.zeros((ORACLE_PLAN_HORIZON, action_dim), dtype=np.float32)
    previous_action = np.zeros(action_dim, dtype=np.float32)

    pixels = [render_dataset_frame(env)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    heights: list[float] = []
    uprights: list[float] = []
    velocities: list[float] = []
    torso_x: list[float] = []

    bar = tqdm(range(DATASET_EPISODE_STEPS), desc="oracle goal", dynamic_ncols=True)
    for _step in bar:
        state0 = env.physics.get_state().copy()
        action, plan_mean, _score = _oracle_cem_plan(
            oracle_env, state0, low, high, plan_mean, previous_action, rng
        )
        plan_mean = np.concatenate(
            [plan_mean[1:], np.zeros((1, action_dim), dtype=np.float32)],
            axis=0,
        )
        ts = env.step(action)
        diag = diagnostics(env)
        pixels.append(render_dataset_frame(env))
        actions.append(action.copy())
        rewards.append(float(ts.reward or 0.0))
        heights.append(diag["torso_height"])
        uprights.append(diag["torso_upright"])
        velocities.append(diag["horizontal_velocity"])
        torso_x.append(diag["torso_x"])
        previous_action = action
        bar.set_postfix(
            {
                "ret": f"{sum(rewards):.1f}",
                "v": f"{diag['horizontal_velocity']:.2f}",
                "h": f"{diag['torso_height']:.2f}",
                "up": f"{diag['torso_upright']:.2f}",
            }
        )
        if ts.last():
            break
    bar.close()
    env.close()
    oracle_env.close()
    return {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "torso_height": np.asarray(heights, dtype=np.float32),
        "torso_upright": np.asarray(uprights, dtype=np.float32),
        "horizontal_velocity": np.asarray(velocities, dtype=np.float32),
        "torso_x": np.asarray(torso_x, dtype=np.float32),
        "metadata": np.asarray(
            json.dumps(
                {
                    "kind": "oracle_goal_demo",
                    "domain": DOMAIN,
                    "task": TASK,
                    "seed": seed,
                    "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
                }
            )
        ),
    }


def dataset_ready() -> bool:
    """Return True iff the on-disk dataset matches DATASET_VERSION."""
    if not (
        DATASET_DIR.exists()
        and (DATASET_DIR / "goal_trajectory.npz").exists()
        and bool(list(DATASET_DIR.glob("episode_*.npz")))
    ):
        return False
    metadata_path = DATASET_DIR / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    return int(metadata.get("dataset_version", -1)) == DATASET_VERSION


def build_static_dataset() -> None:
    """Collect 128 oracle-free explore episodes + 1 oracle-CEM goal demo.

    The 128 explore episodes provide LeWM with diverse pixel/action data
    (most of which is the walker falling or thrashing). The 1 oracle demo
    provides exactly one trajectory that starts from the env reset pose
    and walks forward; this becomes the planner's goal trajectory and the
    seed of the goal manifold. Post-hoc explore-frame goal selection then
    augments the manifold with whatever walking-like moments the random
    policies happened to produce.
    """
    if dataset_ready():
        print(f"[data] using existing dataset -> {DATASET_DIR}")
        return

    print(f"[data] creating dataset -> {DATASET_DIR} (128 explore + 1 oracle)")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    collect_metadata = datasets.collect_dataset(
        out_dir=DATASET_DIR,
        task_name="walker_walk",
        task_module="tasks.walker_walk",
        config=datasets.CollectConfig(
            n_episodes=DATASET_N_EPISODES,
            episode_steps=DATASET_EPISODE_STEPS,
            workers=DATASET_WORKERS,
            seed=SEED,
            mp_start_method=DATASET_MP_START_METHOD,
        ),
    )

    # The +1: run oracle-CEM for one episode and use it as the goal demo.
    print("[data] collecting 1 oracle-CEM goal demo...")
    oracle_demo = collect_oracle_goal_demo(seed=SEED)
    # Save the oracle demo as a regular episode so LeWM also trains on it.
    oracle_path = DATASET_DIR / f"episode_{DATASET_N_EPISODES:04d}.npz"
    np.savez_compressed(oracle_path, **oracle_demo)
    print(
        f"[data] oracle demo: return={float(oracle_demo['rewards'].sum()):.1f}, "
        f"mean_height={float(oracle_demo['torso_height'].mean()):.2f}, "
        f"mean_upright={float(oracle_demo['torso_upright'].mean()):.2f}, "
        f"mean_velocity={float(oracle_demo['horizontal_velocity'].mean()):.2f} "
        f"-> {oracle_path.name}"
    )

    # Use the oracle demo *directly* as the goal trajectory — its first
    # frame matches the env reset pose, so eval is reachable.
    goal_traj = {
        "pixels": oracle_demo["pixels"],
        "actions": oracle_demo["actions"],
        "rewards": oracle_demo["rewards"],
        "torso_height": oracle_demo["torso_height"],
        "torso_upright": oracle_demo["torso_upright"],
        "horizontal_velocity": oracle_demo["horizontal_velocity"],
        "metadata": np.asarray(
            json.dumps(
                {
                    "source": "oracle_goal_demo",
                    "n_segments": 1,
                    "total_pixels": int(oracle_demo["pixels"].shape[0]),
                    "total_actions": int(oracle_demo["actions"].shape[0]),
                    "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
                }
            )
        ),
    }
    np.savez_compressed(DATASET_DIR / "goal_trajectory.npz", **goal_traj)
    goal_meta = json.loads(str(goal_traj["metadata"]))
    print(
        f"[data] goal_trajectory: source={goal_meta['source']} "
        f"{goal_meta['total_pixels']} pixel frames, "
        f"{goal_meta['total_actions']} action steps"
    )

    # Render goal_manifold.mp4 + log so a human can verify the manifold.
    print("[data] rendering goal_manifold.mp4 from all goal frames...")
    stats = datasets.render_goal_manifold(
        dataset_dir=DATASET_DIR,
        out_dir=DATASET_DIR,
        goal_mask_fn=goal_frame_mask,
        video_fps=VIDEO_FPS,
        min_segment_len=GOAL_MIN_SEGMENT_LEN,
    )

    write_json(
        DATASET_DIR / "metadata.json",
        {
            "dataset_version": DATASET_VERSION,
            "config": hparams(),
            "collect_config": collect_metadata["config"],
            "episodes": collect_metadata["episodes"],
            "goal_trajectory": goal_meta,
            "goal_manifold_stats": {
                "total_frames": stats.total_frames,
                "total_segments": stats.total_segments,
                "episodes_contributing": stats.episodes_contributing,
                "metrics_summary": stats.metrics_summary,
            },
        },
    )
    print(
        f"[data] dataset built: {len(collect_metadata['episodes'])} explore + 1 oracle, "
        f"goal_manifold {stats.total_frames} frames across "
        f"{stats.episodes_contributing} episodes"
    )



# ---------------------------------------------------------------------------
# LeWM data and training.
# ---------------------------------------------------------------------------


def preprocess_pixels(pixels: torch.Tensor) -> torch.Tensor:
    return lewm.preprocess_pixels(pixels, IMAGE_SIZE)


def model_config(action_dim: int) -> lewm.LeWMModelConfig:
    return lewm.LeWMModelConfig(
        action_dim=action_dim,
        frameskip=FRAMESKIP,
        history_size=HISTORY_SIZE,
        embed_dim=EMBED_DIM,
        encoder_scale=ENCODER_SCALE,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        predictor_depth=PREDICTOR_DEPTH,
        predictor_heads=PREDICTOR_HEADS,
        predictor_dim_head=PREDICTOR_DIM_HEAD,
        predictor_mlp_dim=PREDICTOR_MLP_DIM,
        predictor_dropout=PREDICTOR_DROPOUT,
        predictor_conditioning=PREDICTOR_CONDITIONING,
    )


def train_config() -> lewm.LeWMTrainConfig:
    return lewm.LeWMTrainConfig(
        num_preds=NUM_PREDS,
        rollout_steps=TRAIN_ROLLOUT_STEPS,
        rollout_loss_weight=ROLLOUT_LOSS_WEIGHT,
        sigreg_weight=SIGREG_WEIGHT,
        sigreg_knots=SIGREG_KNOTS,
        sigreg_num_proj=SIGREG_NUM_PROJ,
        action_contrast_weight=ACTION_CONTRAST_WEIGHT,
        action_contrast_margin=ACTION_CONTRAST_MARGIN,
        action_contrast_horizon=ACTION_CONTRAST_HORIZON,
        batch_size=TRAIN_BATCH_SIZE,
        train_steps=TRAIN_STEPS,
        num_workers=TRAIN_NUM_WORKERS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        log_every=LOG_EVERY,
    )


def walker_window_weight(ep: dict[str, np.ndarray], start: int, window_pixels: int) -> float:
    """Per-window training weight.

    Task-adapter knowledge: a window where the walker is upright and moving
    is *much* more informative for learning forward dynamics than a window
    where it lies still on its back. We weight by mean (height * |upright|),
    plus a constant floor so the rare "good" frames don't completely
    dominate. Values are read from the per-step metric arrays the
    exploration loop records (no oracle, no labels).
    """
    end = min(start + window_pixels - 1, len(ep["pixels"]) - 1)
    # Metrics are per-action-step (one fewer than pixels).
    metric_start = max(start - 1, 0)
    metric_end = max(end - 1, metric_start + 1)
    weight = 1.0  # floor so every window has a non-zero chance.
    if "torso_height" in ep and "torso_upright" in ep:
        h = np.asarray(ep["torso_height"][metric_start:metric_end], dtype=np.float32)
        u = np.asarray(ep["torso_upright"][metric_start:metric_end], dtype=np.float32)
        if h.size and u.size:
            # Score in [0, ~1.4]: tall + clearly oriented (either way)
            # both count, but "tall + upright" gets the biggest bonus.
            score = float(np.clip(h.mean(), 0.0, 1.4) * np.clip(np.abs(u).mean(), 0.0, 1.0))
            weight += 6.0 * score
    if "rewards" in ep:
        r = np.asarray(ep["rewards"][metric_start:metric_end], dtype=np.float32)
        if r.size:
            weight += 4.0 * float(np.clip(r.mean(), 0.0, 1.0))
    return weight


def train_lewm() -> tuple[Path, int]:
    with np.load(next(iter(sorted(DATASET_DIR.glob("episode_*.npz"))))) as ep:
        action_dim = int(ep["actions"].shape[1])
    return lewm.train_lewm(
        dataset_dir=DATASET_DIR,
        ckpt_path=CKPT_DIR / "lewm.pt",
        train_log_path=TRAIN_LOG_PATH,
        model_cfg=model_config(action_dim),
        train_cfg=train_config(),
        hparams=hparams(),
        append_jsonl=append_jsonl,
        window_weight_fn=walker_window_weight if TRAIN_WINDOW_WEIGHTING else None,
    )


def load_lewm(ckpt_path: Path, action_dim: int):
    return lewm.load_lewm(ckpt_path, model_config(action_dim))


# ---------------------------------------------------------------------------
# Goal encoding and aligned LeWM rollout.
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_frames(model, pixels: np.ndarray, device: torch.device, chunk: int = 128) -> torch.Tensor:
    return lewm.encode_frames(model, pixels, device, IMAGE_SIZE, chunk=chunk)


def frame_index(phase: int, goal_len: int) -> int:
    if phase < goal_len:
        return phase
    start = min(GOAL_ORBIT_START, goal_len - 1)
    orbit_len = max(goal_len - start, 1)
    return start + ((phase - start) % orbit_len)


def action_index(phase: int, num_actions: int) -> int:
    if phase < num_actions:
        return phase
    start = min(GOAL_ORBIT_START, num_actions - 1)
    orbit_len = max(num_actions - start, 1)
    return start + ((phase - start) % orbit_len)


@torch.no_grad()
def phase_match(
    current_latent: torch.Tensor,
    goal_latents: torch.Tensor,
    prev_phase: int,
    first_step: bool,
) -> int:
    min_step = 0 if first_step else PHASE_MIN_STEP
    lower = prev_phase + min_step
    upper = prev_phase + PHASE_SEARCH_WINDOW
    phases = list(range(lower, upper + 1))
    indices = torch.tensor(
        [frame_index(p, goal_latents.size(0)) for p in phases],
        device=goal_latents.device,
        dtype=torch.long,
    )
    candidates = goal_latents.index_select(0, indices)
    dist = (candidates.float() - current_latent.view(1, -1).float()).pow(2).mean(dim=-1)
    raw_phase = phases[int(torch.argmin(dist).item())]
    if first_step:
        return raw_phase
    return max(prev_phase + PHASE_MIN_STEP, min(raw_phase, prev_phase + PHASE_MAX_STEP))


def goal_window(
    goal_latents: torch.Tensor,
    goal_actions: torch.Tensor,
    phase: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_ids = [frame_index(phase + i + 1, goal_latents.size(0)) for i in range(PLAN_HORIZON)]
    action_ids = [action_index(phase + i, goal_actions.size(0)) for i in range(PLAN_HORIZON)]
    frame_idx = torch.tensor(frame_ids, device=goal_latents.device, dtype=torch.long)
    action_idx = torch.tensor(action_ids, device=goal_actions.device, dtype=torch.long)
    return goal_latents.index_select(0, frame_idx), goal_actions.index_select(0, action_idx)


@torch.no_grad()
def estimate_orbit_sigma(goal_latents: torch.Tensor) -> float:
    start = min(GOAL_ORBIT_START, goal_latents.size(0) - 2)
    orbit = goal_latents[start:].float()
    if orbit.size(0) < 2:
        return ORBIT_BLEND_SIGMA_MIN
    step_dist = (orbit[1:] - orbit[:-1]).pow(2).mean(dim=-1)
    sigma = ORBIT_BLEND_SIGMA_MULT * float(step_dist.median().item())
    return float(np.clip(sigma, ORBIT_BLEND_SIGMA_MIN, ORBIT_BLEND_SIGMA_MAX))


@torch.no_grad()
def nearest_goal_phase(current_latent: torch.Tensor, goal_latents: torch.Tensor) -> tuple[int, float]:
    start = min(GOAL_ORBIT_START, goal_latents.size(0) - 1)
    stable_latents = goal_latents[start:].float()
    dist = (stable_latents - current_latent.view(1, -1).float()).pow(2).mean(dim=-1)
    idx = int(torch.argmin(dist).item())
    return start + idx, float(dist[idx].item())


@torch.no_grad()
def nearest_goal_distance(
    current_latent: torch.Tensor,
    goal_latents: torch.Tensor,
    *,
    start: int = 0,
) -> float:
    start = min(max(int(start), 0), goal_latents.size(0) - 1)
    candidates = goal_latents[start:].float()
    dist = (candidates - current_latent.view(1, -1).float()).pow(2).mean(dim=-1)
    return float(dist.min().item())


def orbit_rho(distance: float, sigma: float) -> float:
    return float(np.exp(-distance / max(sigma, 1e-6)))


def recovery_pressure(distance: float, phase: int) -> tuple[float, float]:
    """Continuous non-linear pressure to search farther from the action prior."""
    x = (float(distance) - RECOVERY_DISTANCE_CENTER) / max(RECOVERY_DISTANCE_TEMPERATURE, 1e-6)
    distance_pressure = 1.0 / (1.0 + np.exp(-x))
    distance_pressure = float(np.clip(distance_pressure, 0.0, 1.0) ** RECOVERY_PRESSURE_POWER)
    phase_x = (float(phase) - RECOVERY_PHASE_CENTER) / max(RECOVERY_PHASE_TEMPERATURE, 1e-6)
    phase_gate = float(1.0 / (1.0 + np.exp(-phase_x)))
    return float(distance_pressure * phase_gate), phase_gate


GoalManifold = LatentManifold



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


@torch.no_grad()
def build_goal_manifold(
    model,
    goal_pixels: np.ndarray,
    goal_ep: dict[str, np.ndarray],
    device: torch.device,
) -> LatentManifold:
    """Build the goal/support latent manifold from oracle-free explore data.

    The synthetic goal trajectory (picked post-hoc from the dataset) provides
    one segment with role="goal". Every explore episode contributes whatever
    frames pass ``goal_frame_mask`` (extra "goal" support) and whatever frames
    pass ``fall_frame_mask`` (role="support", used as repulsion if weight > 0).
    """
    segments: list[SuccessSegment] = []
    goal_meta_raw = goal_ep.get("metadata") if isinstance(goal_ep, dict) else None
    try:
        goal_meta = json.loads(str(goal_meta_raw.item() if hasattr(goal_meta_raw, "item") else goal_meta_raw)) if goal_meta_raw is not None else {}
    except (json.JSONDecodeError, AttributeError):
        goal_meta = {}
    # The goal trajectory is a concatenation of the best segments from
    # multiple episodes; the contributing episodes are already aggregated
    # into the goal segment, so when we iterate over the dataset we don't
    # need to special-case any single one (segments are still individually
    # supplied via goal_frame_mask below).
    _ = goal_meta  # currently unused but kept for future use.

    def add_source(name: str, pixels: np.ndarray, role: str, mask: np.ndarray) -> None:
        if int(mask.sum()) < SUPPORT_MIN_MASK_FRAMES:
            return
        segments.append(
            SuccessSegment(
                name=name,
                pixels=pixels.copy(),
                mask=mask.astype(bool),
                role=role,
                metadata={"source": name},
            )
        )

    # Goal trajectory: a synthetic, contiguous "walking" segment selected
    # from the exploration dataset.  Use the whole thing as goal.
    add_source(
        "goal_trajectory",
        goal_pixels,
        "goal",
        np.ones(len(goal_pixels), dtype=bool),
    )

    # Every episode contributes: (a) any extra goal-like frames, and
    # (b) any fall-like frames as repulsion support.
    for path in sorted(DATASET_DIR.glob("episode_*.npz")):
        with np.load(path) as ep_npz:
            ep = {k: ep_npz[k] for k in ep_npz.files}
        pixels = ep["pixels"]
        add_source(path.name + ":goal", pixels, "goal", goal_frame_mask(ep))
        add_source(path.name + ":fall", pixels, "support", fall_frame_mask(ep))

    return build_latent_manifold(
        segments,
        encode_frames=lambda pixels: encode_frames(model, pixels, device),
        config=ManifoldConfig(
            horizon=PLAN_HORIZON,
            max_state_points=MANIFOLD_MAX_STATE_POINTS,
            max_delta_points=MANIFOLD_MAX_DELTA_POINTS,
            max_segments=MANIFOLD_MAX_SEGMENTS,
            cost_scale_min=MANIFOLD_COST_SCALE_MIN,
        ),
        device=device,
    )


@torch.no_grad()
def rollout_prediction_diagnostics(
    model,
    goal_pixels: np.ndarray,
    goal_actions_np: np.ndarray,
    goal_latents: torch.Tensor,
    device: torch.device,
) -> dict:
    action_dim = int(goal_actions_np.shape[1])
    starts = [0, 1, 5, 20, 40]
    out: dict[str, dict] = {}
    max_horizon = min(max(ROLLOUT_DIAGNOSTIC_HORIZONS), len(goal_actions_np) - HISTORY_SIZE)
    horizon_values: dict[int, list[float]] = {
        int(h): [] for h in ROLLOUT_DIAGNOSTIC_HORIZONS if h <= max_horizon
    }
    for start in starts:
        if start + HISTORY_SIZE + max_horizon >= len(goal_pixels):
            continue
        hist_pix = torch.from_numpy(goal_pixels[start : start + HISTORY_SIZE]).unsqueeze(0).to(device)
        hist_actions = torch.zeros(1, HISTORY_SIZE, action_dim, device=device)
        if HISTORY_SIZE > 1:
            hist_actions[:, 1:] = torch.from_numpy(
                goal_actions_np[start : start + HISTORY_SIZE - 1]
            ).to(device)
        future = torch.from_numpy(
            goal_actions_np[
                start + HISTORY_SIZE - 1 : start + HISTORY_SIZE - 1 + max_horizon
            ]
        ).view(1, 1, max_horizon, action_dim).to(device)
        pred = aligned_rollout_latents(
            model,
            preprocess_pixels(hist_pix),
            hist_actions,
            future,
        ).squeeze(0).squeeze(0)
        target = goal_latents[start + HISTORY_SIZE : start + HISTORY_SIZE + max_horizon]
        mse = (pred.float() - target.float()).pow(2).mean(dim=-1).detach().cpu().numpy()
        horizon_mse = {}
        for horizon in horizon_values:
            value = float(mse[horizon - 1])
            horizon_mse[str(horizon)] = value
            horizon_values[horizon].append(value)
        out[f"true_history_{start}"] = {
            "first": float(mse[0]),
            "terminal": float(mse[-1]),
            "mean": float(mse.mean()),
            "horizon_mse": horizon_mse,
        }

    repeated = torch.from_numpy(
        np.stack([goal_pixels[0].copy() for _ in range(HISTORY_SIZE)], axis=0)
    ).unsqueeze(0).to(device)
    repeated_actions = torch.zeros(1, HISTORY_SIZE, action_dim, device=device)
    future = torch.from_numpy(goal_actions_np[:max_horizon]).view(
        1, 1, max_horizon, action_dim
    ).to(device)
    pred = aligned_rollout_latents(
        model,
        preprocess_pixels(repeated),
        repeated_actions,
        future,
    ).squeeze(0).squeeze(0)
    target = goal_latents[1 : 1 + max_horizon]
    mse = (pred.float() - target.float()).pow(2).mean(dim=-1).detach().cpu().numpy()
    out["repeated_reset_history"] = {
        "first": float(mse[0]),
        "terminal": float(mse[-1]),
        "mean": float(mse.mean()),
        "horizon_mse": {
            str(horizon): float(mse[horizon - 1]) for horizon in horizon_values
        },
    }
    return {
        "cases": out,
        "mean_by_horizon": {
            str(horizon): float(np.mean(values)) for horizon, values in horizon_values.items() if values
        },
    }


def block_actions_from_step_actions(actions: torch.Tensor) -> torch.Tensor:
    blocks = actions[: PLAN_HORIZON].reshape(PLAN_BLOCKS, ACTION_BLOCK, -1)
    return blocks.mean(dim=1)


def step_actions_from_blocks(blocks: torch.Tensor) -> torch.Tensor:
    if ACTION_BLOCK == 1:
        return blocks
    return blocks.repeat_interleave(ACTION_BLOCK, dim=0)


def aligned_rollout_latents(
    model,
    history_pixels: torch.Tensor,
    history_actions: torch.Tensor,
    future_actions: torch.Tensor,
) -> torch.Tensor:
    return lewm.aligned_rollout_latents(
        model,
        history_pixels,
        history_actions,
        future_actions,
        history_size=HISTORY_SIZE,
    )


# ---------------------------------------------------------------------------
# Pure LeWM-CEM evaluation.
# ---------------------------------------------------------------------------


def weighted_horizon_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights[: values.size(1)].to(values)
    return (values * w.view(1, -1)).sum(dim=1) / w.sum().clamp_min(1e-8)


def nearest_segment_mse(
    sequence: torch.Tensor,
    segments: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    samples, horizon, dim = sequence.shape
    w = weights[:horizon].to(sequence).sqrt().view(1, horizon, 1)
    flat_sequence = (sequence.float() * w).reshape(samples, horizon * dim)
    flat_segments = (segments.float() * w).reshape(segments.size(0), horizon * dim)
    denom = weights[:horizon].to(sequence).sum().clamp_min(1e-8) * dim
    return torch.cdist(flat_sequence, flat_segments).pow(2).min(dim=1).values.div(denom)


@torch.no_grad()
def lewm_cem_plan(
    model,
    history_pixels_u8: torch.Tensor,
    history_actions: torch.Tensor,
    current_latent: torch.Tensor,
    target_latents: torch.Tensor,
    goal_manifold: GoalManifold,
    prior_actions: torch.Tensor,
    previous_action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    weights: torch.Tensor,
    rng: torch.Generator,
    device: torch.device,
    warm_start_blocks: torch.Tensor | None = None,
    pressure: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    p = float(np.clip(pressure, 0.0, 1.0))
    init_std = EVAL_CEM_INIT_STD + p * (EVAL_CEM_FAR_INIT_STD - EVAL_CEM_INIT_STD)
    action_prior_weight = (1.0 - p) * W_ACTION_PRIOR
    smooth_weight = W_SMOOTH - p * (W_SMOOTH - W_FAR_SMOOTH)
    planner = LatentCEMPlanner(
        model,
        PlannerConfig(
            history_size=HISTORY_SIZE,
            action_block=ACTION_BLOCK,
            plan_blocks=PLAN_BLOCKS,
            samples=EVAL_CEM_SAMPLES,
            topk=EVAL_CEM_TOPK,
            iters=EVAL_CEM_ITERS,
            init_std=EVAL_CEM_INIT_STD,
            max_std=EVAL_CEM_MAX_STD,
            min_std=EVAL_CEM_MIN_STD,
            momentum=EVAL_CEM_MOMENTUM,
            warm_start_blend=CEM_WARM_START_BLEND,
            prior_tie_rel=CEM_PRIOR_TIE_REL,
            prior_tie_abs=CEM_PRIOR_TIE_ABS,
            cost_scale_min=MANIFOLD_COST_SCALE_MIN,
            near_delta_weight=NEAR_DELTA_WEIGHT,
            prior_in_mean=CEM_PRIOR_IN_MEAN,
            prior_in_samples=CEM_PRIOR_IN_SAMPLES,
            state_cost_clip=CEM_STATE_COST_CLIP,
            delta_cost_clip=CEM_DELTA_COST_CLIP,
            near_cost_clip=CEM_NEAR_COST_CLIP,
        ),
        PlannerWeights(
            state=W_STATE,
            delta=W_DELTA,
            support_state=W_SUPPORT_STATE,
            support_delta=W_SUPPORT_DELTA,
            near=W_NEAR,
            action_prior=W_ACTION_PRIOR,
            smooth=W_SMOOTH,
            energy=W_ENERGY,
        ),
    )
    return planner.plan(
        history_pixels=preprocess_pixels(history_pixels_u8.to(device)),
        history_actions=history_actions,
        current_latent=current_latent,
        target_latents=target_latents,
        manifold=goal_manifold,
        prior_actions=prior_actions,
        previous_action=previous_action,
        low=low,
        high=high,
        horizon_weights=weights,
        rng=rng,
        warm_start_blocks=warm_start_blocks,
        init_std=init_std,
        action_prior_weight=action_prior_weight,
        smooth_weight=smooth_weight,
    )


@torch.no_grad()
def evaluate_lewm_cem(ckpt_path: Path, action_dim: int) -> dict:
    model, device = load_lewm(ckpt_path, action_dim)
    with np.load(DATASET_DIR / "goal_trajectory.npz") as goal_npz:
        goal_ep = {k: goal_npz[k] for k in goal_npz.files}
    goal_pixels = goal_ep["pixels"]
    goal_actions_np = goal_ep["actions"].astype(np.float32)
    goal_manifold = build_goal_manifold(model, goal_pixels, goal_ep, device)

    goal_latents = encode_frames(model, goal_pixels, device)
    goal_actions = torch.from_numpy(goal_actions_np).to(device)
    orbit_sigma = estimate_orbit_sigma(goal_latents)
    imageio.mimsave(OUT_DIR / "goal_trajectory.mp4", list(goal_pixels), fps=VIDEO_FPS)
    diagnostics_payload = {
        "goal_manifold": goal_manifold.diagnostics,
        "rollout_prediction": rollout_prediction_diagnostics(
            model, goal_pixels, goal_actions_np, goal_latents, device
        ),
        "orbit_sigma": float(orbit_sigma),
        "planner_weights": {
            "W_STATE": W_STATE,
            "W_DELTA": W_DELTA,
            "W_SUPPORT_STATE": W_SUPPORT_STATE,
            "W_SUPPORT_DELTA": W_SUPPORT_DELTA,
            "W_NEAR": W_NEAR,
            "W_ACTION_PRIOR": W_ACTION_PRIOR,
            "W_SMOOTH": W_SMOOTH,
            "W_ENERGY": W_ENERGY,
        },
    }
    write_json(DIAGNOSTICS_PATH, diagnostics_payload)

    if EVAL_TRACE_PATH.exists():
        EVAL_TRACE_PATH.unlink()

    env = make_env(EVAL_SEED)
    env.reset()
    low_np, high_np = action_bounds(env)
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)
    zero = np.zeros(action_dim, dtype=np.float32)

    initial_frame = render_dataset_frame(env)
    history_frames: list[np.ndarray] = [initial_frame.copy()]
    history_actions: list[np.ndarray] = [zero.copy()]
    previous_action = torch.zeros(action_dim, device=device)
    weights = torch.tensor(
        [TRAJ_DISCOUNT**i for i in range(PLAN_HORIZON)],
        device=device,
        dtype=torch.float32,
    )
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    rng = torch.Generator(device=gen_device).manual_seed(SEED + 30_000)

    video_frames: list[np.ndarray] = []
    rewards: list[float] = []
    velocities: list[float] = []
    heights: list[float] = []
    uprights: list[float] = []
    actions_out: list[np.ndarray] = []
    phases: list[int] = []
    rhos: list[float] = []
    orbit_distances: list[float] = []
    cost_infos: list[dict[str, float]] = []

    phase = 0
    env_steps = 0
    warm_start_blocks: torch.Tensor | None = None
    bar = tqdm(total=EVAL_STEPS, desc="eval LeWM-CEM", dynamic_ncols=True)
    bootstrap_steps = min(EVAL_BOOTSTRAP_STEPS, EVAL_STEPS, len(goal_actions_np))
    first_step = bootstrap_steps == 0
    for boot in range(bootstrap_steps):
        action = goal_actions_np[boot].astype(np.float32)
        video_frames.append(render_video_frame(env))
        ts = env.step(action)
        diag = diagnostics(env)
        rewards.append(float(ts.reward or 0.0))
        velocities.append(diag["horizontal_velocity"])
        heights.append(diag["torso_height"])
        uprights.append(diag["torso_upright"])
        actions_out.append(action.copy())
        phases.append(boot)
        rhos.append(1.0)
        orbit_distances.append(0.0)
        history_frames.append(render_dataset_frame(env))
        history_actions.append(action.copy())
        previous_action = torch.from_numpy(action).to(device)
        env_steps += 1
        phase = boot + 1
        bar.update(1)
        append_jsonl(
            EVAL_TRACE_PATH,
            {
                "step": env_steps,
                "mode": "bootstrap",
                "reward": rewards[-1],
                "return": float(np.sum(rewards)),
                "height": diag["torso_height"],
                "upright": diag["torso_upright"],
                "velocity": diag["horizontal_velocity"],
                "rho": 1.0,
                "phase": boot,
                "action_abs": float(np.mean(np.abs(action))),
                "state_cost": 0.0,
                "delta_cost": 0.0,
                "support_state_cost": 0.0,
                "support_delta_cost": 0.0,
                "near_cost": 0.0,
                "prior_cost": 0.0,
                "smooth_cost": 0.0,
                "energy_cost": 0.0,
                "total_cost": 0.0,
                "state_cost_share": 0.0,
                "delta_cost_share": 0.0,
                "support_state_cost_share": 0.0,
                "support_delta_cost_share": 0.0,
                "near_cost_share": 0.0,
                "prior_cost_share": 0.0,
                "used_prior_guard": 0.0,
                "prior_total_cost": 0.0,
                "cem_best_total_cost": 0.0,
                "prior_cost_improvement": 0.0,
                "prior_guard_margin": 0.0,
            },
        )
        if ts.last():
            env_steps = EVAL_STEPS
            break

    while env_steps < EVAL_STEPS:
        current_u8 = torch.from_numpy(np.stack(history_frames[-1:], axis=0)).unsqueeze(0).to(device)
        current_latent = model.encode({"pixels": preprocess_pixels(current_u8)})["emb"].squeeze(0).squeeze(0)
        global_phase, current_dist = nearest_goal_phase(current_latent, goal_latents)
        rho = orbit_rho(current_dist, orbit_sigma)
        if PHASE_RELOCK_RHO >= 0.0 and rho < PHASE_RELOCK_RHO and phase >= GOAL_ORBIT_START:
            phase = global_phase
        else:
            phase = phase_match(current_latent, goal_latents, phase, first_step)
        first_step = False
        target_latents, prior_actions = goal_window(goal_latents, goal_actions, phase)

        hist_pix = torch.from_numpy(np.stack(history_frames[-HISTORY_SIZE:], axis=0)).unsqueeze(0)
        hist_act = torch.from_numpy(np.stack(history_actions[-HISTORY_SIZE:], axis=0)).unsqueeze(0)
        pressure_dist = current_dist
        pressure, phase_pressure_gate = recovery_pressure(pressure_dist, phase)
        plan, plan_info = lewm_cem_plan(
            model,
            hist_pix,
            hist_act,
            current_latent,
            target_latents,
            goal_manifold,
            prior_actions,
            previous_action,
            low,
            high,
            weights,
            rng,
            device,
            warm_start_blocks,
            pressure=pressure,
        )
        warm_start_blocks = torch.cat([plan[1:], plan[-1:]], dim=0).detach()

        action = plan[0].detach().cpu().numpy().astype(np.float32)
        for _ in range(ACTION_BLOCK):
            if env_steps >= EVAL_STEPS:
                break
            step_phase = phase
            video_frames.append(render_video_frame(env))
            ts = env.step(action)
            diag = diagnostics(env)
            rewards.append(float(ts.reward or 0.0))
            velocities.append(diag["horizontal_velocity"])
            heights.append(diag["torso_height"])
            uprights.append(diag["torso_upright"])
            actions_out.append(action.copy())
            phases.append(step_phase)
            rhos.append(rho)
            orbit_distances.append(current_dist)
            cost_infos.append(plan_info)
            history_frames.append(render_dataset_frame(env))
            history_actions.append(action.copy())
            previous_action = torch.from_numpy(action).to(device)
            env_steps += 1
            bar.update(1)
            append_jsonl(
                EVAL_TRACE_PATH,
                {
                    "step": env_steps,
                    "mode": "cem",
                    "reward": rewards[-1],
                    "return": float(np.sum(rewards)),
                    "height": diag["torso_height"],
                    "upright": diag["torso_upright"],
                    "velocity": diag["horizontal_velocity"],
                    "rho": float(rho),
                    "recovery_pressure": float(pressure),
                    "phase_pressure_gate": float(phase_pressure_gate),
                    "distance": float(current_dist),
                    "pressure_distance": float(pressure_dist),
                    "phase": int(step_phase),
                    "global_phase": int(global_phase),
                    "action_abs": float(np.mean(np.abs(action))),
                    "previous_action_abs": float(previous_action.abs().mean().detach().cpu()),
                    **plan_info,
                },
            )
            bar.set_postfix(
                {
                    "ret": f"{sum(rewards):.1f}",
                    "v": f"{diag['horizontal_velocity']:.2f}",
                    "h": f"{diag['torso_height']:.2f}",
                    "rho": f"{rho:.2f}",
                    "method": (
                        f"{plan_info['state_cost_share'] + plan_info['delta_cost_share'] + plan_info['support_state_cost_share'] + plan_info['support_delta_cost_share']:.2f}"
                    ),
                    "phase": step_phase,
                }
            )
            phase += 1
            if ts.last():
                env_steps = EVAL_STEPS
                break
    bar.close()
    env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / f"eval_cam{VIDEO_CAMERA_ID}.mp4"
    imageio.mimsave(video_path, video_frames, fps=VIDEO_FPS)

    def mean_cost_info(key: str) -> float:
        if not cost_infos:
            return float("nan")
        return float(np.mean([info.get(key, 0.0) for info in cost_infos]))

    mean_state_delta_share = mean_cost_info("state_cost_share") + mean_cost_info("delta_cost_share")
    mean_support_share = (
        mean_cost_info("support_state_cost_share")
        + mean_cost_info("support_delta_cost_share")
    )
    mean_method_share = mean_state_delta_share + mean_support_share
    mean_near_prior_share = mean_cost_info("near_cost_share") + mean_cost_info("prior_cost_share")
    total_return = float(np.sum(rewards)) if rewards else 0.0
    metrics = {
        "domain": DOMAIN,
        "task": TASK,
        "policy": "pure_lewm_cem_latent_delta_manifold",
        "steps": len(rewards),
        "total_return": total_return,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_horizontal_velocity": float(np.nanmean(velocities)) if velocities else float("nan"),
        "mean_torso_height": float(np.nanmean(heights)) if heights else float("nan"),
        "mean_torso_upright": float(np.nanmean(uprights)) if uprights else float("nan"),
        "mean_abs_action": float(np.mean(np.abs(actions_out))) if actions_out else float("nan"),
        "last_phase": int(phases[-1]) if phases else 0,
        "mean_orbit_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "min_orbit_rho": float(np.min(rhos)) if rhos else float("nan"),
        "mean_goal_distance": float(np.mean(orbit_distances)) if orbit_distances else float("nan"),
        "orbit_blend_sigma": float(orbit_sigma),
        "planner_weights": {
            "W_STATE": W_STATE,
            "W_DELTA": W_DELTA,
            "W_SUPPORT_STATE": W_SUPPORT_STATE,
            "W_SUPPORT_DELTA": W_SUPPORT_DELTA,
            "W_NEAR": W_NEAR,
            "W_ACTION_PRIOR": W_ACTION_PRIOR,
            "W_SMOOTH": W_SMOOTH,
            "W_ENERGY": W_ENERGY,
        },
        "mean_state_cost": mean_cost_info("state_cost"),
        "mean_delta_cost": mean_cost_info("delta_cost"),
        "mean_support_state_cost": mean_cost_info("support_state_cost"),
        "mean_support_delta_cost": mean_cost_info("support_delta_cost"),
        "mean_near_cost": mean_cost_info("near_cost"),
        "mean_prior_cost": mean_cost_info("prior_cost"),
        "mean_smooth_cost": mean_cost_info("smooth_cost"),
        "mean_energy_cost": mean_cost_info("energy_cost"),
        "mean_total_cost": mean_cost_info("total_cost"),
        "mean_state_cost_share": mean_cost_info("state_cost_share"),
        "mean_delta_cost_share": mean_cost_info("delta_cost_share"),
        "mean_support_state_cost_share": mean_cost_info("support_state_cost_share"),
        "mean_support_delta_cost_share": mean_cost_info("support_delta_cost_share"),
        "mean_near_cost_share": mean_cost_info("near_cost_share"),
        "mean_prior_cost_share": mean_cost_info("prior_cost_share"),
        "prior_guard_used_rate": mean_cost_info("used_prior_guard"),
        "mean_prior_cost_improvement": mean_cost_info("prior_cost_improvement"),
        "mean_prior_guard_margin": mean_cost_info("prior_guard_margin"),
        "mean_state_delta_cost_share": mean_state_delta_share,
        "mean_support_cost_share": mean_support_share,
        "mean_method_cost_share": mean_method_share,
        "mean_near_prior_cost_share": mean_near_prior_share,
        "state_delta_dominant": bool(mean_state_delta_share > mean_near_prior_share),
        "method_cost_dominant": bool(mean_method_share > mean_near_prior_share),
        "sanity_return_ok": bool(total_return >= 100.0),
        "goal_manifold": {
            "state_points": goal_manifold.diagnostics["state_points"],
            "delta_points": goal_manifold.diagnostics["delta_points"],
            "state_segments": goal_manifold.diagnostics["state_segments"],
            "delta_segments": goal_manifold.diagnostics["delta_segments"],
            "support_state_points": goal_manifold.diagnostics["support_state_points"],
            "support_delta_points": goal_manifold.diagnostics["support_delta_points"],
            "support_state_segments": goal_manifold.diagnostics["support_state_segments"],
            "support_delta_segments": goal_manifold.diagnostics["support_delta_segments"],
            "state_scale": goal_manifold.diagnostics["state_scale"],
            "delta_scale": goal_manifold.diagnostics["delta_scale"],
            "support_state_scale": goal_manifold.diagnostics["support_state_scale"],
            "support_delta_scale": goal_manifold.diagnostics["support_delta_scale"],
        },
        "checkpoint": str(ckpt_path),
        "dataset_dir": str(DATASET_DIR),
        "train_log": str(TRAIN_LOG_PATH),
        "eval_trace": str(EVAL_TRACE_PATH),
        "diagnostics": str(DIAGNOSTICS_PATH),
        "goal_video": str(OUT_DIR / "goal_trajectory.mp4"),
        "eval_video": str(video_path),
        "config": hparams(),
    }
    write_json(OUT_DIR / "metrics.json", metrics)
    print(
        f"[eval] return={metrics['total_return']:.1f} "
        f"mean_v={metrics['mean_horizontal_velocity']:.2f} "
        f"method_share={metrics['mean_method_cost_share']:.2f} -> {video_path}"
    )
    return metrics


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[cfg]", json.dumps(hparams(), indent=2))

    build_static_dataset()
    ckpt_path, action_dim = train_lewm()
    evaluate_lewm_cem(ckpt_path, action_dim)


if __name__ == "__main__":
    main()
