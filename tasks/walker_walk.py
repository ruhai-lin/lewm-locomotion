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
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from queue import Empty
from typing import Iterable, List, Tuple

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

from src import lewm
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
DATASET_VERSION = 5


# ---------------------------------------------------------------------------
# Offline dataset generation.
# ---------------------------------------------------------------------------


IMAGE_SIZE = 112
DATASET_CAMERA_ID = 0
VIDEO_CAMERA_ID = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 40

CLEAN_DEMO_EPISODES = 4
PERTURBED_DEMO_EPISODES = 2
DATASET_EPISODE_STEPS = 400
DATASET_WORKERS = min(8, os.cpu_count() or 1)
DATASET_MP_START_METHOD = "forkserver"
MAX_ATTEMPTS_PER_SUCCESS = 6
CLEAN_MIN_RETURN = 250.0
CLEAN_MIN_MEAN_HEIGHT = 1.0
QUALITY_TAIL_STEPS = 80
TAIL_MIN_REWARD = 0.85
TAIL_MIN_HEIGHT = 1.05
TAIL_MIN_UPRIGHT = 0.85
TAIL_MIN_VELOCITY = 0.5
RECOVERY_MAX_MIN_HEIGHT = 0.9
RECOVERY_MAX_MIN_UPRIGHT = 0.5

ORACLE_PLAN_HORIZON = 20
ORACLE_CEM_SAMPLES = 128
ORACLE_CEM_TOPK = 16
ORACLE_CEM_ITERS = 4
ORACLE_CEM_INIT_STD = 0.55
ORACLE_CEM_MIN_STD = 0.06
ORACLE_CEM_MOMENTUM = 0.15
ORACLE_REWARD_GAMMA = 0.98

TARGET_SPEED = 1.2
PROGRESS_BONUS = 0.75
SPEED_TRACKING_BONUS = 0.12
OVERSPEED_PENALTY = 0.35
POSTURE_PENALTY = 1.2
ACTION_PENALTY = 0.01
ACTION_SMOOTHNESS_PENALTY = 0.015

PERTURB_SCHEDULE: list[tuple[int, int, float]] = [
    (50, 12, 0.15),
    (130, 12, 0.25),
    (220, 14, 0.35),
    (310, 14, 0.50),
]
OU_RHO = float(np.exp(-0.025 / 0.15))

RECOVERY_STEPS = 260
RECOVERY_REPEATS = 2
RECOVERY_POSES: list[dict[str, float | str]] = [
    {
        "name": "forward_prone",
        "rootz": -0.72,
        "rooty": 1.55,
        "qvel_rootx": 7.5,
        "qvel_rootz": -2.0,
        "qvel_rooty": 8.0,
    },
    {
        "name": "backward_supine",
        "rootz": -0.72,
        "rooty": -1.55,
        "qvel_rootx": -7.5,
        "qvel_rootz": -2.0,
        "qvel_rooty": -8.0,
    },
    {
        "name": "upward_flip",
        "rootz": -0.35,
        "rooty": 3.0,
        "qvel_rootx": 2.0,
        "qvel_rootz": 7.0,
        "qvel_rooty": 10.0,
    },
    {
        "name": "downward_crumple",
        "rootz": -0.92,
        "rooty": 2.65,
        "qvel_rootx": -2.0,
        "qvel_rootz": -8.0,
        "qvel_rooty": -10.0,
    },
]


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

SIGREG_WEIGHT = 0.07
SIGREG_KNOTS = 17
SIGREG_NUM_PROJ = 1024
TRAIN_BATCH_SIZE = 64
TRAIN_STEPS = 9000
TRAIN_NUM_WORKERS = 0
LR = 5e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
LOG_EVERY = 50
TRAIN_ROLLOUT_STEPS = 16
ROLLOUT_LOSS_WEIGHT = 1.0
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
CEM_PRIOR_TIE_ABS = 0.05
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
W_NEAR = 0.01
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


def perturb_sigma(step: int) -> float:
    for start, length, sigma in PERTURB_SCHEDULE:
        if start <= step < start + length:
            return sigma
    return 0.0


def burst_starts(step: int) -> bool:
    return any(step == start for start, _, _ in PERTURB_SCHEDULE)


def emit_progress(progress_queue, progress_id: int | None, postfix: dict) -> None:
    if progress_queue is not None and progress_id is not None:
        progress_queue.put((int(progress_id), 1, postfix))


def score_oracle_candidate(
    oracle_env,
    state0: np.ndarray,
    actions: np.ndarray,
    previous_action: np.ndarray,
) -> float:
    oracle_env.reset()
    oracle_env.physics.set_state(state0)
    oracle_env.physics.forward()
    start_x = float(oracle_env.physics.named.data.xpos["torso", "x"])

    rewards: List[float] = []
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
        speed_tracking += float(np.exp(-((velocity - TARGET_SPEED) ** 2) / 0.5))
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
        + PROGRESS_BONUS * progress
        + SPEED_TRACKING_BONUS * (speed_tracking / n)
        - OVERSPEED_PENALTY * (overspeed_cost / n)
        - POSTURE_PENALTY * (posture_cost / n)
        - ACTION_PENALTY * float(np.mean(actions**2))
        - ACTION_SMOOTHNESS_PENALTY * smooth_cost
    )


def oracle_cem_plan(
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
            scores[i] = score_oracle_candidate(
                oracle_env, state0, samples[i], previous_action
            )

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


def collect_oracle_episode(
    seed: int,
    noisy: bool,
    desc: str,
    progress: bool = True,
    progress_queue=None,
    progress_id: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    env = make_env(seed)
    oracle_env = make_env(seed + 10_000)
    low, high = action_bounds(env)
    action_dim = int(low.shape[0])
    env.reset()

    plan_mean = np.zeros((ORACLE_PLAN_HORIZON, action_dim), dtype=np.float32)
    previous_action = np.zeros(action_dim, dtype=np.float32)
    eta = np.zeros(action_dim, dtype=np.float32)
    ou_scale = float(np.sqrt(1.0 - OU_RHO * OU_RHO))

    pixels: List[np.ndarray] = [render_dataset_frame(env)]
    actions: List[np.ndarray] = []
    oracle_actions: List[np.ndarray] = []
    rewards: List[float] = []
    heights: List[float] = []
    uprights: List[float] = []
    velocities: List[float] = []
    torso_x: List[float] = []
    sigmas: List[float] = []
    planner_scores: List[float] = []

    bar = tqdm(
        range(DATASET_EPISODE_STEPS),
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not progress,
    )
    for step in bar:
        state0 = env.physics.get_state().copy()
        oracle_action, plan_mean, score = oracle_cem_plan(
            oracle_env, state0, low, high, plan_mean, previous_action, rng
        )
        plan_mean = np.concatenate(
            [plan_mean[1:], np.zeros((1, action_dim), dtype=np.float32)],
            axis=0,
        )

        sigma = perturb_sigma(step) if noisy else 0.0
        if burst_starts(step):
            eta[:] = 0.0
        if sigma > 0.0:
            eta = OU_RHO * eta + ou_scale * rng.standard_normal(action_dim).astype(np.float32)
            action = np.clip(oracle_action + sigma * eta, low, high).astype(np.float32)
        else:
            eta[:] = 0.0
            action = oracle_action

        ts = env.step(action)
        diag = diagnostics(env)
        pixels.append(render_dataset_frame(env))
        actions.append(action.copy())
        oracle_actions.append(oracle_action.copy())
        rewards.append(float(ts.reward or 0.0))
        heights.append(diag["torso_height"])
        uprights.append(diag["torso_upright"])
        velocities.append(diag["horizontal_velocity"])
        torso_x.append(diag["torso_x"])
        sigmas.append(sigma)
        planner_scores.append(float(score))
        previous_action = action

        bar.set_postfix(
            {
                "ret": f"{sum(rewards):.1f}",
                "v": f"{diag['horizontal_velocity']:.2f}",
                "h": f"{diag['torso_height']:.2f}",
                "up": f"{diag['torso_upright']:.2f}",
                "sig": f"{sigma:.2f}",
            }
        )
        emit_progress(
            progress_queue,
            progress_id,
            {
                "ret": f"{sum(rewards):.1f}",
                "v": f"{diag['horizontal_velocity']:.2f}",
                "h": f"{diag['torso_height']:.2f}",
                "up": f"{diag['torso_upright']:.2f}",
                "sig": f"{sigma:.2f}",
            },
        )
        if ts.last():
            break

    bar.close()
    env.close()
    oracle_env.close()

    metadata = {
        "domain": DOMAIN,
        "task": TASK,
        "seed": seed,
        "noisy": noisy,
        "perturb_schedule": [
            {"start": s, "length": length, "sigma": sigma}
            for s, length, sigma in PERTURB_SCHEDULE
        ],
        "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
    }
    return {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "oracle_actions": np.stack(oracle_actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "torso_height": np.asarray(heights, dtype=np.float32),
        "torso_upright": np.asarray(uprights, dtype=np.float32),
        "horizontal_velocity": np.asarray(velocities, dtype=np.float32),
        "torso_x": np.asarray(torso_x, dtype=np.float32),
        "sigmas": np.asarray(sigmas, dtype=np.float32),
        "planner_scores": np.asarray(planner_scores, dtype=np.float32),
        "metadata": json.dumps(metadata),
    }


def apply_recovery_pose(env, scenario: dict[str, float | str], rng: np.random.Generator) -> None:
    """Place walker in a hard off-manifold pose before oracle recovery."""
    env.physics.named.data.qpos["rootz"] = float(scenario["rootz"])
    env.physics.named.data.qpos["rooty"] = float(scenario["rooty"])
    env.physics.named.data.qvel["rootx"] = float(scenario["qvel_rootx"])
    env.physics.named.data.qvel["rootz"] = float(scenario["qvel_rootz"])
    env.physics.named.data.qvel["rooty"] = float(scenario["qvel_rooty"])
    for name in [
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    ]:
        env.physics.named.data.qpos[name] += float(rng.normal(0.0, 0.08))
        env.physics.named.data.qvel[name] = float(rng.normal(0.0, 1.0))
    env.physics.forward()


def collect_recovery_episode(
    seed: int,
    scenario: dict[str, float | str],
    desc: str,
    progress: bool = True,
    progress_queue=None,
    progress_id: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    env = make_env(seed)
    oracle_env = make_env(seed + 10_000)
    low, high = action_bounds(env)
    action_dim = int(low.shape[0])
    env.reset()
    apply_recovery_pose(env, scenario, rng)

    plan_mean = np.zeros((ORACLE_PLAN_HORIZON, action_dim), dtype=np.float32)
    previous_action = np.zeros(action_dim, dtype=np.float32)
    pixels: List[np.ndarray] = [render_dataset_frame(env)]
    actions: List[np.ndarray] = []
    oracle_actions: List[np.ndarray] = []
    rewards: List[float] = []
    heights: List[float] = []
    uprights: List[float] = []
    velocities: List[float] = []
    torso_x: List[float] = []
    sigmas: List[float] = []
    planner_scores: List[float] = []

    bar = tqdm(
        range(RECOVERY_STEPS),
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not progress,
    )
    for _ in bar:
        state0 = env.physics.get_state().copy()
        oracle_action, plan_mean, score = oracle_cem_plan(
            oracle_env, state0, low, high, plan_mean, previous_action, rng
        )
        plan_mean = np.concatenate(
            [plan_mean[1:], np.zeros((1, action_dim), dtype=np.float32)],
            axis=0,
        )
        ts = env.step(oracle_action)
        diag = diagnostics(env)
        pixels.append(render_dataset_frame(env))
        actions.append(oracle_action.copy())
        oracle_actions.append(oracle_action.copy())
        rewards.append(float(ts.reward or 0.0))
        heights.append(diag["torso_height"])
        uprights.append(diag["torso_upright"])
        velocities.append(diag["horizontal_velocity"])
        torso_x.append(diag["torso_x"])
        sigmas.append(0.0)
        planner_scores.append(float(score))
        previous_action = oracle_action
        bar.set_postfix(
            {
                "ret": f"{sum(rewards):.1f}",
                "v": f"{diag['horizontal_velocity']:.2f}",
                "h": f"{diag['torso_height']:.2f}",
                "up": f"{diag['torso_upright']:.2f}",
            }
        )
        emit_progress(
            progress_queue,
            progress_id,
            {
                "ret": f"{sum(rewards):.1f}",
                "v": f"{diag['horizontal_velocity']:.2f}",
                "h": f"{diag['torso_height']:.2f}",
                "up": f"{diag['torso_upright']:.2f}",
            },
        )
        if ts.last():
            break

    bar.close()
    env.close()
    oracle_env.close()
    metadata = {
        "domain": DOMAIN,
        "task": TASK,
        "seed": seed,
        "kind": "scripted_recovery",
        "scenario": scenario,
        "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
    }
    return {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "oracle_actions": np.stack(oracle_actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "torso_height": np.asarray(heights, dtype=np.float32),
        "torso_upright": np.asarray(uprights, dtype=np.float32),
        "horizontal_velocity": np.asarray(velocities, dtype=np.float32),
        "torso_x": np.asarray(torso_x, dtype=np.float32),
        "sigmas": np.asarray(sigmas, dtype=np.float32),
        "planner_scores": np.asarray(planner_scores, dtype=np.float32),
        "metadata": json.dumps(metadata),
    }


def dataset_ready() -> bool:
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


def save_episode(path: Path, episode: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **episode)


def dataset_targets() -> dict[str, int]:
    targets = {
        "clean": CLEAN_DEMO_EPISODES,
        "perturbed": PERTURBED_DEMO_EPISODES,
    }
    for scenario in RECOVERY_POSES:
        targets[f"recovery/{scenario['name']}"] = RECOVERY_REPEATS
    return targets


def make_attempt_job(kind: str, attempt: int, idx: int, have: int, target: int) -> dict:
    desc = f"try{idx:03d} {kind} {have + 1}/{target}"
    job = {
        "idx": idx,
        "kind": kind,
        "bar_desc": desc,
        "desc": desc,
    }
    if kind == "clean":
        job.update(
            {
                "collector": "oracle",
                "seed": SEED + attempt,
                "noisy": False,
            }
        )
        return job
    if kind == "perturbed":
        job.update(
            {
                "collector": "oracle",
                "seed": SEED + 1_000 + attempt,
                "noisy": True,
            }
        )
        return job

    scenario_name = kind.split("/", 1)[1]
    for scenario_idx, scenario in enumerate(RECOVERY_POSES):
        if scenario["name"] == scenario_name:
            job.update(
                {
                    "collector": "recovery",
                    "seed": SEED + 2_000 + scenario_idx * 10_000 + attempt,
                    "scenario": dict(scenario),
                }
            )
            return job
    raise ValueError(f"unknown dataset kind: {kind}")


def collect_episode_job(job: dict) -> tuple[int, str, dict]:
    progress = bool(job.get("progress", False))
    progress_queue = job.get("progress_queue")
    progress_id = int(job["idx"]) if progress_queue is not None else None
    if job["collector"] == "oracle":
        episode = collect_oracle_episode(
            seed=int(job["seed"]),
            noisy=bool(job["noisy"]),
            desc=str(job["desc"]),
            progress=progress,
            progress_queue=progress_queue,
            progress_id=progress_id,
        )
    elif job["collector"] == "recovery":
        episode = collect_recovery_episode(
            seed=int(job["seed"]),
            scenario=job["scenario"],
            desc=str(job["desc"]),
            progress=progress,
            progress_queue=progress_queue,
            progress_id=progress_id,
        )
    else:
        raise ValueError(f"unknown dataset collector: {job['collector']}")
    return int(job["idx"]), str(job["kind"]), episode


def collect_dataset_episodes_sequential(
    jobs: list[dict],
    save_dir: Path,
) -> list[tuple[Path, str]]:
    saved: list[tuple[Path, str]] = []
    for job in jobs:
        idx, kind, ep = collect_episode_job({**job, "progress": True})
        path = save_dir / f"candidate_{idx:04d}.npz"
        save_episode(path, ep)
        saved.append((path, kind))
    return saved


def remove_partial_episodes(save_dir: Path) -> None:
    for path in save_dir.glob("candidate_*.npz"):
        path.unlink()


def job_total(job: dict) -> int:
    return RECOVERY_STEPS if job["collector"] == "recovery" else DATASET_EPISODE_STEPS


def job_bar_desc(job: dict) -> str:
    return str(job.get("bar_desc", f"try{int(job['idx']):03d} {job['kind']}"))


def collect_dataset_episodes(
    jobs: list[dict],
    save_dir: Path,
) -> list[tuple[Path, str]]:
    workers = max(1, min(DATASET_WORKERS, len(jobs)))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] collecting {len(jobs)} candidate episodes with workers={workers}")

    if workers == 1:
        return collect_dataset_episodes_sequential(jobs, save_dir)

    try:
        ctx = mp.get_context(DATASET_MP_START_METHOD)
        saved: list[tuple[Path, str]] = []
        futures = {}
        with ctx.Manager() as manager:
            progress_queue = manager.Queue()
            bars = {
                int(job["idx"]): tqdm(
                    total=job_total(job),
                    desc=job_bar_desc(job),
                    position=pos,
                    leave=True,
                    dynamic_ncols=True,
                )
                for pos, job in enumerate(jobs)
            }
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    for job in jobs:
                        futures[
                            pool.submit(
                                collect_episode_job,
                                {**job, "progress_queue": progress_queue},
                            )
                        ] = job
                    pending = set(futures)
                    while pending:
                        while True:
                            try:
                                progress_id, n_steps, postfix = progress_queue.get_nowait()
                            except Empty:
                                break
                            bar = bars.get(int(progress_id))
                            if bar is not None:
                                step_inc = min(
                                    int(n_steps),
                                    max(int(bar.total or 0) - int(bar.n), 0),
                                )
                                if step_inc > 0:
                                    bar.update(step_inc)
                                if postfix:
                                    bar.set_postfix(postfix)

                        done = [fut for fut in pending if fut.done()]
                        if not done:
                            time.sleep(0.05)
                            continue
                        for fut in done:
                            pending.remove(fut)
                            idx, kind, ep = fut.result()
                            path = save_dir / f"candidate_{idx:04d}.npz"
                            save_episode(path, ep)
                            saved.append((path, kind))
                            bar = bars.get(idx)
                            if bar is not None and bar.n < bar.total:
                                bar.update(bar.total - bar.n)
                        for bar in bars.values():
                            bar.refresh()

                while True:
                    try:
                        progress_id, n_steps, postfix = progress_queue.get_nowait()
                    except Empty:
                        break
                    bar = bars.get(int(progress_id))
                    if bar is not None:
                        step_inc = min(int(n_steps), max(int(bar.total or 0) - int(bar.n), 0))
                        if step_inc > 0:
                            bar.update(step_inc)
                        if postfix:
                            bar.set_postfix(postfix)
            finally:
                for bar in bars.values():
                    bar.close()
        return sorted(saved, key=lambda item: item[0].name)
    except Exception as exc:
        print(f"[data] parallel collection failed ({type(exc).__name__}: {exc})")
        print("[data] retrying dataset collection sequentially")
        remove_partial_episodes(save_dir)
        return collect_dataset_episodes_sequential(jobs, save_dir)


def build_static_dataset() -> None:
    if dataset_ready():
        print(f"[data] using existing dataset -> {DATASET_DIR}")
        return

    print(f"[data] creating static dataset -> {DATASET_DIR}")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    kept, summaries = collect_successful_dataset()

    clean_paths = [path for path, kind in kept if kind == "clean"]
    if not clean_paths:
        raise RuntimeError("No successful clean walking demos were collected.")

    goal_path = pick_best_clean_episode(clean_paths)
    with np.load(goal_path) as goal:
        np.savez_compressed(
            DATASET_DIR / "goal_trajectory.npz",
            pixels=goal["pixels"],
            actions=goal["actions"],
            rewards=goal["rewards"],
            torso_height=goal["torso_height"],
            torso_upright=goal["torso_upright"],
            horizontal_velocity=goal["horizontal_velocity"],
            metadata=json.dumps(
                {
                    "source_episode": goal_path.name,
                    "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
                }
            ),
        )

    write_json(
        DATASET_DIR / "metadata.json",
        {
            "dataset_version": DATASET_VERSION,
            "config": hparams(),
            "episodes": summaries,
            "goal_episode": goal_path.name,
        },
    )
    recovery_count = sum(kind.startswith("recovery/") for _, kind in kept)
    skipped = sum(bool(summary.get("skipped", False)) for summary in summaries)
    print(
        f"[data] wrote {len(kept)} usable episodes "
        f"({recovery_count} recovery, {skipped} skipped) and goal trajectory"
    )


def collect_successful_dataset() -> tuple[list[tuple[Path, str]], list[dict]]:
    targets = dataset_targets()
    successes = {kind: 0 for kind in targets}
    attempts = {kind: 0 for kind in targets}
    candidate_dir = DATASET_DIR / "candidates"
    kept: list[tuple[Path, str]] = []
    summaries: list[dict] = []
    next_candidate_idx = 0
    next_episode_idx = 0

    while any(successes[kind] < target for kind, target in targets.items()):
        jobs: list[dict] = []
        for kind, target in targets.items():
            missing = target - successes[kind]
            if missing <= 0:
                continue
            max_attempts = target * MAX_ATTEMPTS_PER_SUCCESS
            remaining_attempts = max_attempts - attempts[kind]
            if remaining_attempts <= 0:
                raise RuntimeError(
                    f"Could not collect enough successful {kind} episodes: "
                    f"{successes[kind]}/{target} succeeded after {attempts[kind]} attempts."
                )
            for _ in range(min(missing, remaining_attempts)):
                jobs.append(
                    make_attempt_job(
                        kind=kind,
                        attempt=attempts[kind],
                        idx=next_candidate_idx,
                        have=successes[kind],
                        target=target,
                    )
                )
                attempts[kind] += 1
                next_candidate_idx += 1

        for candidate_path, kind in collect_dataset_episodes(jobs, candidate_dir):
            summary = summarize_episode(candidate_path, kind)
            summary["attempt_file"] = candidate_path.name
            skip_reason = episode_skip_reason(summary)
            if skip_reason is not None:
                summary["skipped"] = True
                summary["skip_reason"] = skip_reason
                candidate_path.unlink(missing_ok=True)
                print(
                    f"[data] skip {candidate_path.name} {kind}: {skip_reason} "
                    f"(ret={summary['return']:.1f}, "
                    f"tail_r={summary['tail_reward']:.2f}, "
                    f"tail_h={summary['tail_height']:.2f}, "
                    f"tail_up={summary['tail_upright']:.2f})"
                )
                summaries.append(summary)
                continue

            if successes[kind] >= targets[kind]:
                candidate_path.unlink(missing_ok=True)
                continue

            final_path = DATASET_DIR / f"episode_{next_episode_idx:03d}.npz"
            candidate_path.replace(final_path)
            successes[kind] += 1
            next_episode_idx += 1
            summary["file"] = final_path.name
            summary["skipped"] = False
            summaries.append(summary)
            kept.append((final_path, kind))
            print(
                f"[data] keep {final_path.name} {kind}: "
                f"{successes[kind]}/{targets[kind]} "
                f"(ret={summary['return']:.1f}, tail_r={summary['tail_reward']:.2f})"
            )

        progress = ", ".join(
            f"{kind}={successes[kind]}/{target}" for kind, target in targets.items()
        )
        print(f"[data] success quotas: {progress}")

    shutil.rmtree(candidate_dir, ignore_errors=True)
    return kept, summaries


def summarize_episode(path: Path, kind: str) -> dict:
    with np.load(path) as ep:
        rewards = ep["rewards"]
        tail_steps = min(QUALITY_TAIL_STEPS, len(rewards))
        tail = slice(-tail_steps, None) if tail_steps else slice(0, 0)
        heights = ep["torso_height"]
        uprights = ep["torso_upright"]
        velocities = ep["horizontal_velocity"]
        return {
            "file": path.name,
            "kind": kind,
            "steps": int(len(rewards)),
            "return": float(rewards.sum()) if len(rewards) else 0.0,
            "mean_height": float(np.nanmean(heights)),
            "mean_upright": float(np.nanmean(uprights)),
            "mean_velocity": float(np.nanmean(velocities)),
            "min_height": float(np.nanmin(heights)),
            "min_upright": float(np.nanmin(uprights)),
            "tail_reward": float(np.nanmean(rewards[tail])) if tail_steps else 0.0,
            "tail_height": float(np.nanmean(heights[tail])) if tail_steps else 0.0,
            "tail_upright": float(np.nanmean(uprights[tail])) if tail_steps else 0.0,
            "tail_velocity": float(np.nanmean(velocities[tail])) if tail_steps else 0.0,
            "max_sigma": float(np.max(ep["sigmas"])) if "sigmas" in ep.files else 0.0,
        }


def tail_recovered(summary: dict) -> bool:
    return (
        summary["tail_reward"] >= TAIL_MIN_REWARD
        and summary["tail_height"] >= TAIL_MIN_HEIGHT
        and summary["tail_upright"] >= TAIL_MIN_UPRIGHT
        and summary["tail_velocity"] >= TAIL_MIN_VELOCITY
    )


def has_real_off_manifold_prefix(summary: dict) -> bool:
    return (
        summary["min_height"] <= RECOVERY_MAX_MIN_HEIGHT
        or summary["min_upright"] <= RECOVERY_MAX_MIN_UPRIGHT
    )


def episode_skip_reason(summary: dict) -> str | None:
    kind = str(summary["kind"])
    if kind == "clean":
        if (
            summary["return"] < CLEAN_MIN_RETURN
            or summary["mean_height"] < CLEAN_MIN_MEAN_HEIGHT
            or not tail_recovered(summary)
        ):
            return "failed_clean_demo"
        return None

    if kind == "perturbed":
        if not tail_recovered(summary):
            return "failed_perturbed_recovery"
        return None

    if kind.startswith("recovery/"):
        if not tail_recovered(summary):
            return "failed_recovery_tail"
        if not has_real_off_manifold_prefix(summary):
            return "not_off_manifold_recovery"
        return None

    return None


def pick_best_clean_episode(paths: list[Path]) -> Path:
    best_path = paths[0]
    best_return = -float("inf")
    for path in paths:
        with np.load(path) as ep:
            ret = float(ep["rewards"].sum())
        if ret > best_return:
            best_return = ret
            best_path = path
    return best_path


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
    )


def train_config() -> lewm.LeWMTrainConfig:
    return lewm.LeWMTrainConfig(
        num_preds=NUM_PREDS,
        rollout_steps=TRAIN_ROLLOUT_STEPS,
        rollout_loss_weight=ROLLOUT_LOSS_WEIGHT,
        sigreg_weight=SIGREG_WEIGHT,
        sigreg_knots=SIGREG_KNOTS,
        sigreg_num_proj=SIGREG_NUM_PROJ,
        batch_size=TRAIN_BATCH_SIZE,
        train_steps=TRAIN_STEPS,
        num_workers=TRAIN_NUM_WORKERS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        log_every=LOG_EVERY,
    )


def walker_window_weight(ep: dict[str, np.ndarray], start: int, window_pixels: int) -> float:
    """Task-adapter weighting; src only sees the resulting windows."""
    end = min(start + window_pixels - 1, len(ep["pixels"]) - 1)
    weight = 1.0
    meta = metadata_from_array_dict(ep)
    if bool(meta.get("noisy", False)):
        weight *= 2.0
    if meta.get("kind") == "scripted_recovery":
        weight *= 4.0
    if "torso_height" in ep:
        metric_start = max(start - 1, 0)
        metric_end = max(end - 1, metric_start + 1)
        h = ep["torso_height"][metric_start:metric_end]
        if len(h) and float(np.nanmin(h)) < RECOVERY_MAX_MIN_HEIGHT:
            weight *= 3.0
    if "torso_upright" in ep:
        metric_start = max(start - 1, 0)
        metric_end = max(end - 1, metric_start + 1)
        up = ep["torso_upright"][metric_start:metric_end]
        if len(up) and float(np.nanmin(up)) < RECOVERY_MAX_MIN_UPRIGHT:
            weight *= 2.0
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


def metadata_json(ep) -> dict:
    if "metadata" not in ep.files:
        return {}
    raw = ep["metadata"]
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def metadata_from_array_dict(ep: dict[str, np.ndarray]) -> dict:
    if "metadata" not in ep:
        return {}
    raw = ep["metadata"]
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def stable_frame_mask(ep, num_frames: int) -> np.ndarray:
    action_steps = max(num_frames - 1, 1)
    frame_ids = np.arange(num_frames)
    metric_ids = np.clip(frame_ids - 1, 0, action_steps - 1)
    mask = frame_ids >= min(GOAL_ORBIT_START, num_frames - 1)
    for key, threshold in (
        ("rewards", GOAL_MANIFOLD_MIN_REWARD),
        ("torso_height", GOAL_MANIFOLD_MIN_HEIGHT),
        ("torso_upright", GOAL_MANIFOLD_MIN_UPRIGHT),
        ("horizontal_velocity", GOAL_MANIFOLD_MIN_VELOCITY),
    ):
        if key in ep.files:
            values = ep[key]
            ids = np.clip(metric_ids, 0, len(values) - 1)
            mask = mask & (values[ids] >= threshold)
    return mask


def is_high_quality_clean_episode(path: Path) -> bool:
    with np.load(path) as ep:
        meta = metadata_json(ep)
        if meta.get("kind") is not None:
            return False
        if bool(meta.get("noisy", False)):
            return False
        if len(ep["pixels"]) < GOAL_ORBIT_START + PLAN_HORIZON + 2:
            return False
        if "rewards" not in ep.files or "torso_height" not in ep.files:
            return False
        tail = slice(max(0, len(ep["rewards"]) - QUALITY_TAIL_STEPS), None)
        return (
            float(ep["rewards"][tail].mean()) >= TAIL_MIN_REWARD
            and float(ep["torso_height"][tail].mean()) >= TAIL_MIN_HEIGHT
            and float(ep["torso_upright"][tail].mean()) >= TAIL_MIN_UPRIGHT
            and float(ep["horizontal_velocity"][tail].mean()) >= TAIL_MIN_VELOCITY
        )


def support_transition_mask(ep, num_frames: int) -> np.ndarray:
    """Task adapter: expose successful transition segments to src manifold code."""
    stable = stable_frame_mask(ep, num_frames)
    if stable.sum() < SUPPORT_MIN_MASK_FRAMES:
        return np.zeros(num_frames, dtype=bool)
    tail_n = min(SUPPORT_TAIL_STABLE_STEPS, num_frames)
    if not bool(stable[-tail_n:].all()):
        return np.zeros(num_frames, dtype=bool)

    meta = metadata_json(ep)
    action_steps = max(num_frames - 1, 1)
    has_off_manifold = bool(meta.get("kind") == "scripted_recovery" or meta.get("noisy", False))
    if "torso_height" in ep.files:
        has_off_manifold = has_off_manifold or float(np.nanmin(ep["torso_height"][:action_steps])) < RECOVERY_MAX_MIN_HEIGHT
    if "torso_upright" in ep.files:
        has_off_manifold = has_off_manifold or float(np.nanmin(ep["torso_upright"][:action_steps])) < RECOVERY_MAX_MIN_UPRIGHT
    if not has_off_manifold:
        return np.zeros(num_frames, dtype=bool)

    first_stable = int(np.argmax(stable))
    if meta.get("kind") == "scripted_recovery":
        start = 0
    else:
        start = max(min(GOAL_ORBIT_START, num_frames - 1), first_stable - PLAN_HORIZON)
    mask = np.zeros(num_frames, dtype=bool)
    mask[start:] = True
    return mask


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
    goal_ep,
    device: torch.device,
) -> LatentManifold:
    segments: list[SuccessSegment] = []
    goal_meta = metadata_json(goal_ep)
    skipped_source = str(goal_meta.get("source_episode", ""))

    def add_source(name: str, ep, pixels: np.ndarray, role: str, mask: np.ndarray) -> None:
        if int(mask.sum()) < SUPPORT_MIN_MASK_FRAMES:
            return
        segments.append(
            SuccessSegment(
                name=name,
                pixels=pixels.copy(),
                mask=mask.astype(bool),
                role=role,
                metadata=metadata_json(ep),
            )
        )

    add_source("goal_trajectory", goal_ep, goal_pixels, "goal", stable_frame_mask(goal_ep, len(goal_pixels)))
    for path in sorted(DATASET_DIR.glob("episode_*.npz")):
        if path.name == skipped_source:
            continue
        with np.load(path) as ep:
            pixels = ep["pixels"]
            if is_high_quality_clean_episode(path):
                add_source(path.name, ep, pixels, "goal", stable_frame_mask(ep, len(pixels)))
            support_mask = support_transition_mask(ep, len(pixels))
            add_source(path.name, ep, pixels, "support", support_mask)

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
    with np.load(DATASET_DIR / "goal_trajectory.npz") as goal:
        goal_pixels = goal["pixels"].copy()
        goal_actions_np = goal["actions"].astype(np.float32)
        goal_manifold = build_goal_manifold(model, goal_pixels, goal, device)

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
