"""Clean reference-style LeWM pipeline for dm_control walker/walk.

Run:
    uv run python walker_walk.py

The script is intentionally split into reusable stages:
1. Build a static dataset under DATASET_DIR if it does not already exist.
2. Train LeWM from that dataset with one-step prediction + SIGReg.
3. Evaluate pure LeWM-CEM against the clean goal trajectory.

No MuJoCo oracle, reward, or intervention is used during training or eval.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from queue import Empty
from typing import Iterable, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from dm_control import suite
from tqdm import tqdm

from model import SIGReg, build_lewm


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

SIGREG_WEIGHT = 0.09
SIGREG_KNOTS = 17
SIGREG_NUM_PROJ = 1024
TRAIN_BATCH_SIZE = 64
TRAIN_STEPS = 7000
TRAIN_NUM_WORKERS = 2
LR = 5e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
LOG_EVERY = 50
TRAIN_ROLLOUT_STEPS = 6
ROLLOUT_LOSS_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Pure LeWM-CEM eval.
# ---------------------------------------------------------------------------


EVAL_STEPS = 400
EVAL_SEED = SEED
EVAL_BOOTSTRAP_STEPS = HISTORY_SIZE - 1
ACTION_BLOCK = 1
PLAN_BLOCKS = 12
PLAN_HORIZON = ACTION_BLOCK * PLAN_BLOCKS
EVAL_CEM_SAMPLES = 128
EVAL_CEM_TOPK = 16
EVAL_CEM_ITERS = 3
EVAL_CEM_INIT_STD = 0.002
EVAL_CEM_MIN_STD = 0.0005
EVAL_CEM_MOMENTUM = 0.15

GOAL_ORBIT_START = 40
PHASE_SEARCH_WINDOW = 18
PHASE_MIN_STEP = 1
PHASE_MAX_STEP = 4
TRAJ_DISCOUNT = 0.92
LATENT_VELOCITY_WEIGHT = 0.5
ACTION_PRIOR_WEIGHT = 1000.0
ACTION_SMOOTHNESS_WEIGHT = 0.05
ACTION_ENERGY_WEIGHT = 0.002
FAR_CEM_EXTRA_STD = 0.05
ORBIT_BLEND_SIGMA_MULT = 8.0
ORBIT_BLEND_SIGMA_MIN = 0.025
ORBIT_BLEND_SIGMA_MAX = 0.35
PHASE_RELOCK_RHO = -1.0
RETURN_TERMINAL_WEIGHT = 0.75


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
    pixels = pixels.float() / 255.0
    pixels = pixels.permute(0, 1, 4, 2, 3).contiguous()
    if pixels.size(-1) != IMAGE_SIZE or pixels.size(-2) != IMAGE_SIZE:
        b, t = pixels.shape[:2]
        pixels = pixels.view(b * t, *pixels.shape[2:])
        pixels = F.interpolate(
            pixels,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        pixels = pixels.view(b, t, *pixels.shape[1:])
    return (pixels - _IMAGENET_MEAN.to(pixels)) / _IMAGENET_STD.to(pixels)


class WalkerWindowDataset(torch.utils.data.Dataset):
    """Contiguous windows where action[t] predicts pixels[t+1]."""

    def __init__(self, dataset_dir: Path):
        self.window_pixels = HISTORY_SIZE + max(NUM_PREDS, TRAIN_ROLLOUT_STEPS)
        self.window_actions = self.window_pixels - 1
        self.episodes: list[dict[str, np.ndarray]] = []
        for path in sorted(dataset_dir.glob("episode_*.npz")):
            with np.load(path) as ep:
                pixels = ep["pixels"]
                actions = ep["actions"].astype(np.float32)
            if len(pixels) >= self.window_pixels and len(actions) >= self.window_actions:
                self.episodes.append({"pixels": pixels, "actions": actions})

        self.index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            max_start = min(
                len(ep["pixels"]) - self.window_pixels,
                len(ep["actions"]) - self.window_actions,
            )
            for start in range(max_start + 1):
                self.index.append((ep_idx, start))

        if not self.index:
            raise RuntimeError(f"No valid training windows in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        pixels = ep["pixels"][start : start + self.window_pixels]
        actions = ep["actions"][start : start + self.window_actions]
        return torch.from_numpy(pixels), torch.from_numpy(actions)


def new_model(action_dim: int, device: torch.device):
    return build_lewm(
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
    ).to(device)


def train_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def lewm_one_step_loss(
    model,
    sigreg: SIGReg,
    pixels: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    actions = torch.nan_to_num(actions, 0.0)
    info = model.encode({"pixels": pixels, "action": actions[:, :HISTORY_SIZE]})
    emb = info["emb"]
    act_emb = info["act_emb"]

    ctx_emb = emb[:, :HISTORY_SIZE]
    target_emb = emb[:, 1 : HISTORY_SIZE + 1]
    pred_emb = model.predict(ctx_emb, act_emb[:, :HISTORY_SIZE])
    pred_loss = (pred_emb - target_emb).pow(2).mean()

    rollout_steps = min(TRAIN_ROLLOUT_STEPS, pixels.size(1) - HISTORY_SIZE)
    hist_actions = torch.zeros_like(actions[:, :HISTORY_SIZE])
    if HISTORY_SIZE > 1:
        hist_actions[:, 1:] = actions[:, : HISTORY_SIZE - 1]
    future_actions = actions[:, HISTORY_SIZE - 1 : HISTORY_SIZE - 1 + rollout_steps]
    rollout_pred = aligned_rollout_latents(
        model,
        pixels[:, :HISTORY_SIZE],
        hist_actions,
        future_actions.unsqueeze(1),
    ).squeeze(1)
    rollout_target = emb[:, HISTORY_SIZE : HISTORY_SIZE + rollout_steps]
    rollout_loss = (rollout_pred - rollout_target).pow(2).mean()

    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + ROLLOUT_LOSS_WEIGHT * rollout_loss + SIGREG_WEIGHT * sigreg_loss
    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
    }


def train_lewm() -> tuple[Path, int]:
    dataset = WalkerWindowDataset(DATASET_DIR)
    action_dim = int(dataset.episodes[0]["actions"].shape[1])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=TRAIN_NUM_WORKERS,
        persistent_workers=TRAIN_NUM_WORKERS > 0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = new_model(action_dim, device)
    sigreg = SIGReg(knots=SIGREG_KNOTS, num_proj=SIGREG_NUM_PROJ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train] device={device} params={n_params / 1e6:.2f}M "
        f"windows={len(dataset)} episodes={len(dataset.episodes)}"
    )
    if TRAIN_LOG_PATH.exists():
        TRAIN_LOG_PATH.unlink()
    append_jsonl(
        TRAIN_LOG_PATH,
        {
            "event": "start",
            "device": str(device),
            "params": n_params,
            "windows": len(dataset),
            "episodes": len(dataset.episodes),
            "config": hparams(),
        },
    )

    model.train()
    step = 0
    last_log: dict[str, float] = {}
    pbar = tqdm(total=TRAIN_STEPS, desc="train LeWM", dynamic_ncols=True)
    while step < TRAIN_STEPS:
        for pixels_u8, actions in loader:
            if step >= TRAIN_STEPS:
                break
            pixels_u8 = pixels_u8.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            pixels = preprocess_pixels(pixels_u8)

            with train_autocast(device):
                losses = lewm_one_step_loss(model, sigreg, pixels, actions)

            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            step += 1
            pbar.update(1)
            if step % LOG_EVERY == 0 or step == 1:
                last_log = {
                    "event": "step",
                    "step": step,
                    "loss": float(losses["loss"].detach()),
                    "pred_loss": float(losses["pred_loss"]),
                    "rollout_loss": float(losses["rollout_loss"]),
                    "sigreg_loss": float(losses["sigreg_loss"]),
                    "lr": float(optim.param_groups[0]["lr"]),
                }
                append_jsonl(TRAIN_LOG_PATH, last_log)
                pbar.set_postfix(
                    {
                        "loss": f"{float(losses['loss'].detach()):.4f}",
                        "pred": f"{float(losses['pred_loss']):.4f}",
                        "roll": f"{float(losses['rollout_loss']):.4f}",
                        "sig": f"{float(losses['sigreg_loss']):.4f}",
                    }
                )
    pbar.close()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "lewm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "action_dim": action_dim,
            "config": hparams(),
        },
        ckpt_path,
    )
    print(f"[train] saved -> {ckpt_path}")
    append_jsonl(
        TRAIN_LOG_PATH,
        {
            "event": "done",
            "step": step,
            "checkpoint": str(ckpt_path),
            "last_log": last_log,
        },
    )
    return ckpt_path, action_dim


def load_lewm(ckpt_path: Path, action_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model(action_dim, device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, device


# ---------------------------------------------------------------------------
# Goal encoding and aligned LeWM rollout.
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_frames(model, pixels: np.ndarray, device: torch.device, chunk: int = 128) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(pixels), chunk):
        batch = torch.from_numpy(pixels[start : start + chunk]).unsqueeze(0).to(device)
        emb = model.encode({"pixels": preprocess_pixels(batch)})["emb"].squeeze(0)
        chunks.append(emb)
    return torch.cat(chunks, dim=0)


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
    dist = (goal_latents.float() - current_latent.view(1, -1).float()).pow(2).mean(dim=-1)
    idx = int(torch.argmin(dist).item())
    return idx, float(dist[idx].item())


def orbit_rho(distance: float, sigma: float) -> float:
    return float(np.exp(-distance / max(sigma, 1e-6)))


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
    max_horizon = min(PLAN_HORIZON, len(goal_actions_np) - HISTORY_SIZE)
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
        out[f"true_history_{start}"] = {
            "first": float(mse[0]),
            "terminal": float(mse[-1]),
            "mean": float(mse.mean()),
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
    }
    return out


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
    """Roll out one-step LeWM dynamics with correct action alignment.

    history_actions is shifted so action_buf[:, -H+1:] are the known actions
    between context frames. The current-frame action is the candidate
    future_actions[:, :, 0], so it affects the first prediction.
    """
    b, h = history_pixels.shape[:2]
    samples, horizon = future_actions.shape[1:3]
    hist = model.encode({"pixels": history_pixels})
    emb = hist["emb"].unsqueeze(1).expand(b, samples, h, -1).reshape(b * samples, h, -1)
    action_buf = (
        history_actions.unsqueeze(1)
        .expand(b, samples, h, -1)
        .reshape(b * samples, h, -1)
        .float()
    )
    future = future_actions.reshape(b * samples, horizon, -1).float()

    preds = []
    for t in range(horizon):
        emb_in = emb[:, -HISTORY_SIZE:]
        next_action = future[:, t : t + 1]
        if HISTORY_SIZE == 1:
            raw_actions = next_action
        else:
            raw_actions = torch.cat([action_buf[:, -HISTORY_SIZE + 1 :], next_action], dim=1)
        act_emb = model.action_encoder(raw_actions)
        pred = model.predict(emb_in, act_emb)[:, -1:]
        preds.append(pred)
        emb = torch.cat([emb, pred], dim=1)
        action_buf = torch.cat([action_buf, next_action], dim=1)

    pred = torch.cat(preds, dim=1)
    return pred.reshape(b, samples, horizon, -1)


# ---------------------------------------------------------------------------
# Pure LeWM-CEM evaluation.
# ---------------------------------------------------------------------------


@torch.no_grad()
def lewm_cem_plan(
    model,
    history_pixels_u8: torch.Tensor,
    history_actions: torch.Tensor,
    target_latents: torch.Tensor,
    return_latents: torch.Tensor,
    prior_actions: torch.Tensor,
    rho: float,
    previous_action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    weights: torch.Tensor,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    action_dim = low.numel()
    prior_blocks = block_actions_from_step_actions(prior_actions).to(device)
    rho_t = torch.tensor(float(rho), device=device, dtype=torch.float32).clamp(0.0, 1.0)
    mean = torch.clamp(prior_blocks, low, high)
    std_scale = EVAL_CEM_INIT_STD + (1.0 - float(rho_t)) * FAR_CEM_EXTRA_STD
    std = torch.full_like(mean, std_scale)

    history_pixels = preprocess_pixels(history_pixels_u8.to(device))
    history_actions = history_actions.to(device)
    target_latents = target_latents.to(device)
    return_latents = return_latents.to(device).float()

    best_plan: torch.Tensor | None = None
    best_cost = float("inf")
    for _ in range(EVAL_CEM_ITERS):
        eps = torch.randn((EVAL_CEM_SAMPLES, PLAN_BLOCKS, action_dim), generator=rng, device=device)
        block_samples = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * eps, low, high)
        block_samples[0] = prior_blocks
        block_samples[1] = mean
        step_samples = block_samples.repeat_interleave(ACTION_BLOCK, dim=1)

        pred = aligned_rollout_latents(
            model,
            history_pixels,
            history_actions,
            step_samples.unsqueeze(0),
        ).squeeze(0)

        goal = target_latents.unsqueeze(0).expand_as(pred)
        diff = (pred.float() - goal.float()).pow(2).mean(dim=-1)
        phase_cost = (diff * weights).mean(dim=1)
        if LATENT_VELOCITY_WEIGHT > 0 and PLAN_HORIZON > 1:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            goal_vel = goal[:, 1:] - goal[:, :-1]
            vel_diff = (pred_vel.float() - goal_vel.float()).pow(2).mean(dim=-1)
            phase_cost = phase_cost + LATENT_VELOCITY_WEIGHT * (vel_diff * weights[1:]).mean(dim=1)

        samples, horizon, dim = pred.shape
        return_dist = torch.cdist(
            pred.reshape(samples * horizon, dim).float(),
            return_latents,
        ).pow(2).min(dim=1).values.div(dim).reshape(samples, horizon)
        return_cost = (return_dist * weights).mean(dim=1)
        return_cost = return_cost + RETURN_TERMINAL_WEIGHT * return_dist[:, -1]
        cost = rho_t * phase_cost + (1.0 - rho_t) * return_cost

        prior_cost = (block_samples - prior_blocks.unsqueeze(0)).pow(2).mean(dim=(1, 2))
        prev = torch.cat(
            [previous_action.view(1, 1, -1).expand(EVAL_CEM_SAMPLES, 1, -1), block_samples[:, :-1]],
            dim=1,
        )
        smooth_cost = (block_samples - prev).pow(2).mean(dim=(1, 2))
        energy_cost = block_samples.pow(2).mean(dim=(1, 2))
        cost = (
            cost
            + (rho_t * ACTION_PRIOR_WEIGHT) * prior_cost
            + ACTION_SMOOTHNESS_WEIGHT * smooth_cost
            + ACTION_ENERGY_WEIGHT * energy_cost
        )

        topk = min(EVAL_CEM_TOPK, EVAL_CEM_SAMPLES)
        elite_idx = torch.topk(cost, topk, largest=False).indices
        elites = block_samples[elite_idx]
        elite_mean = elites.mean(dim=0)
        elite_std = elites.std(dim=0, unbiased=False).clamp_min(EVAL_CEM_MIN_STD)
        mean = (1.0 - EVAL_CEM_MOMENTUM) * elite_mean + EVAL_CEM_MOMENTUM * mean
        std = elite_std

        i_best = int(torch.argmin(cost).item())
        if float(cost[i_best]) < best_cost:
            best_cost = float(cost[i_best])
            best_plan = block_samples[i_best].clone()

    assert best_plan is not None
    return best_plan


@torch.no_grad()
def evaluate_lewm_cem(ckpt_path: Path, action_dim: int) -> dict:
    model, device = load_lewm(ckpt_path, action_dim)
    with np.load(DATASET_DIR / "goal_trajectory.npz") as goal:
        goal_pixels = goal["pixels"]
        goal_actions_np = goal["actions"].astype(np.float32)

    goal_latents = encode_frames(model, goal_pixels, device)
    goal_actions = torch.from_numpy(goal_actions_np).to(device)
    return_start = min(GOAL_ORBIT_START, goal_latents.size(0) - 1)
    return_latents = goal_latents[return_start:]
    orbit_sigma = estimate_orbit_sigma(goal_latents)
    imageio.mimsave(OUT_DIR / "goal_trajectory.mp4", list(goal_pixels), fps=VIDEO_FPS)
    diagnostics_payload = {
        "rollout_prediction": rollout_prediction_diagnostics(
            model, goal_pixels, goal_actions_np, goal_latents, device
        ),
        "orbit_sigma": float(orbit_sigma),
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

    phase = 0
    env_steps = 0
    first_step = True
    bar = tqdm(total=EVAL_STEPS, desc="eval LeWM-CEM", dynamic_ncols=True)
    bootstrap_steps = min(EVAL_BOOTSTRAP_STEPS, EVAL_STEPS, len(goal_actions_np))
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
        if first_step or rho < PHASE_RELOCK_RHO:
            phase = global_phase
        else:
            phase = phase_match(current_latent, goal_latents, phase, first_step)
        first_step = False
        target_latents, prior_actions = goal_window(goal_latents, goal_actions, phase)

        hist_pix = torch.from_numpy(np.stack(history_frames[-HISTORY_SIZE:], axis=0)).unsqueeze(0)
        hist_act = torch.from_numpy(np.stack(history_actions[-HISTORY_SIZE:], axis=0)).unsqueeze(0)
        plan = lewm_cem_plan(
            model,
            hist_pix,
            hist_act,
            target_latents,
            return_latents,
            prior_actions,
            rho,
            previous_action,
            low,
            high,
            weights,
            rng,
            device,
        )

        action = plan[0].detach().cpu().numpy().astype(np.float32)
        for _ in range(ACTION_BLOCK):
            if env_steps >= EVAL_STEPS:
                break
            video_frames.append(render_video_frame(env))
            ts = env.step(action)
            diag = diagnostics(env)
            rewards.append(float(ts.reward or 0.0))
            velocities.append(diag["horizontal_velocity"])
            heights.append(diag["torso_height"])
            uprights.append(diag["torso_upright"])
            actions_out.append(action.copy())
            phases.append(phase)
            rhos.append(rho)
            orbit_distances.append(current_dist)
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
                    "distance": float(current_dist),
                    "phase": int(phase),
                    "global_phase": int(global_phase),
                    "action_abs": float(np.mean(np.abs(action))),
                    "previous_action_abs": float(previous_action.abs().mean().detach().cpu()),
                },
            )
            bar.set_postfix(
                {
                    "ret": f"{sum(rewards):.1f}",
                    "v": f"{diag['horizontal_velocity']:.2f}",
                    "h": f"{diag['torso_height']:.2f}",
                    "rho": f"{rho:.2f}",
                    "phase": phase,
                }
            )
            if ts.last():
                env_steps = EVAL_STEPS
                break
    bar.close()
    env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / f"eval_cam{VIDEO_CAMERA_ID}.mp4"
    imageio.mimsave(video_path, video_frames, fps=VIDEO_FPS)

    metrics = {
        "domain": DOMAIN,
        "task": TASK,
        "policy": "pure_lewm_cem_goal_trajectory",
        "steps": len(rewards),
        "total_return": float(np.sum(rewards)) if rewards else 0.0,
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
        f"mean_v={metrics['mean_horizontal_velocity']:.2f} -> {video_path}"
    )
    return metrics


def main() -> None:
    if NUM_PREDS != 1:
        raise ValueError("walker_walk.py currently implements one-step LeWM only; keep NUM_PREDS=1.")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[cfg]", json.dumps(hparams(), indent=2))

    build_static_dataset()
    ckpt_path, action_dim = train_lewm()
    evaluate_lewm_cem(ckpt_path, action_dim)


if __name__ == "__main__":
    main()
