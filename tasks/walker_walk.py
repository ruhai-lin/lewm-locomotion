"""LeWM-CEM pipeline for dm_control walker/walk.

One-shot entrypoint:

    uv run python tasks/walker_walk.py

This adapter contributes only the *task-specific* parts (env creation,
rendering, per-step diagnostics, frame-quality thresholds, oracle reward
shaping). Dataset collection, oracle-CEM goal demo, LeWM training, and
CEM evaluation all live in ``src/``.

Pipeline:
1. ``src.datasets.collect_dataset`` writes ``DATASET_N_EPISODES`` explore
   episodes under ``DATASET_DIR``.
2. ``src.datasets.collect_oracle_goal_demo`` adds 1 oracle-CEM demo
   that starts from the env reset pose and walks forward — used purely
   as the planner's goal trajectory.
3. ``src.lewm.train_lewm`` trains a LeWM-pure model (adaln_id init).
4. ``src.eval.evaluate_lewm_cem`` runs the CEM eval and writes
   ``metrics.json`` + ``eval_cam.mp4``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from dm_control import suite

from src import datasets, eval as eval_module, lewm
from src.planner import PlannerWeights


# ---------------------------------------------------------------------------
# Identity and paths.
# ---------------------------------------------------------------------------


DOMAIN = "walker"
TASK = "walk"
SEED = 42

DATASET_DIR = Path("datasets/walker_walk")
OUT_DIR = Path("outputs/walker_walk")
CKPT_DIR = OUT_DIR / "ckpt"
TRAIN_LOG_PATH = OUT_DIR / "train_log.jsonl"
DATASET_VERSION = 7  # 128 explore + 1 oracle goal demo (E7).


# ---------------------------------------------------------------------------
# Rendering / dataset collection.
# ---------------------------------------------------------------------------


IMAGE_SIZE = 112
DATASET_CAMERA_ID = 0
VIDEO_CAMERA_ID = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 40
DT = 0.025  # dm_control walker step length (40 Hz).

DATASET_N_EPISODES = 128
DATASET_EPISODE_STEPS = 400
DATASET_WORKERS = min(8, os.cpu_count() or 1)
DATASET_MP_START_METHOD = "forkserver"


# ---------------------------------------------------------------------------
# Goal- and fall-frame thresholds (drive the post-hoc goal manifold).
# ---------------------------------------------------------------------------


GOAL_FRAME_MIN_REWARD = 0.0
GOAL_FRAME_MIN_HEIGHT = 0.9
GOAL_FRAME_MIN_UPRIGHT = 0.5
GOAL_FRAME_MIN_VELOCITY = 0.1
GOAL_MIN_SEGMENT_LEN = 4

SUPPORT_FRAME_MAX_HEIGHT = 0.7
SUPPORT_FRAME_MAX_UPRIGHT = 0.3


# ---------------------------------------------------------------------------
# Oracle CEM reward shaping (walker: forward speed + upright posture).
# ---------------------------------------------------------------------------


ORACLE_TARGET_SPEED = 1.2
ORACLE_PROGRESS_BONUS = 0.75
ORACLE_SPEED_TRACKING_BONUS = 0.12
ORACLE_OVERSPEED_PENALTY = 0.35
ORACLE_POSTURE_PENALTY = 1.2
ORACLE_ACTION_PENALTY = 0.01
ORACLE_ACTION_SMOOTHNESS_PENALTY = 0.015
ORACLE_REWARD_GAMMA = 0.98


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
PREDICTOR_CONDITIONING = "adaln_id"  # E4d: nonzero action conditioning at init.

SIGREG_WEIGHT = 0.07
SIGREG_KNOTS = 17
SIGREG_NUM_PROJ = 1024
TRAIN_BATCH_SIZE = 64
TRAIN_STEPS = 10000
TRAIN_NUM_WORKERS = 0
LR = 5e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
LOG_EVERY = 50
TRAIN_ROLLOUT_STEPS = 16
ROLLOUT_LOSS_WEIGHT = 1.0
ACTION_CONTRAST_WEIGHT = 0.0  # disabled; ``adaln_id`` alone is enough (E4d-v2).
ACTION_CONTRAST_MARGIN = 0.0
ACTION_CONTRAST_HORIZON = 0
TRAIN_WINDOW_WEIGHTING = False  # E6 follow-up: weighting backfired.


# ---------------------------------------------------------------------------
# Eval / CEM planner.
# ---------------------------------------------------------------------------


EVAL_STEPS = 400
EVAL_BOOTSTRAP_STEPS = HISTORY_SIZE - 1
ACTION_BLOCK = 1
PLAN_BLOCKS = 16
PLAN_HORIZON = ACTION_BLOCK * PLAN_BLOCKS
EVAL_CEM_SAMPLES = 256
EVAL_CEM_TOPK = 32
EVAL_CEM_ITERS = 4
EVAL_CEM_INIT_STD = 0.01
EVAL_CEM_MAX_STD = 0.65
EVAL_CEM_MIN_STD = 0.002
EVAL_CEM_MOMENTUM = 0.15
CEM_WARM_START_BLEND = 0.0
CEM_PRIOR_TIE_REL = 0.0
CEM_PRIOR_TIE_ABS = 0.0  # E7: guard off, CEM elite wins on cost alone.
CEM_PRIOR_IN_MEAN = True
CEM_PRIOR_IN_SAMPLES = True
CEM_STATE_COST_CLIP = 0.0
CEM_DELTA_COST_CLIP = 0.0
CEM_NEAR_COST_CLIP = 0.0

TRAJ_DISCOUNT = 0.92
NEAR_DELTA_WEIGHT = 0.5
W_STATE = 0.2
W_DELTA = 4.0
W_SUPPORT_STATE = 0.0
W_SUPPORT_DELTA = 0.0
W_NEAR = 0.01
W_ACTION_PRIOR = 0.1
W_SMOOTH = 0.1
W_ENERGY = 0.002

SUPPORT_MIN_MASK_FRAMES = PLAN_HORIZON + 1
MANIFOLD_MAX_STATE_POINTS = 4096
MANIFOLD_MAX_DELTA_POINTS = 4096
MANIFOLD_MAX_SEGMENTS = 4096
MANIFOLD_COST_SCALE_MIN = 1e-4


# ---------------------------------------------------------------------------
# Helpers (this is all the task-specific code there is).
# ---------------------------------------------------------------------------


def hparams() -> dict:
    simple = (str, int, float, bool, type(None))
    out: dict = {}
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


def make_env(seed: int):
    return suite.load(
        domain_name=DOMAIN,
        task_name=TASK,
        task_kwargs={"random": seed},
        visualize_reward=False,
    )


def action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    spec = env.action_spec()
    return (
        np.asarray(spec.minimum, dtype=np.float32),
        np.asarray(spec.maximum, dtype=np.float32),
    )


def render_dataset_frame(env) -> np.ndarray:
    return env.physics.render(
        height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=DATASET_CAMERA_ID,
    ).astype(np.uint8)


def render_video_frame(env) -> np.ndarray:
    return env.physics.render(
        height=VIDEO_HEIGHT, width=VIDEO_WIDTH, camera_id=VIDEO_CAMERA_ID,
    ).astype(np.uint8)


def diagnostics(env) -> dict[str, float]:
    """Per-step task metrics — these field names show up verbatim in the
    eval trace + metrics.json, so they are the contract between this task
    and any dashboard / analysis script."""
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
    for r in rewards:
        total += discount * float(r)
        discount *= gamma
    return float(total)


# ---------------------------------------------------------------------------
# Frame masks for the post-hoc goal manifold.
# ---------------------------------------------------------------------------


def goal_frame_mask(ep: dict[str, np.ndarray]) -> np.ndarray:
    """True wherever the walker looks like it is walking."""
    num_frames = len(ep["pixels"])
    if num_frames == 0:
        return np.zeros(0, dtype=bool)
    action_steps = max(num_frames - 1, 1)
    metric_ids = np.clip(np.arange(num_frames) - 1, 0, action_steps - 1)
    mask = np.ones(num_frames, dtype=bool)
    for key, threshold in (
        ("rewards", GOAL_FRAME_MIN_REWARD),
        ("torso_height", GOAL_FRAME_MIN_HEIGHT),
        ("torso_upright", GOAL_FRAME_MIN_UPRIGHT),
        ("horizontal_velocity", GOAL_FRAME_MIN_VELOCITY),
    ):
        if key in ep:
            arr = np.asarray(ep[key])
            mask = mask & (arr[np.clip(metric_ids, 0, len(arr) - 1)] >= threshold)
    return mask


def fall_frame_mask(ep: dict[str, np.ndarray]) -> np.ndarray:
    """True wherever the walker looks like it is on the ground."""
    num_frames = len(ep["pixels"])
    if num_frames == 0:
        return np.zeros(0, dtype=bool)
    action_steps = max(num_frames - 1, 1)
    metric_ids = np.clip(np.arange(num_frames) - 1, 0, action_steps - 1)
    out = np.zeros(num_frames, dtype=bool)
    if "torso_height" in ep:
        h = np.asarray(ep["torso_height"])
        out = out | (h[np.clip(metric_ids, 0, len(h) - 1)] <= SUPPORT_FRAME_MAX_HEIGHT)
    if "torso_upright" in ep:
        u = np.asarray(ep["torso_upright"])
        out = out | (u[np.clip(metric_ids, 0, len(u) - 1)] <= SUPPORT_FRAME_MAX_UPRIGHT)
    return out


# ---------------------------------------------------------------------------
# Oracle CEM reward shaping for the +1 goal demo. Task-specific.
# ---------------------------------------------------------------------------


def oracle_score_fn(
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
    smooth_cost = speed_tracking = overspeed_cost = posture_cost = 0.0
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


def walker_window_weight(ep: dict[str, np.ndarray], start: int, window_pixels: int) -> float:
    """Optional training sampler bias toward "upright + moving" windows.
    Disabled by default (E6: weighting hurt action Jacobian)."""
    end = min(start + window_pixels - 1, len(ep["pixels"]) - 1)
    metric_start = max(start - 1, 0)
    metric_end = max(end - 1, metric_start + 1)
    weight = 1.0
    if "torso_height" in ep and "torso_upright" in ep:
        h = np.asarray(ep["torso_height"][metric_start:metric_end], dtype=np.float32)
        u = np.asarray(ep["torso_upright"][metric_start:metric_end], dtype=np.float32)
        if h.size and u.size:
            weight += 6.0 * float(np.clip(h.mean(), 0.0, 1.4) * np.clip(np.abs(u).mean(), 0.0, 1.0))
    if "rewards" in ep:
        r = np.asarray(ep["rewards"][metric_start:metric_end], dtype=np.float32)
        if r.size:
            weight += 4.0 * float(np.clip(r.mean(), 0.0, 1.0))
    return weight


# ---------------------------------------------------------------------------
# Register task handles with src.datasets so explore-episode workers
# (forkserver, no closures) can re-import this module and find them.
# ---------------------------------------------------------------------------


datasets.register_task(
    "walker_walk",
    datasets.TaskHandles(
        env_fn=make_env,
        render_fn=render_dataset_frame,
        action_bounds_fn=action_bounds,
        record_fn=diagnostics,
        dt=DT,
    ),
)


# ---------------------------------------------------------------------------
# Pipeline stages.
# ---------------------------------------------------------------------------


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


def build_static_dataset() -> None:
    """Collect 128 explore episodes + 1 oracle-CEM goal demo."""
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

    print("[data] collecting 1 oracle-CEM goal demo...")
    oracle_demo = datasets.collect_oracle_goal_demo(
        env_fn=make_env,
        action_bounds_fn=action_bounds,
        render_fn=render_dataset_frame,
        record_fn=diagnostics,
        score_fn=oracle_score_fn,
        episode_steps=DATASET_EPISODE_STEPS,
        seed=SEED,
        domain=DOMAIN,
        task=TASK,
    )
    oracle_path = DATASET_DIR / f"episode_{DATASET_N_EPISODES:04d}.npz"
    np.savez_compressed(oracle_path, **oracle_demo)
    print(
        f"[data] oracle demo: return={float(oracle_demo['rewards'].sum()):.1f} "
        f"-> {oracle_path.name}"
    )

    # The oracle demo IS the goal trajectory (first frame matches env reset).
    goal_traj = dict(oracle_demo)
    goal_traj["metadata"] = np.asarray(
        json.dumps(
            {
                "source": "oracle_goal_demo",
                "total_pixels": int(oracle_demo["pixels"].shape[0]),
                "total_actions": int(oracle_demo["actions"].shape[0]),
                "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
            }
        )
    )
    np.savez_compressed(DATASET_DIR / "goal_trajectory.npz", **goal_traj)

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


def preprocess_pixels(pixels: torch.Tensor) -> torch.Tensor:
    return lewm.preprocess_pixels(pixels, IMAGE_SIZE)


def eval_config() -> eval_module.EvalConfig:
    return eval_module.EvalConfig(
        eval_steps=EVAL_STEPS,
        eval_seed=SEED,
        bootstrap_steps=EVAL_BOOTSTRAP_STEPS,
        history_size=HISTORY_SIZE,
        plan_horizon=PLAN_HORIZON,
        action_block=ACTION_BLOCK,
        plan_blocks=PLAN_BLOCKS,
        samples=EVAL_CEM_SAMPLES,
        topk=EVAL_CEM_TOPK,
        iters=EVAL_CEM_ITERS,
        init_std=EVAL_CEM_INIT_STD,
        min_std=EVAL_CEM_MIN_STD,
        max_std=EVAL_CEM_MAX_STD,
        momentum=EVAL_CEM_MOMENTUM,
        warm_start_blend=CEM_WARM_START_BLEND,
        prior_tie_rel=CEM_PRIOR_TIE_REL,
        prior_tie_abs=CEM_PRIOR_TIE_ABS,
        prior_in_mean=CEM_PRIOR_IN_MEAN,
        prior_in_samples=CEM_PRIOR_IN_SAMPLES,
        state_cost_clip=CEM_STATE_COST_CLIP,
        delta_cost_clip=CEM_DELTA_COST_CLIP,
        near_cost_clip=CEM_NEAR_COST_CLIP,
        near_delta_weight=NEAR_DELTA_WEIGHT,
        cost_scale_min=MANIFOLD_COST_SCALE_MIN,
        traj_discount=TRAJ_DISCOUNT,
        video_fps=VIDEO_FPS,
    )


def planner_weights() -> PlannerWeights:
    return PlannerWeights(
        state=W_STATE,
        delta=W_DELTA,
        support_state=W_SUPPORT_STATE,
        support_delta=W_SUPPORT_DELTA,
        near=W_NEAR,
        action_prior=W_ACTION_PRIOR,
        smooth=W_SMOOTH,
        energy=W_ENERGY,
    )


def evaluate_lewm_cem(ckpt_path: Path, action_dim: int) -> dict:
    return eval_module.evaluate_lewm_cem(
        ckpt_path=ckpt_path,
        dataset_dir=DATASET_DIR,
        out_dir=OUT_DIR,
        model_cfg=model_config(action_dim),
        eval_cfg=eval_config(),
        weights=planner_weights(),
        env_fn=make_env,
        action_bounds_fn=action_bounds,
        render_dataset_fn=render_dataset_frame,
        render_video_fn=render_video_frame,
        record_fn=diagnostics,
        preprocess_pixels_fn=preprocess_pixels,
        goal_frame_mask_fn=goal_frame_mask,
        fall_frame_mask_fn=fall_frame_mask,
        manifold_min_mask_frames=SUPPORT_MIN_MASK_FRAMES,
        manifold_max_points=MANIFOLD_MAX_STATE_POINTS,
        manifold_max_segments=MANIFOLD_MAX_SEGMENTS,
        extra_metrics={"config": hparams()},
        domain=DOMAIN,
        task=TASK,
    )


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
