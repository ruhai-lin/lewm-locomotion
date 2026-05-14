"""LeWM-CEM pipeline for dm_control hopper/stand.

Same shape as ``tasks/walker_walk.py`` (everything generic lives in
``src/``); only the task-specific pieces differ:
* env identity (DOMAIN / TASK / dt / action_dim);
* ``diagnostics`` (uses ``physics.height()`` and ``physics.speed()``);
* ``goal_frame_mask`` / ``fall_frame_mask`` thresholds;
* ``oracle_score_fn`` (stand: body height + small control).

We use the ``stand`` variant rather than ``hop`` because pure-random
oracle CEM cannot solve hopper/hop (sparse stand x hop reward; see
``README.md`` "Limitations"). Stand has a dense reward and is reliably
solved by the same CEM recipe as walker / cheetah.

Run:
    uv run python tasks/hopper_stand.py
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


DOMAIN = "hopper"
TASK = "stand"
SEED = 42

DATASET_DIR = Path("datasets/hopper_stand")
OUT_DIR = Path("outputs/hopper_stand")
CKPT_DIR = OUT_DIR / "ckpt"
TRAIN_LOG_PATH = OUT_DIR / "train_log.jsonl"
DATASET_VERSION = 1


# ---------------------------------------------------------------------------
# Rendering / dataset collection.
# ---------------------------------------------------------------------------


IMAGE_SIZE = 112
DATASET_CAMERA_ID = 0
VIDEO_CAMERA_ID = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 50
DT = 0.02  # hopper control_timestep.

DATASET_N_EPISODES = 128
DATASET_EPISODE_STEPS = 400
DATASET_WORKERS = min(8, os.cpu_count() or 1)
DATASET_MP_START_METHOD = "forkserver"


# ---------------------------------------------------------------------------
# Goal- and fall-frame thresholds.
# "Good" frames for hopper/stand: body high and approximately stationary.
# ---------------------------------------------------------------------------


GOAL_FRAME_MIN_REWARD = 0.4   # dm_control stand reward saturates at ~1.0; 0.4 = "clearly standing".
GOAL_FRAME_MIN_HEIGHT = 0.8
GOAL_MIN_SEGMENT_LEN = 4

SUPPORT_FRAME_MAX_HEIGHT = 0.5  # collapsed / lying on side.


# ---------------------------------------------------------------------------
# Oracle CEM reward shaping (hopper/stand: maximise env reward = stand x small_control).
# ---------------------------------------------------------------------------


# hopper/stand is dense (stand x small_control, both smooth). Pure-random
# CEM solves it with the shared default config; no shaping needed.
USE_ORACLE_GOAL_DEMO = True

ORACLE_REWARD_GAMMA = 0.98
ORACLE_ACTION_PENALTY = 0.005
ORACLE_ACTION_SMOOTHNESS_PENALTY = 0.005

# Shared CEM strength across walker/walk, cheetah/run, hopper/stand.
ORACLE_PLAN_HORIZON = 40
ORACLE_CEM_SAMPLES = 512
ORACLE_CEM_TOPK = 64
ORACLE_CEM_ITERS = 6
ORACLE_CEM_INIT_STD = 0.55
ORACLE_CEM_MIN_STD = 0.06
ORACLE_CEM_MOMENTUM = 0.15


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
PREDICTOR_CONDITIONING = "adaln_id"

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
ACTION_CONTRAST_WEIGHT = 0.0
ACTION_CONTRAST_MARGIN = 0.0
ACTION_CONTRAST_HORIZON = 0
TRAIN_WINDOW_WEIGHTING = False


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
CEM_PRIOR_TIE_ABS = 0.0
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
# Helpers.
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
    """Hopper per-step task metrics. The two task-relevant quantities are
    body height (controls whether the hopper is on the ground) and forward
    speed; everything else is derived from those."""
    out = {
        "height": float("nan"),
        "speed": float("nan"),
        "torso_x": float("nan"),
        "torso_z": float("nan"),
    }
    try:
        out["height"] = float(env.physics.height())
        out["speed"] = float(env.physics.speed())
        out["torso_x"] = float(env.physics.named.data.xpos["torso", "x"])
        out["torso_z"] = float(env.physics.named.data.xpos["torso", "z"])
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
# Frame masks.
# ---------------------------------------------------------------------------


def goal_frame_mask(ep: dict[str, np.ndarray]) -> np.ndarray:
    """True wherever the hopper is upright and the env reward is high."""
    num_frames = len(ep["pixels"])
    if num_frames == 0:
        return np.zeros(0, dtype=bool)
    action_steps = max(num_frames - 1, 1)
    metric_ids = np.clip(np.arange(num_frames) - 1, 0, action_steps - 1)
    mask = np.ones(num_frames, dtype=bool)
    for key, threshold in (
        ("rewards", GOAL_FRAME_MIN_REWARD),
        ("height", GOAL_FRAME_MIN_HEIGHT),
    ):
        if key in ep:
            arr = np.asarray(ep[key])
            mask = mask & (arr[np.clip(metric_ids, 0, len(arr) - 1)] >= threshold)
    return mask


def fall_frame_mask(ep: dict[str, np.ndarray]) -> np.ndarray:
    """True wherever the hopper looks collapsed."""
    num_frames = len(ep["pixels"])
    if num_frames == 0:
        return np.zeros(0, dtype=bool)
    action_steps = max(num_frames - 1, 1)
    metric_ids = np.clip(np.arange(num_frames) - 1, 0, action_steps - 1)
    out = np.zeros(num_frames, dtype=bool)
    if "height" in ep:
        h = np.asarray(ep["height"])
        out = out | (h[np.clip(metric_ids, 0, len(h) - 1)] <= SUPPORT_FRAME_MAX_HEIGHT)
    return out


# ---------------------------------------------------------------------------
# Oracle CEM reward shaping for the +1 goal demo.
# ---------------------------------------------------------------------------


def oracle_score_fn(
    oracle_env,
    state0: np.ndarray,
    actions: np.ndarray,
    previous_action: np.ndarray,
) -> float:
    """Hopper/stand score = discounted env reward minus action regularisers.

    The dm_control hopper/stand reward is already dense
    (``standing(height>=0.6) * small_control``), so no extra shaping is
    needed — both factors are smooth in (state, action) and CEM can climb
    them directly.
    """
    oracle_env.reset()
    oracle_env.physics.set_state(state0)
    oracle_env.physics.forward()
    rewards: list[float] = []
    last_action = previous_action
    smooth_cost = 0.0
    for action in actions:
        ts = oracle_env.step(action.astype(np.float32))
        rewards.append(float(ts.reward or 0.0))
        smooth_cost += float(np.mean((action - last_action) ** 2))
        last_action = action
        if ts.last():
            break
    return (
        discounted_sum(rewards, ORACLE_REWARD_GAMMA)
        - ORACLE_ACTION_PENALTY * float(np.mean(actions**2))
        - ORACLE_ACTION_SMOOTHNESS_PENALTY * smooth_cost
    )


# ---------------------------------------------------------------------------
# Register task handles for parallel collection.
# ---------------------------------------------------------------------------


datasets.register_task(
    "hopper_stand",
    datasets.TaskHandles(
        env_fn=make_env,
        render_fn=render_dataset_frame,
        action_bounds_fn=action_bounds,
        record_fn=diagnostics,
        dt=DT,
    ),
)


# ---------------------------------------------------------------------------
# Pipeline stages (mirror tasks/walker_walk.py).
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
    if dataset_ready():
        print(f"[data] using existing dataset -> {DATASET_DIR}")
        return

    print(f"[data] creating dataset -> {DATASET_DIR} (128 explore + 1 oracle)")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    collect_metadata = datasets.collect_dataset(
        out_dir=DATASET_DIR,
        task_name="hopper_stand",
        task_module="tasks.hopper_stand",
        config=datasets.CollectConfig(
            n_episodes=DATASET_N_EPISODES,
            episode_steps=DATASET_EPISODE_STEPS,
            workers=DATASET_WORKERS,
            seed=SEED,
            mp_start_method=DATASET_MP_START_METHOD,
        ),
    )

    # Optional oracle demo + best-explore fallback.
    oracle_return = -float("inf")
    if USE_ORACLE_GOAL_DEMO:
        print("[data] collecting 1 oracle-CEM goal demo...")
        oracle_demo = datasets.collect_oracle_goal_demo(
            env_fn=make_env,
            action_bounds_fn=action_bounds,
            render_fn=render_dataset_frame,
            record_fn=diagnostics,
            score_fn=oracle_score_fn,
            episode_steps=DATASET_EPISODE_STEPS,
            seed=SEED,
            cfg=datasets.OracleCEMConfig(
                plan_horizon=ORACLE_PLAN_HORIZON,
                samples=ORACLE_CEM_SAMPLES,
                topk=ORACLE_CEM_TOPK,
                iters=ORACLE_CEM_ITERS,
                init_std=ORACLE_CEM_INIT_STD,
                min_std=ORACLE_CEM_MIN_STD,
                momentum=ORACLE_CEM_MOMENTUM,
            ),
            domain=DOMAIN,
            task=TASK,
        )
        oracle_path = DATASET_DIR / f"episode_{DATASET_N_EPISODES:04d}.npz"
        np.savez_compressed(oracle_path, **oracle_demo)
        oracle_return = float(oracle_demo["rewards"].sum())
        print(f"[data] oracle demo: return={oracle_return:.1f} -> {oracle_path.name}")
    else:
        print("[data] oracle goal demo skipped (USE_ORACLE_GOAL_DEMO=False); "
              "using best explore episode as goal trajectory.")
        oracle_demo = None

    best_explore_idx = max(
        range(len(collect_metadata["episodes"])),
        key=lambda i: collect_metadata["episodes"][i]["return"],
    )
    best_explore_return = float(collect_metadata["episodes"][best_explore_idx]["return"])
    if oracle_demo is None or best_explore_return > oracle_return:
        explore_path = DATASET_DIR / f"episode_{best_explore_idx:04d}.npz"
        with np.load(explore_path) as ep_npz:
            goal_source = {k: ep_npz[k] for k in ep_npz.files}
        goal_traj = dict(goal_source)
        goal_traj["metadata"] = np.asarray(
            json.dumps(
                {
                    "source": "best_explore_episode",
                    "source_path": str(explore_path.name),
                    "explore_return": best_explore_return,
                    "oracle_return": oracle_return,
                    "total_pixels": int(goal_source["pixels"].shape[0]),
                    "total_actions": int(goal_source["actions"].shape[0]),
                    "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
                }
            )
        )
        print(
            f"[data] goal trajectory = best explore episode "
            f"{explore_path.name} (return={best_explore_return:.2f})"
        )
    else:
        goal_traj = dict(oracle_demo)
        goal_traj["metadata"] = np.asarray(
            json.dumps(
                {
                    "source": "oracle_goal_demo",
                    "oracle_return": oracle_return,
                    "best_explore_return": best_explore_return,
                    "total_pixels": int(oracle_demo["pixels"].shape[0]),
                    "total_actions": int(oracle_demo["actions"].shape[0]),
                    "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
                }
            )
        )
        print(f"[data] goal trajectory = oracle demo (return={oracle_return:.2f})")
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
        window_weight_fn=None,
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
