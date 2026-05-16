"""LeWM-CEM pipeline for dm_control walker/walk.

Run:
    uv run python tasks/walker_walk.py

Pipeline (everything generic lives in ``src/``):
1. ``datasets.collect_dataset`` writes ``DATASET_N_EPISODES`` explore episodes.
2. ``datasets.collect_sac_goal_demo`` trains SB3 SAC and rolls one demo.
3. ``lewm.train_lewm`` trains a LeWM-pure model (adaln_id init).
4. ``eval.evaluate_lewm_cem`` runs the CEM eval.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
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
DATASET_VERSION = 11  # bumped: single perturbed.npz; burst after warmup; no goal_manifold viz.


# ---------------------------------------------------------------------------
# Rendering / dataset collection.
# ---------------------------------------------------------------------------

IMAGE_SIZE = 112
DATASET_CAMERA_ID = 0
VIDEO_CAMERA_ID = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 40
DT = 0.025  # dm_control walker control_timestep (40 Hz).

DATASET_N_EPISODES = 128
DATASET_EPISODE_STEPS = 400
# Severity ~ Beta(a, b). Default (5, 2): mean 0.714, right-skewed —
# most episodes get strong perturbation, light wobbles are rare.
DATASET_SEVERITY_BETA_A = 5.0
DATASET_SEVERITY_BETA_B = 2.0
DATASET_WARMUP_STEPS_MAX = 100
DATASET_PUSH_BURST_SIGMA_MAX = 0.7
DATASET_PUSH_BURST_LEN = 12
DATASET_PUSH_BURST_WINDOW = (60, 200)
DATASET_OU_TAU = 0.15
SAC_TIMESTEPS = 1_000_000


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
CEM_PRIOR_TIE_ABS = 0.0
CEM_PRIOR_IN_MEAN = True
CEM_PRIOR_IN_SAMPLES = True

TRAJ_DISCOUNT = 0.92
W_STATE = 0.2
W_DELTA = 4.0
W_NEAR = 0.01
W_ACTION_PRIOR = 0.1
W_SMOOTH = 0.1
W_ENERGY = 0.002

MANIFOLD_MAX_POINTS = 4096
MANIFOLD_MAX_SEGMENTS = 4096
MANIFOLD_COST_SCALE_MIN = 1e-4


# ---------------------------------------------------------------------------
# Task callables (the only task-specific code in this file).
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
        domain_name=DOMAIN, task_name=TASK,
        task_kwargs={"random": seed}, visualize_reward=False,
    )


def action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    spec = env.action_spec()
    return (np.asarray(spec.minimum, dtype=np.float32),
            np.asarray(spec.maximum, dtype=np.float32))


def render_dataset_frame(env) -> np.ndarray:
    return env.physics.render(
        height=IMAGE_SIZE, width=IMAGE_SIZE, camera_id=DATASET_CAMERA_ID,
    ).astype(np.uint8)


def render_video_frame(env) -> np.ndarray:
    return env.physics.render(
        height=VIDEO_HEIGHT, width=VIDEO_WIDTH, camera_id=VIDEO_CAMERA_ID,
    ).astype(np.uint8)


def diagnostics(env) -> dict[str, float]:
    """Per-step task metrics — these field names appear verbatim in the
    eval trace and metrics.json (contract with downstream dashboards)."""
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
        and (DATASET_DIR / "perturbed.npz").exists()
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
    """Collect 128 SAC-perturbed rollouts + 1 deterministic SAC goal demo."""
    if dataset_ready():
        print(f"[data] using existing dataset -> {DATASET_DIR}")
        return

    print(f"[data] creating dataset -> {DATASET_DIR} ({DATASET_N_EPISODES} SAC-perturbed + 1 goal demo)")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    model = datasets.train_or_load_sac(
        env_fn=make_env,
        episode_steps=DATASET_EPISODE_STEPS,
        sac_timesteps=SAC_TIMESTEPS,
        seed=SEED,
        out_dir=OUT_DIR,
        desc=f"SAC[{TASK}]",
    )

    collect_metadata = datasets.collect_sac_perturbed_dataset(
        env_fn=make_env,
        action_bounds_fn=action_bounds,
        render_fn=render_dataset_frame,
        record_fn=diagnostics,
        control_timestep=DT,
        episode_steps=DATASET_EPISODE_STEPS,
        seed=SEED,
        out_dir=OUT_DIR,
        dataset_dir=DATASET_DIR,
        sac_timesteps=SAC_TIMESTEPS,
        cfg=datasets.PerturbedRolloutConfig(
            n_episodes=DATASET_N_EPISODES,
            severity_beta_a=DATASET_SEVERITY_BETA_A,
            severity_beta_b=DATASET_SEVERITY_BETA_B,
            warmup_steps_max=DATASET_WARMUP_STEPS_MAX,
            push_burst_sigma_max=DATASET_PUSH_BURST_SIGMA_MAX,
            push_burst_len=DATASET_PUSH_BURST_LEN,
            push_burst_window=DATASET_PUSH_BURST_WINDOW,
            ou_tau=DATASET_OU_TAU,
        ),
        domain=DOMAIN,
        task=TASK,
    )

    sac_demo = datasets.collect_sac_goal_demo(
        model=model,
        env_fn=make_env,
        action_bounds_fn=action_bounds,
        render_fn=render_dataset_frame,
        record_fn=diagnostics,
        control_timestep=DT,
        episode_steps=DATASET_EPISODE_STEPS,
        seed=SEED,
        domain=DOMAIN,
        task=TASK,
    )
    sac_return = float(sac_demo["rewards"].sum())
    goal_traj = dict(sac_demo)
    goal_traj["metadata"] = np.asarray(json.dumps({
        "source": "sac_goal_demo",
        "return": sac_return,
        "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
    }))
    np.savez_compressed(DATASET_DIR / "goal_trajectory.npz", **goal_traj)
    print(f"[data] SAC goal demo: return={sac_return:.1f} -> goal_trajectory.npz")

    render_perturbed_samples(model, collect_metadata["episodes"])

    write_json(DATASET_DIR / "metadata.json", {
        "dataset_version": DATASET_VERSION,
        "config": hparams(),
        "collect_config": collect_metadata["config"],
        "episodes": collect_metadata["episodes"],
        "goal_return": sac_return,
    })
    print(f"[data] dataset ready: 1 goal demo + {DATASET_N_EPISODES} perturbed -> {DATASET_DIR}")


def render_perturbed_samples(model, episodes: list[dict]) -> None:
    """Re-roll 2 representative perturbed episodes (max severity + median
    severity) at video resolution and write mp4s for eyeballing."""
    if not episodes:
        return
    sorted_by_sev = sorted(episodes, key=lambda e: e["severity"])
    picks = [
        ("high_severity", sorted_by_sev[-1]),
        ("mid_severity", sorted_by_sev[len(sorted_by_sev) // 2]),
    ]
    for label, ep_meta in picks:
        rng = np.random.default_rng(SEED + 1 + ep_meta["ep_idx"])
        ep = datasets.rollout_sac_episode(
            model=model,
            env_fn=make_env,
            action_bounds_fn=action_bounds,
            render_fn=render_video_frame,
            record_fn=diagnostics,
            control_timestep=DT,
            episode_steps=DATASET_EPISODE_STEPS,
            seed=SEED + 1 + ep_meta["ep_idx"],
            rng=rng,
            random_warmup_steps=ep_meta["random_warmup_steps"],
            push_burst_start=ep_meta["push_burst_start"],
            push_burst_len=ep_meta["push_burst_len"],
            push_burst_sigma=ep_meta["push_burst_sigma"],
            ou_tau=DATASET_OU_TAU,
            domain=DOMAIN,
            task=TASK,
            kind="sac_perturbed_viz",
            severity=ep_meta["severity"],
        )
        mp4 = OUT_DIR / f"perturbed_{label}.mp4"
        imageio.mimsave(mp4, list(ep["pixels"]), fps=VIDEO_FPS)
        print(f"[viz] {label} (sev={ep_meta['severity']:.2f}, return={float(ep['rewards'].sum()):.1f}) -> {mp4}")


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
        batch_size=TRAIN_BATCH_SIZE,
        train_steps=TRAIN_STEPS,
        num_workers=TRAIN_NUM_WORKERS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        log_every=LOG_EVERY,
    )


def train_lewm() -> tuple[Path, int]:
    with np.load(DATASET_DIR / "perturbed.npz") as ds:
        action_dim = int(ds["actions"].shape[-1])
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
        prior_tie_abs=CEM_PRIOR_TIE_ABS,
        prior_in_mean=CEM_PRIOR_IN_MEAN,
        prior_in_samples=CEM_PRIOR_IN_SAMPLES,
        cost_scale_min=MANIFOLD_COST_SCALE_MIN,
        traj_discount=TRAJ_DISCOUNT,
        video_fps=VIDEO_FPS,
    )


def planner_weights() -> PlannerWeights:
    return PlannerWeights(
        state=W_STATE, delta=W_DELTA, near=W_NEAR,
        action_prior=W_ACTION_PRIOR, smooth=W_SMOOTH, energy=W_ENERGY,
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
        manifold_max_points=MANIFOLD_MAX_POINTS,
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
