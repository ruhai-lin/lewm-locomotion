"""Single-script pipeline for dm_control walker/walk:
    1. Generate Oracle-CEM demonstrations (pixels + actions + a goal trajectory).
    2. Train LeWM (`model.py`) on those demonstrations.
    3. Evaluate LeWM-CEM in pixel space — plan by tracking the latent
       trajectory of an Oracle-CEM demonstration (no oracle simulator at
       deployment), and render a video.

Usage:
    uv run python walker_walk.py
    uv run python walker_walk.py --skip-train  # if a checkpoint already exists
    uv run python walker_walk.py --no-eval     # data gen + train only

Outputs (default): outputs/walker_walk/
    data/episode_*.npz        per-episode raw rollouts
    data/goal_trajectory.npz  the goal demonstration used at eval time
    ckpt/lewm.pt              trained LeWM weights
    eval_cam0.mp4             LeWM-CEM evaluation video
    metrics.json              eval metrics
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dm_control import suite
import imageio.v2 as imageio

from model import SIGReg, build_lewm, lewm_loss


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass
class Config:
    seed: int = 42
    out_dir: str = "outputs/walker_walk"

    # ----- Oracle CEM data generation -----
    num_demo_episodes: int = 8
    episode_steps: int = 400  # 10s of walker/walk control_timestep=0.025
    plan_horizon: int = 20
    cem_samples: int = 96
    cem_topk: int = 12
    cem_iters: int = 3
    cem_init_std: float = 0.55
    cem_min_std: float = 0.06
    cem_momentum: float = 0.15
    reward_gamma: float = 0.98
    target_speed: float = 1.2
    progress_bonus: float = 0.75
    speed_tracking_bonus: float = 0.12
    overspeed_penalty: float = 0.35
    posture_penalty: float = 1.2
    action_penalty: float = 0.01
    action_smoothness_penalty: float = 0.015

    # ----- LeWM training -----
    image_size: int = 224
    patch_size: int = 14
    encoder_scale: str = "tiny"
    embed_dim: int = 192
    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 1  # we already capture every env step.
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    train_batch_size: int = 32
    train_steps: int = 2000
    lr: float = 5e-5
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 50

    # ----- LeWM-CEM evaluation -----
    eval_steps: int = 400
    eval_plan_horizon: int = 12
    eval_cem_samples: int = 256
    eval_cem_topk: int = 32
    eval_cem_iters: int = 3
    eval_cem_init_std: float = 0.4
    eval_cem_min_std: float = 0.05
    eval_cem_momentum: float = 0.15
    eval_traj_weight_gamma: float = 0.9  # weights[t] = gamma**t (heavier near-term)
    goal_lookahead: int = 12  # how many future goal frames the planner targets

    # ----- Video -----
    video_camera_id: int = 0
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 40

    # NB: data tensors (frames at training resolution) are stored in fp16 to
    # keep disk usage modest.

    def out_path(self) -> Path:
        return Path(self.out_dir)


# ---------------------------------------------------------------------------
# Oracle CEM (adapted from references/oracle_cem_walker_walk.py).
# ---------------------------------------------------------------------------


DOMAIN = "walker"
TASK = "walk"


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


def discounted_sum(rewards: List[float], gamma: float) -> float:
    total = 0.0
    discount = 1.0
    for r in rewards:
        total += discount * float(r)
        discount *= gamma
    return total


def diagnostics(env) -> dict:
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


def _score_candidate(
    cfg: Config,
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
        speed_tracking += float(np.exp(-((velocity - cfg.target_speed) ** 2) / 0.5))
        overspeed_cost += max(velocity - 1.8, 0.0) ** 2
        posture_cost += max(1.0 - height, 0.0) ** 2 + max(0.75 - upright, 0.0) ** 2
        last_action = action
        if ts.last():
            break

    final_x = float(oracle_env.physics.named.data.xpos["torso", "x"])
    progress = max(final_x - start_x, 0.0)
    n = max(len(rewards), 1)
    reward_score = discounted_sum(rewards, cfg.reward_gamma)
    energy_cost = float(np.mean(actions**2))
    return (
        reward_score
        + cfg.progress_bonus * progress
        + cfg.speed_tracking_bonus * (speed_tracking / n)
        - cfg.overspeed_penalty * (overspeed_cost / n)
        - cfg.posture_penalty * (posture_cost / n)
        - cfg.action_penalty * energy_cost
        - cfg.action_smoothness_penalty * smooth_cost
    )


def _oracle_cem_plan(
    cfg: Config,
    oracle_env,
    state0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    init_mean: np.ndarray,
    previous_action: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray]:
    action_dim = low.shape[0]
    mean = init_mean.astype(np.float32).copy()
    std = np.full_like(mean, cfg.cem_init_std, dtype=np.float32)
    topk = min(cfg.cem_topk, cfg.cem_samples)

    best_plan: np.ndarray | None = None
    best_score = -float("inf")

    for _ in range(cfg.cem_iters):
        samples = rng.normal(
            loc=mean[None],
            scale=std[None],
            size=(cfg.cem_samples, cfg.plan_horizon, action_dim),
        ).astype(np.float32)
        samples[0] = mean
        samples = np.clip(samples, low, high)

        scores = np.empty(samples.shape[0], dtype=np.float32)
        for i in range(samples.shape[0]):
            scores[i] = _score_candidate(cfg, oracle_env, state0, samples[i], previous_action)

        elite_idx = np.argpartition(-scores, topk - 1)[:topk]
        elites = samples[elite_idx]
        elite_mean = elites.mean(axis=0).astype(np.float32)
        elite_std = np.maximum(elites.std(axis=0), cfg.cem_min_std).astype(np.float32)
        mean = (1.0 - cfg.cem_momentum) * elite_mean + cfg.cem_momentum * mean
        std = elite_std

        i_best = int(np.argmax(scores))
        if float(scores[i_best]) > best_score:
            best_score = float(scores[i_best])
            best_plan = samples[i_best].copy()

    assert best_plan is not None
    return best_plan[0].astype(np.float32), best_score, mean.astype(np.float32)


# ---------------------------------------------------------------------------
# Episode collection.
# ---------------------------------------------------------------------------


def render_frame(env, cfg: Config) -> np.ndarray:
    """Render a frame at training image_size (square)."""
    return env.physics.render(
        height=cfg.image_size,
        width=cfg.image_size,
        camera_id=cfg.video_camera_id,
    ).astype(np.uint8)


def collect_oracle_episode(cfg: Config, episode_seed: int, desc: str = "oracle") -> dict:
    """Run one Oracle-CEM episode and return pixels (uint8), actions (fp32), rewards."""
    rng = np.random.default_rng(episode_seed)
    main_env = make_env(episode_seed)
    oracle_env = make_env(episode_seed + 10_000)

    low, high = action_bounds(main_env)
    action_dim = int(low.shape[0])
    main_env.reset()

    plan_mean = np.zeros((cfg.plan_horizon, action_dim), dtype=np.float32)
    previous_action = np.zeros(action_dim, dtype=np.float32)

    pixels: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    diags: List[dict] = []

    bar = tqdm(range(cfg.episode_steps), desc=desc, dynamic_ncols=True, leave=False)
    for _ in bar:
        pixels.append(render_frame(main_env, cfg))
        state0 = main_env.physics.get_state().copy()

        action, _, plan_mean = _oracle_cem_plan(
            cfg, oracle_env, state0, low, high, plan_mean, previous_action, rng
        )
        plan_mean = np.concatenate(
            [plan_mean[1:], np.zeros((1, action_dim), dtype=np.float32)], axis=0
        )

        ts = main_env.step(action)
        diag = diagnostics(main_env)
        diags.append(diag)
        actions.append(action.copy())
        rewards.append(float(ts.reward or 0.0))
        previous_action = action

        bar.set_postfix(
            {
                "ret": f"{sum(rewards):.1f}",
                "v": f"{diag['horizontal_velocity']:.2f}",
                "h": f"{diag['torso_height']:.2f}",
            }
        )
        if ts.last():
            break
    bar.close()

    main_env.close()
    oracle_env.close()

    return {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),       # (T, H, W, 3)
        "actions": np.stack(actions, axis=0).astype(np.float32),   # (T, action_dim)
        "rewards": np.asarray(rewards, dtype=np.float32),
        "diagnostics": diags,
    }


def generate_dataset(cfg: Config) -> Path:
    """Generate `num_demo_episodes` episodes via Oracle CEM. Returns the
    directory containing the per-episode npz files."""
    data_dir = cfg.out_path() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] Generating {cfg.num_demo_episodes} oracle episodes -> {data_dir}")
    for i in range(cfg.num_demo_episodes):
        ep_path = data_dir / f"episode_{i:03d}.npz"
        if ep_path.exists():
            print(f"  - {ep_path.name} already exists, skipping")
            continue
        ep = collect_oracle_episode(cfg, episode_seed=cfg.seed + i, desc=f"demo {i+1}/{cfg.num_demo_episodes}")
        np.savez_compressed(
            ep_path,
            pixels=ep["pixels"],
            actions=ep["actions"],
            rewards=ep["rewards"],
        )
        ep_return = float(ep["rewards"].sum())
        print(f"  + {ep_path.name}  steps={len(ep['actions'])}  return={ep_return:.1f}")
    # Also pick the best-return episode and save as goal trajectory.
    best_path = pick_best_episode(data_dir)
    goal_path = data_dir / "goal_trajectory.npz"
    if not goal_path.exists():
        ep = np.load(best_path)
        np.savez_compressed(goal_path, pixels=ep["pixels"], actions=ep["actions"], rewards=ep["rewards"])
        print(f"[data] Goal trajectory <- {best_path.name}")
    return data_dir


def pick_best_episode(data_dir: Path) -> Path:
    eps = sorted(data_dir.glob("episode_*.npz"))
    best_path = eps[0]
    best_return = -float("inf")
    for p in eps:
        with np.load(p) as ep:
            ret = float(ep["rewards"].sum())
        if ret > best_return:
            best_return = ret
            best_path = p
    return best_path


# ---------------------------------------------------------------------------
# Dataset / training.
# ---------------------------------------------------------------------------


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def preprocess_pixels(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
    """uint8 (B, T, H, W, 3) -> float (B, T, 3, image_size, image_size), ImageNet-normalized."""
    pixels = pixels.float() / 255.0
    pixels = pixels.permute(0, 1, 4, 2, 3).contiguous()
    if pixels.size(-1) != image_size or pixels.size(-2) != image_size:
        b, t = pixels.shape[:2]
        pixels = pixels.view(b * t, *pixels.shape[2:])
        pixels = F.interpolate(pixels, size=(image_size, image_size), mode="bilinear", align_corners=False)
        pixels = pixels.view(b, t, *pixels.shape[1:])
    mean = _IMAGENET_MEAN.to(pixels)
    std = _IMAGENET_STD.to(pixels)
    return (pixels - mean) / std


class WalkerWindowDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over collected oracle episodes.

    Each item is a contiguous window of `history_size + num_preds` frames and
    actions from a single episode. Pixels are kept as uint8 to save memory;
    preprocessing happens on the fly at GPU side.
    """

    def __init__(self, data_dir: Path, history_size: int, num_preds: int):
        self.window = history_size + num_preds
        self.episodes: list[dict] = []
        for p in sorted(data_dir.glob("episode_*.npz")):
            with np.load(p) as ep:
                pix = ep["pixels"]
                act = ep["actions"]
            if len(pix) < self.window:
                continue
            self.episodes.append({"pixels": pix, "actions": act.astype(np.float32)})

        self.index: list[Tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            n = len(ep["actions"]) - self.window + 1
            for s in range(n):
                self.index.append((ei, s))

        if not self.index:
            raise RuntimeError(f"No valid windows in {data_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ei, s = self.index[idx]
        ep = self.episodes[ei]
        pix = ep["pixels"][s : s + self.window]      # (T, H, W, 3) uint8
        act = ep["actions"][s : s + self.window]     # (T, action_dim) fp32
        return torch.from_numpy(pix), torch.from_numpy(act)


def train_lewm(cfg: Config, data_dir: Path) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    dataset = WalkerWindowDataset(data_dir, cfg.history_size, cfg.num_preds)
    print(f"[train] windows={len(dataset)}  episodes={len(dataset.episodes)}")

    sample_actions = dataset.episodes[0]["actions"]
    action_dim = int(sample_actions.shape[1])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_lewm(
        action_dim=action_dim,
        frameskip=cfg.frameskip,
        history_size=cfg.history_size,
        embed_dim=cfg.embed_dim,
        encoder_scale=cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
        predictor_depth=cfg.predictor_depth,
        predictor_heads=cfg.predictor_heads,
        predictor_dim_head=cfg.predictor_dim_head,
        predictor_mlp_dim=cfg.predictor_mlp_dim,
        predictor_dropout=cfg.predictor_dropout,
    ).to(device)
    sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] LeWM params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    step = 0
    pbar = tqdm(total=cfg.train_steps, desc="train", dynamic_ncols=True)
    while step < cfg.train_steps:
        for pixels_u8, actions in loader:
            if step >= cfg.train_steps:
                break
            pixels_u8 = pixels_u8.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            pixels = preprocess_pixels(pixels_u8, cfg.image_size)

            losses = lewm_loss(
                model, sigreg, pixels, actions,
                history_size=cfg.history_size,
                num_preds=cfg.num_preds,
                sigreg_weight=cfg.sigreg_weight,
            )
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            step += 1
            pbar.update(1)
            if step % cfg.log_every == 0 or step == 1:
                pbar.set_postfix({
                    "loss": f"{float(losses['loss']):.4f}",
                    "pred": f"{float(losses['pred_loss']):.4f}",
                    "sig":  f"{float(losses['sigreg_loss']):.4f}",
                })
    pbar.close()

    ckpt_dir = cfg.out_path() / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "lewm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "action_dim": action_dim,
        },
        ckpt_path,
    )
    print(f"[train] saved -> {ckpt_path}")
    return ckpt_path


def load_lewm(cfg: Config, ckpt_path: Path, action_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lewm(
        action_dim=action_dim,
        frameskip=cfg.frameskip,
        history_size=cfg.history_size,
        embed_dim=cfg.embed_dim,
        encoder_scale=cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
        predictor_depth=cfg.predictor_depth,
        predictor_heads=cfg.predictor_heads,
        predictor_dim_head=cfg.predictor_dim_head,
        predictor_mlp_dim=cfg.predictor_mlp_dim,
        predictor_dropout=cfg.predictor_dropout,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, device


# ---------------------------------------------------------------------------
# LeWM-CEM evaluation: track the latent trajectory of an oracle demo.
# ---------------------------------------------------------------------------


def encode_goal_trajectory(model, cfg: Config, goal_pixels: np.ndarray, device) -> torch.Tensor:
    """Encode the full (T, H, W, 3) uint8 goal trajectory into latents (T, D)."""
    g = torch.from_numpy(goal_pixels).unsqueeze(0).to(device)  # (1, T, H, W, 3)
    g = preprocess_pixels(g, cfg.image_size)
    out = model.encode({"pixels": g})
    return out["emb"].squeeze(0)  # (T, D)


def lewm_cem_plan(
    model,
    cfg: Config,
    history_pixels_u8: torch.Tensor,    # (1, H, H_img, W_img, 3) uint8
    history_actions: torch.Tensor,      # (1, H, action_dim)
    goal_latents_window: torch.Tensor,  # (1, T_plan, D)
    init_mean: torch.Tensor,            # (T_plan, action_dim)
    low: torch.Tensor,
    high: torch.Tensor,
    weights: torch.Tensor,
    rng: torch.Generator,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    action_dim = init_mean.size(-1)
    mean = init_mean.clone()
    std = torch.full_like(mean, cfg.eval_cem_init_std)

    history_pixels = preprocess_pixels(history_pixels_u8.to(device), cfg.image_size)

    best_plan: torch.Tensor | None = None
    best_cost = float("inf")

    for _ in range(cfg.eval_cem_iters):
        eps = torch.randn(
            (cfg.eval_cem_samples, cfg.eval_plan_horizon, action_dim),
            generator=rng,
            device=device,
        )
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        samples[0] = mean  # keep current mean as one candidate
        samples = torch.clamp(samples, low, high)

        future_actions = samples.unsqueeze(0)  # (1, S, T_plan, action_dim)

        cost = model.cem_rollout_cost(
            history_pixels=history_pixels,
            history_actions=history_actions.to(device),
            future_actions=future_actions,
            goal_latents=goal_latents_window,
            weights=weights,
        )  # (1, S)
        cost = cost.squeeze(0)

        topk = min(cfg.eval_cem_topk, cfg.eval_cem_samples)
        elite_idx = torch.topk(cost, topk, largest=False).indices
        elites = samples[elite_idx]
        elite_mean = elites.mean(dim=0)
        elite_std = elites.std(dim=0).clamp_min(cfg.eval_cem_min_std)
        mean = (1.0 - cfg.eval_cem_momentum) * elite_mean + cfg.eval_cem_momentum * mean
        std = elite_std

        i_best = int(torch.argmin(cost).item())
        if float(cost[i_best]) < best_cost:
            best_cost = float(cost[i_best])
            best_plan = samples[i_best].clone()

    assert best_plan is not None
    return best_plan, mean


def evaluate_lewm_cem(cfg: Config, ckpt_path: Path, data_dir: Path) -> dict:
    goal_path = data_dir / "goal_trajectory.npz"
    if not goal_path.exists():
        raise FileNotFoundError(f"missing goal trajectory: {goal_path}")
    with np.load(goal_path) as g:
        goal_pixels = g["pixels"]      # (T_goal, H, W, 3) uint8
        goal_actions = g["actions"]    # (T_goal, action_dim)

    action_dim = int(goal_actions.shape[1])
    model, device = load_lewm(cfg, ckpt_path, action_dim)
    print(f"[eval] device={device}  goal_T={len(goal_pixels)}")

    # encode goal once
    goal_latents = encode_goal_trajectory(model, cfg, goal_pixels, device)  # (T_goal, D)
    T_goal = goal_latents.size(0)

    # weights[t] = gamma**t
    weights = torch.tensor(
        [cfg.eval_traj_weight_gamma ** i for i in range(cfg.eval_plan_horizon)],
        device=device, dtype=torch.float32,
    )

    main_env = make_env(cfg.seed + 9999)
    main_env.reset()
    low_np, high_np = action_bounds(main_env)
    low = torch.tensor(low_np, device=device)
    high = torch.tensor(high_np, device=device)

    # warm up history with `history_size` zero-action steps
    history_frames: List[np.ndarray] = []
    history_actions_list: List[np.ndarray] = []
    for _ in range(cfg.history_size):
        history_frames.append(render_frame(main_env, cfg))
        zero_act = np.zeros(action_dim, dtype=np.float32)
        ts = main_env.step(zero_act)
        history_actions_list.append(zero_act.copy())
        if ts.last():
            break

    plan_mean = torch.zeros((cfg.eval_plan_horizon, action_dim), device=device)
    rng_th = torch.Generator(device=device).manual_seed(cfg.seed + 7777)

    frames: List[np.ndarray] = []
    rewards: List[float] = []
    velocities: List[float] = []
    heights: List[float] = []
    uprights: List[float] = []

    # Goal pointer: which frame in the goal trajectory we currently align to.
    goal_ptr = 0

    bar = tqdm(range(cfg.eval_steps), desc="lewm-cem eval", dynamic_ncols=True)
    for t in bar:
        # Render full-size frame for the video; convert to current history.
        big_frame = main_env.physics.render(
            height=cfg.video_height,
            width=cfg.video_width,
            camera_id=cfg.video_camera_id,
        ).astype(np.uint8)
        frames.append(big_frame)

        # Build CEM history tensors.
        hist_pix = np.stack(history_frames[-cfg.history_size:], axis=0)  # (H, H_img, W_img, 3)
        hist_act = np.stack(history_actions_list[-cfg.history_size:], axis=0)
        history_pixels_u8 = torch.from_numpy(hist_pix).unsqueeze(0)
        history_actions = torch.from_numpy(hist_act).unsqueeze(0)

        # Slice the goal trajectory for the next plan-horizon steps.
        g_start = min(goal_ptr + 1, T_goal - 1)
        g_end = min(g_start + cfg.eval_plan_horizon, T_goal)
        if g_end - g_start < cfg.eval_plan_horizon:
            # Pad the tail with the final goal latent.
            tail = goal_latents[-1:].expand(cfg.eval_plan_horizon - (g_end - g_start), -1)
            goal_window = torch.cat([goal_latents[g_start:g_end], tail], dim=0)
        else:
            goal_window = goal_latents[g_start:g_end]
        goal_window = goal_window.unsqueeze(0)  # (1, T_plan, D)

        plan, plan_mean = lewm_cem_plan(
            model, cfg,
            history_pixels_u8=history_pixels_u8,
            history_actions=history_actions,
            goal_latents_window=goal_window,
            init_mean=plan_mean,
            low=low, high=high,
            weights=weights,
            rng=rng_th, device=device,
        )

        # Receding-horizon warm start.
        plan_mean = torch.cat([plan_mean[1:], torch.zeros((1, action_dim), device=device)], dim=0)

        action = plan[0].detach().cpu().numpy().astype(np.float32)
        ts = main_env.step(action)
        diag = diagnostics(main_env)

        history_frames.append(render_frame(main_env, cfg))
        history_actions_list.append(action.copy())

        rewards.append(float(ts.reward or 0.0))
        velocities.append(diag["horizontal_velocity"])
        heights.append(diag["torso_height"])
        uprights.append(diag["torso_upright"])

        goal_ptr = min(goal_ptr + 1, T_goal - 1)

        bar.set_postfix({
            "ret": f"{sum(rewards):.1f}",
            "v": f"{diag['horizontal_velocity']:.2f}",
            "h": f"{diag['torso_height']:.2f}",
            "g": f"{goal_ptr}/{T_goal}",
        })
        if ts.last():
            break
    bar.close()
    main_env.close()

    out_dir = cfg.out_path()
    video_path = out_dir / f"eval_cam{cfg.video_camera_id}.mp4"
    imageio.mimsave(video_path, frames, fps=cfg.video_fps)

    metrics = {
        "domain": DOMAIN,
        "task": TASK,
        "policy": "lewm_cem",
        "steps": len(rewards),
        "total_return": float(np.sum(rewards)) if rewards else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_horizontal_velocity": float(np.nanmean(velocities)) if velocities else float("nan"),
        "mean_torso_height": float(np.nanmean(heights)) if heights else float("nan"),
        "mean_torso_upright": float(np.nanmean(uprights)) if uprights else float("nan"),
        "video_path": str(video_path),
        "goal_trajectory": str(goal_path),
        "config": asdict(cfg),
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] video -> {video_path}")
    print(f"[eval] return={metrics['total_return']:.1f}  mean_v={metrics['mean_horizontal_velocity']:.2f}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs/walker_walk")
    p.add_argument("--num-demo-episodes", type=int, default=None)
    p.add_argument("--episode-steps", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--skip-data", action="store_true", help="reuse existing oracle data")
    p.add_argument("--skip-train", action="store_true", help="reuse existing checkpoint")
    p.add_argument("--no-eval", action="store_true", help="skip the LeWM-CEM evaluation")
    return p.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.num_demo_episodes is not None:
        cfg.num_demo_episodes = args.num_demo_episodes
    if args.episode_steps is not None:
        cfg.episode_steps = args.episode_steps
    if args.train_steps is not None:
        cfg.train_steps = args.train_steps
    if args.eval_steps is not None:
        cfg.eval_steps = args.eval_steps
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    cfg.out_path().mkdir(parents=True, exist_ok=True)
    print("[cfg]", json.dumps(asdict(cfg), indent=2))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data_dir = cfg.out_path() / "data"
    if args.skip_data and data_dir.exists():
        print(f"[data] skipping generation, using {data_dir}")
    else:
        data_dir = generate_dataset(cfg)

    ckpt_path = cfg.out_path() / "ckpt" / "lewm.pt"
    if args.skip_train and ckpt_path.exists():
        print(f"[train] skipping training, using {ckpt_path}")
    else:
        ckpt_path = train_lewm(cfg, data_dir)

    if args.no_eval:
        print("[eval] skipped per --no-eval")
        return

    evaluate_lewm_cem(cfg, ckpt_path, data_dir)


if __name__ == "__main__":
    main()
