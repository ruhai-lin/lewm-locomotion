"""Reference-aligned LeWM training and rollout utilities.

The model architecture lives in ``model.py`` and intentionally stays close to
``references/le-wm``. This module keeps the training loss aligned with the
upstream LeWM forward pass: encode pixels/actions, predict future latent states,
and apply SIGReg on the encoded latent sequence. Locomotion-specific code may
add recursive rollout supervision, but task semantics do not enter here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import SIGReg, build_lewm


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@dataclass(frozen=True)
class LeWMModelConfig:
    image_size: int
    action_dim: int
    history_size: int
    frameskip: int = 1
    embed_dim: int = 192
    encoder_scale: str = "tiny"
    patch_size: int = 14
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1


@dataclass(frozen=True)
class LeWMTrainConfig:
    num_preds: int = 1
    rollout_steps: int = 0
    rollout_loss_weight: float = 0.0
    sigreg_weight: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    batch_size: int = 64
    train_steps: int = 10000
    num_workers: int = 2
    checkpoint_every: int = 1000
    lr: float = 5e-5
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 50


WindowWeightFn = Callable[[dict[str, np.ndarray], int, int], float]


def preprocess_pixels(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert uint8 NHWTC pixels to ImageNet-normalized NCTHW tensors."""
    pixels = pixels.float() / 255.0
    pixels = pixels.permute(0, 1, 4, 2, 3).contiguous()
    if pixels.size(-1) != image_size or pixels.size(-2) != image_size:
        b, t = pixels.shape[:2]
        pixels = pixels.view(b * t, *pixels.shape[2:])
        pixels = F.interpolate(
            pixels,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        pixels = pixels.view(b, t, *pixels.shape[1:])
    return (pixels - _IMAGENET_MEAN.to(pixels)) / _IMAGENET_STD.to(pixels)


class StaticWindowDataset(torch.utils.data.Dataset):
    """Contiguous windows where action[t] advances pixels[t] to pixels[t+1]."""

    def __init__(
        self,
        dataset_dir: Path,
        history_size: int,
        num_preds: int,
        rollout_steps: int,
        window_weight_fn: WindowWeightFn | None = None,
    ):
        self.window_pixels = history_size + max(num_preds, rollout_steps)
        self.window_actions = self.window_pixels - 1
        self.episodes: list[dict[str, np.ndarray]] = []
        for path in sorted(dataset_dir.glob("episode_*.npz")):
            with np.load(path) as ep:
                episode = {key: ep[key].copy() for key in ep.files}
                episode["__path__"] = np.asarray(str(path))
            pixels = episode["pixels"]
            actions = episode["actions"].astype(np.float32)
            if len(pixels) >= self.window_pixels and len(actions) >= self.window_actions:
                episode["actions"] = actions
                self.episodes.append(episode)

        self.index: list[tuple[int, int]] = []
        weights: list[float] = []
        for ep_idx, ep in enumerate(self.episodes):
            max_start = min(
                len(ep["pixels"]) - self.window_pixels,
                len(ep["actions"]) - self.window_actions,
            )
            for start in range(max_start + 1):
                self.index.append((ep_idx, start))
                if window_weight_fn is None:
                    weights.append(1.0)
                else:
                    weights.append(max(float(window_weight_fn(ep, start, self.window_pixels)), 1e-6))

        if not self.index:
            raise RuntimeError(f"No valid training windows in {dataset_dir}")
        self.sample_weights = torch.tensor(weights, dtype=torch.double)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        pixels = ep["pixels"][start : start + self.window_pixels]
        actions = ep["actions"][start : start + self.window_actions]
        return torch.from_numpy(pixels), torch.from_numpy(actions)


def build_model(cfg: LeWMModelConfig, device: torch.device):
    return build_lewm(
        action_dim=cfg.action_dim,
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


def train_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def lewm_loss(
    model,
    sigreg: SIGReg,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    cfg: LeWMTrainConfig,
    history_size: int,
) -> dict[str, torch.Tensor]:
    actions = torch.nan_to_num(actions, 0.0)

    # Reference LeWM forward: use the context action at each context frame to
    # predict the latent shifted by NUM_PREDS.
    info = model.encode({"pixels": pixels, "action": actions[:, :history_size]})
    emb = info["emb"]
    act_emb = info["act_emb"]
    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    target_emb = emb[:, cfg.num_preds : cfg.num_preds + history_size]
    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred_emb - target_emb).pow(2).mean()

    rollout_loss = pred_loss.new_zeros(())
    rollout_steps = min(cfg.rollout_steps, pixels.size(1) - history_size)
    if cfg.rollout_loss_weight > 0.0 and rollout_steps > 0:
        hist_actions = torch.zeros_like(actions[:, :history_size])
        if history_size > 1:
            hist_actions[:, 1:] = actions[:, : history_size - 1]
        future_actions = actions[:, history_size - 1 : history_size - 1 + rollout_steps]
        rollout_pred = aligned_rollout_latents(
            model,
            pixels[:, :history_size],
            hist_actions,
            future_actions.unsqueeze(1),
            history_size=history_size,
        ).squeeze(1)
        rollout_target = emb[:, history_size : history_size + rollout_steps]
        rollout_loss = (rollout_pred - rollout_target).pow(2).mean()

    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + cfg.rollout_loss_weight * rollout_loss + cfg.sigreg_weight * sigreg_loss
    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
    }


def train_lewm(
    *,
    dataset_dir: Path,
    ckpt_path: Path,
    train_log_path: Path,
    model_cfg: LeWMModelConfig,
    train_cfg: LeWMTrainConfig,
    hparams: dict,
    append_jsonl: Callable[[Path, dict], None],
    window_weight_fn: WindowWeightFn | None = None,
) -> tuple[Path, int]:
    dataset = StaticWindowDataset(
        dataset_dir,
        history_size=model_cfg.history_size,
        num_preds=train_cfg.num_preds,
        rollout_steps=train_cfg.rollout_steps,
        window_weight_fn=window_weight_fn,
    )
    sampler = None
    shuffle = True
    if window_weight_fn is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(model_cfg, device)
    sigreg = SIGReg(knots=train_cfg.sigreg_knots, num_proj=train_cfg.sigreg_num_proj).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train] device={device} params={n_params / 1e6:.2f}M "
        f"windows={len(dataset)} episodes={len(dataset.episodes)}"
    )
    if train_log_path.exists():
        train_log_path.unlink()
    append_jsonl(
        train_log_path,
        {
            "event": "start",
            "device": str(device),
            "params": n_params,
            "windows": len(dataset),
            "episodes": len(dataset.episodes),
            "weighted_sampler": bool(window_weight_fn is not None),
            "config": hparams,
        },
    )

    model.train()
    step = 0
    last_log: dict[str, float] = {}
    pbar = tqdm(total=train_cfg.train_steps, desc="train LeWM", dynamic_ncols=True)
    while step < train_cfg.train_steps:
        for pixels_u8, actions in loader:
            if step >= train_cfg.train_steps:
                break
            pixels_u8 = pixels_u8.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            pixels = preprocess_pixels(pixels_u8, model_cfg.image_size)

            with train_autocast(device):
                losses = lewm_loss(model, sigreg, pixels, actions, train_cfg, model_cfg.history_size)

            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()

            step += 1
            pbar.update(1)
            if step % train_cfg.log_every == 0 or step == 1:
                last_log = {
                    "event": "step",
                    "step": step,
                    "loss": float(losses["loss"].detach()),
                    "pred_loss": float(losses["pred_loss"]),
                    "rollout_loss": float(losses["rollout_loss"]),
                    "sigreg_loss": float(losses["sigreg_loss"]),
                    "lr": float(optim.param_groups[0]["lr"]),
                }
                append_jsonl(train_log_path, last_log)
                pbar.set_postfix(
                    {
                        "loss": f"{last_log['loss']:.4f}",
                        "pred": f"{last_log['pred_loss']:.4f}",
                        "roll": f"{last_log['rollout_loss']:.4f}",
                        "sig": f"{last_log['sigreg_loss']:.4f}",
                    }
                )
            if train_cfg.checkpoint_every > 0 and step % train_cfg.checkpoint_every == 0:
                latest_path = ckpt_path.with_name("lewm_latest.pt")
                latest_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "action_dim": model_cfg.action_dim,
                        "config": hparams,
                        "step": step,
                    },
                    latest_path,
                )
    pbar.close()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "action_dim": model_cfg.action_dim,
            "config": hparams,
        },
        ckpt_path,
    )
    print(f"[train] saved -> {ckpt_path}")
    append_jsonl(
        train_log_path,
        {
            "event": "done",
            "step": step,
            "checkpoint": str(ckpt_path),
            "last_log": last_log,
        },
    )
    return ckpt_path, model_cfg.action_dim


def load_lewm(ckpt_path: Path, model_cfg: LeWMModelConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, device


@torch.no_grad()
def encode_frames(
    model,
    pixels: np.ndarray,
    device: torch.device,
    image_size: int,
    chunk: int = 128,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(pixels), chunk):
        batch = torch.from_numpy(pixels[start : start + chunk]).unsqueeze(0).to(device)
        emb = model.encode({"pixels": preprocess_pixels(batch, image_size)})["emb"].squeeze(0)
        chunks.append(emb)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def aligned_rollout_latents(
    model,
    history_pixels: torch.Tensor,
    history_actions: torch.Tensor,
    future_actions: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Roll out one-step LeWM dynamics with action[t] -> frame[t+1].

    ``history_actions`` is shifted so the last raw action fed to the predictor
    is the candidate current-frame action. Therefore
    ``future_actions[:, :, 0]`` affects the first predicted latent.
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
        emb_in = emb[:, -history_size:]
        next_action = future[:, t : t + 1]
        if history_size == 1:
            raw_actions = next_action
        else:
            raw_actions = torch.cat([action_buf[:, -history_size + 1 :], next_action], dim=1)
        act_emb = model.action_encoder(raw_actions)
        pred = model.predict(emb_in, act_emb)[:, -1:]
        preds.append(pred)
        emb = torch.cat([emb, pred], dim=1)
        action_buf = torch.cat([action_buf, next_action], dim=1)

    pred = torch.cat(preds, dim=1)
    return pred.reshape(b, samples, horizon, -1)


@torch.no_grad()
def rollout_prediction_diagnostics(
    model,
    goal_pixels: np.ndarray,
    goal_actions_np: np.ndarray,
    goal_latents: torch.Tensor,
    device: torch.device,
    *,
    image_size: int,
    history_size: int,
    horizons: list[int],
) -> dict:
    action_dim = int(goal_actions_np.shape[1])
    starts = [0, 1, 5, 20, 40]
    out: dict[str, dict] = {}
    max_horizon = min(max(horizons), len(goal_actions_np) - history_size)
    horizon_values: dict[int, list[float]] = {
        int(h): [] for h in horizons if h <= max_horizon
    }
    for start in starts:
        if start + history_size + max_horizon >= len(goal_pixels):
            continue
        hist_pix = torch.from_numpy(goal_pixels[start : start + history_size]).unsqueeze(0).to(device)
        hist_actions = torch.zeros(1, history_size, action_dim, device=device)
        if history_size > 1:
            hist_actions[:, 1:] = torch.from_numpy(
                goal_actions_np[start : start + history_size - 1]
            ).to(device)
        future = torch.from_numpy(
            goal_actions_np[
                start + history_size - 1 : start + history_size - 1 + max_horizon
            ]
        ).view(1, 1, max_horizon, action_dim).to(device)
        pred = aligned_rollout_latents(
            model,
            preprocess_pixels(hist_pix, image_size),
            hist_actions,
            future,
            history_size=history_size,
        ).squeeze(0).squeeze(0)
        target = goal_latents[start + history_size : start + history_size + max_horizon]
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
        np.stack([goal_pixels[0].copy() for _ in range(history_size)], axis=0)
    ).unsqueeze(0).to(device)
    repeated_actions = torch.zeros(1, history_size, action_dim, device=device)
    future = torch.from_numpy(goal_actions_np[:max_horizon]).view(
        1, 1, max_horizon, action_dim
    ).to(device)
    pred = aligned_rollout_latents(
        model,
        preprocess_pixels(repeated, image_size),
        repeated_actions,
        future,
        history_size=history_size,
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
