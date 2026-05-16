"""Reference-aligned LeWM architecture + training + rollout utilities.

The architecture (``SIGReg``, ``JEPA``, ``ARPredictor``, etc.) is kept
intentionally close to ``references/le-wm`` so the only places where
locomotion concerns enter the model are:

* ``_ConditionalBlock(conditioning="adaln_id")`` — preserves the upstream
  AdaLN structure but starts the action-conditioning MLP with non-zero
  gating so the predictor cannot collapse onto pure visual continuity at
  init. See ``experiments.md`` E4d.
* ``JEPA.rollout_latents`` / ``JEPA.cem_rollout_cost`` — batched no-grad
  rollouts used by the CEM planner; they do not change the training loss.

Training uses the upstream loss form ``pred_loss + λ * sigreg_loss``, with
an optional recursive rollout supervision term that is itself
prediction-style. Task semantics do not enter this file.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from tqdm import tqdm
from transformers import ViTConfig, ViTModel


# ---------------------------------------------------------------------------
# SIGReg (verbatim from upstream LeWM).
# ---------------------------------------------------------------------------


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian regularizer (single-GPU)."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Transformer building blocks (verbatim from upstream LeWM `module.py`).
# ---------------------------------------------------------------------------


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class _ConditionalBlock(nn.Module):
    """Transformer block with AdaLN conditioning.

    By default (``conditioning="adaln_zero"``) the AdaLN MLP is initialised
    to zero, matching upstream LeWM. This makes the action effectively
    disabled at init and must climb out of a flat basin during training.

    With ``conditioning="adaln_id"`` we initialise the AdaLN MLP so the
    block starts as an *identity-with-action-residual* — gates default to 1
    and shifts/scales to 0, but the linear is left at its default Xavier
    init for the gates (which means non-zero residual influence of the
    action embedding from step 1). Loss form is unchanged; the only
    difference is initial conditioning strength.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        conditioning: str = "adaln_zero",
    ):
        super().__init__()
        self.attn = _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.conditioning = conditioning
        if conditioning == "adaln_zero":
            nn.init.constant_(self.adaLN[-1].weight, 0.0)
            nn.init.constant_(self.adaLN[-1].bias, 0.0)
        elif conditioning == "adaln_id":
            # Default Linear init (Kaiming uniform on weight, zeros on bias).
            # Then nudge biases so the block starts roughly identity-plus-action.
            nn.init.zeros_(self.adaLN[-1].bias)
            with torch.no_grad():
                bias = self.adaLN[-1].bias.view(6, dim)
                # gate_msa = bias[2], gate_mlp = bias[5]; bias them slightly
                # positive so the attention/mlp paths are open at init.
                bias[2].fill_(0.1)
                bias[5].fill_(0.1)
        else:
            raise ValueError(f"unknown conditioning '{conditioning}'")

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        conditioning: str = "adaln_zero",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                _ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout, conditioning=conditioning)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for blk in self.layers:
            x = blk(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction (upstream LeWM)."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        conditioning: str = "adaln_zero",
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            conditioning=conditioning,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)


class Embedder(nn.Module):
    """Action embedder (upstream LeWM `module.Embedder`)."""

    def __init__(self, input_dim: int = 10, smoothed_dim: int = 10, emb_dim: int = 10, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """Simple MLP (upstream LeWM `module.MLP`)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ViT encoder factory — equivalent to `stable_pretraining.backbone.utils.vit_hf`
# with `pretrained=False, use_mask_token=False`.
# ---------------------------------------------------------------------------


_VIT_SIZE_PRESETS = {
    # Match HuggingFace `google/vit-tiny-patch16-224` style configs.
    "tiny":  dict(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,  intermediate_size=768),
    "small": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,  intermediate_size=1536),
    "base":  dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
}


def build_vit_encoder(scale: str = "tiny", patch_size: int = 14, image_size: int = 224) -> ViTModel:
    """Build a randomly-initialized ViT encoder matching upstream LeWM."""
    if scale not in _VIT_SIZE_PRESETS:
        raise ValueError(f"unknown vit scale '{scale}'; choose from {list(_VIT_SIZE_PRESETS)}")
    cfg = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        **_VIT_SIZE_PRESETS[scale],
    )
    return ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)


# ---------------------------------------------------------------------------
# JEPA / LeWM trajectory model.
# ---------------------------------------------------------------------------


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    """LeWM JEPA — same architecture as upstream `references/le-wm/jepa.JEPA`,
    extended with trajectory-tracking cost / batched CEM rollout.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    # ------------------------------------------------------------------
    # Encoding (matches upstream).
    # ------------------------------------------------------------------

    def encode(self, info: dict) -> dict:
        """Encode observations and actions into embeddings.
        info: dict with `pixels` and (optional) `action`.
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next-state embeddings.
        emb: (B, T, D); act_emb: (B, T, A_emb).
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    # ------------------------------------------------------------------
    # Inference rollout (upstream-style — kept for compatibility).
    # ------------------------------------------------------------------

    def rollout(self, info: dict, action_sequence: torch.Tensor, history_size: int = 3) -> dict:
        """Rollout the model given an initial info dict and action sequence.

        pixels: (B, S, T, C, H, W) — only `pixels[:, :, :history_size]` is used.
        action_sequence: (B, S, T, action_dim).
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: _detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    # ------------------------------------------------------------------
    # Trajectory-tracking cost (extension over upstream LeWM).
    # ------------------------------------------------------------------

    def criterion(self, info_dict: dict) -> torch.Tensor:
        """Cost between predicted embeddings and goal embeddings.

        Upstream LeWM only used the *last* frame of `goal_emb` and `pred_emb`.
        For locomotion / goal-trajectory tracking we instead aggregate MSE over
        the entire predicted trajectory vs the corresponding window of the
        goal trajectory.

        - pred_emb: (B, S, T_pred, D)
        - goal_emb: (B, S, T_goal, D)  — broadcast across S if needed.
        """
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        if goal_emb.dim() == pred_emb.dim() - 1:
            goal_emb = goal_emb.unsqueeze(1)  # (B, 1, T, D) → broadcast over S
        T = min(pred_emb.size(-2), goal_emb.size(-2))
        pred_emb = pred_emb[..., -T:, :]
        goal_emb = goal_emb[..., -T:, :].expand_as(pred_emb)
        weights = info_dict.get("traj_weights")
        velocity_weight = float(info_dict.get("velocity_weight", 0.0))
        diff = (pred_emb - goal_emb.detach()).pow(2).mean(dim=-1)  # (B, S, T)
        if weights is not None:
            w = weights[:T].to(diff)
            diff = diff * w
        cost = diff.mean(dim=-1)

        if velocity_weight > 0.0 and T > 1:
            pred_vel = pred_emb[..., 1:, :] - pred_emb[..., :-1, :]
            goal_vel = goal_emb[..., 1:, :] - goal_emb[..., :-1, :]
            vel_diff = (pred_vel - goal_vel.detach()).pow(2).mean(dim=-1)  # (B, S, T - 1)
            if weights is not None:
                vel_w = weights[1:T].to(vel_diff)
                vel_diff = vel_diff * vel_w
            cost = cost + velocity_weight * vel_diff.mean(dim=-1)
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute cost of action candidates given an info dict with goal trajectory + initial state."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # Encode the *goal trajectory* into a latent trajectory.
        goal = {k: v[:, 0] if torch.is_tensor(v) else v for k, v in info_dict.items()}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)
        info_dict["goal_emb"] = goal["emb"]

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)

    # ------------------------------------------------------------------
    # Batched CEM rollout (extension): efficient, no-grad, takes a fixed
    # history (no `S` dim on pixels) and a goal latent trajectory.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout_latents(
        self,
        history_pixels: torch.Tensor,
        history_actions: torch.Tensor,
        future_actions: torch.Tensor,
        history_size: int | None = None,
    ) -> torch.Tensor:
        """Roll out predicted latents for every action candidate.

        history_pixels: (B, H, C, H_img, W_img). Past frames.
        history_actions: (B, H, action_dim). Past actions.
        future_actions: (B, S, T_future, action_dim). Plan candidates.

        Returns pred_latents: (B, S, T_future, D).
        """
        B, H = history_pixels.shape[:2]
        S = future_actions.size(1)
        T_fut = future_actions.size(2)
        HS = history_size or H

        hist = self.encode({"pixels": history_pixels, "action": history_actions})
        hist_emb = hist["emb"]            # (B, H, D)
        hist_act_emb = hist["act_emb"]    # (B, H, D)

        emb = hist_emb.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1)
        act_emb_buf = hist_act_emb.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1)
        future_act_emb = self.action_encoder(future_actions.reshape(B * S, T_fut, -1))

        all_pred = []
        for t in range(T_fut):
            emb_in = emb[:, -HS:]
            act_in = act_emb_buf[:, -HS:]
            pred = self.predict(emb_in, act_in)[:, -1:]  # (B*S, 1, D)
            all_pred.append(pred)
            emb = torch.cat([emb, pred], dim=1)
            act_emb_buf = torch.cat([act_emb_buf, future_act_emb[:, t : t + 1]], dim=1)

        pred = torch.cat(all_pred, dim=1)  # (B*S, T_fut, D)
        return pred.reshape(B, S, T_fut, -1)

    @torch.no_grad()
    def cem_rollout_cost(
        self,
        history_pixels: torch.Tensor,
        history_actions: torch.Tensor,
        future_actions: torch.Tensor,
        goal_latents: torch.Tensor,
        weights: torch.Tensor | None = None,
        velocity_weight: float = 0.0,
        history_size: int | None = None,
    ) -> torch.Tensor:
        """Rollout + trajectory cost in one call. Returns (B, S)."""
        pred = self.rollout_latents(history_pixels, history_actions, future_actions, history_size)
        info = {
            "predicted_emb": pred,
            "goal_emb": goal_latents,
            "velocity_weight": velocity_weight,
        }
        if weights is not None:
            info["traj_weights"] = weights
        return self.criterion(info)


# ---------------------------------------------------------------------------
# Convenience factory matching upstream LeWM `lewm.yaml`.
# ---------------------------------------------------------------------------


def build_lewm(
    *,
    action_dim: int,
    frameskip: int = 1,
    history_size: int = 3,
    embed_dim: int = 192,
    encoder_scale: str = "tiny",
    patch_size: int = 14,
    image_size: int = 224,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_dim_head: int = 64,
    predictor_mlp_dim: int = 2048,
    predictor_dropout: float = 0.1,
    predictor_emb_dropout: float = 0.0,
    predictor_conditioning: str = "adaln_zero",
    proj_hidden: int = 2048,
) -> JEPA:
    """Build a LeWM with the same hyperparameters as `config/train/lewm.yaml`."""
    encoder = build_vit_encoder(scale=encoder_scale, patch_size=patch_size, image_size=image_size)
    hidden_dim = encoder.config.hidden_size
    effective_act_dim = frameskip * action_dim

    predictor = ARPredictor(
        num_frames=history_size,
        depth=predictor_depth,
        heads=predictor_heads,
        mlp_dim=predictor_mlp_dim,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dim_head=predictor_dim_head,
        dropout=predictor_dropout,
        emb_dropout=predictor_emb_dropout,
        conditioning=predictor_conditioning,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, smoothed_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden,
        norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden,
        norm_fn=nn.BatchNorm1d,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


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
    # "adaln_zero" matches upstream LeWM (zero-init action conditioning).
    # "adaln_id" starts with non-zero action effect; loss form unchanged.
    predictor_conditioning: str = "adaln_zero"


@dataclass(frozen=True)
class LeWMTrainConfig:
    num_preds: int = 1
    rollout_steps: int = 0
    rollout_loss_weight: float = 0.0
    sigreg_weight: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    # Action-contrast loss (experiment E4 ablation): forces the predictor
    # to produce measurably different latents under different actions.
    # Hardcoded margin=0 / horizon=8 — only the weight is a knob.
    # ``action_contrast_weight = 0`` reproduces the upstream LeWM loss.
    action_contrast_weight: float = 0.0
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

        def _add(name: str, pixels: np.ndarray, actions: np.ndarray, extras: dict) -> None:
            actions = actions.astype(np.float32)
            if len(pixels) >= self.window_pixels and len(actions) >= self.window_actions:
                ep = {"pixels": pixels, "actions": actions, "__path__": np.asarray(name)}
                ep.update(extras)
                self.episodes.append(ep)

        # Goal demo (single trajectory).
        goal_path = dataset_dir / "goal_trajectory.npz"
        if goal_path.exists():
            with np.load(goal_path) as g:
                _add(
                    str(goal_path),
                    g["pixels"].copy(),
                    g["actions"].copy(),
                    {k: g[k].copy() for k in g.files if k not in ("pixels", "actions")},
                )

        # Perturbed batch (N episodes stacked).
        perturbed_path = dataset_dir / "perturbed.npz"
        if perturbed_path.exists():
            with np.load(perturbed_path) as p:
                pixels_batch = p["pixels"]   # (N, T+1, H, W, 3)
                actions_batch = p["actions"] # (N, T, A)
                # Per-step task metrics stack as (N, T, ...).
                per_episode_keys = [k for k in p.files
                                    if k not in ("pixels", "actions", "metadata")
                                    and p[k].ndim >= 1
                                    and p[k].shape[0] == pixels_batch.shape[0]]
                for i in range(pixels_batch.shape[0]):
                    extras = {k: p[k][i].copy() for k in per_episode_keys}
                    _add(
                        f"{perturbed_path}#{i:04d}",
                        pixels_batch[i].copy(),
                        actions_batch[i].copy(),
                        extras,
                    )

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
        predictor_conditioning=cfg.predictor_conditioning,
    ).to(device)


def train_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _rollout_with_actions(
    model,
    history_pixels: torch.Tensor,
    history_actions: torch.Tensor,
    future_actions: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Grad-enabled rollout that mirrors ``aligned_rollout_latents`` for use
    inside training. Returns (B, T_fut, D)."""
    b, h = history_pixels.shape[:2]
    horizon = future_actions.size(1)
    hist = model.encode({"pixels": history_pixels})
    emb = hist["emb"]
    action_buf = history_actions.float()
    future = future_actions.float()

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
    return torch.cat(preds, dim=1)


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
    rollout_pred = None
    rollout_target = None
    if cfg.rollout_loss_weight > 0.0 and rollout_steps > 0:
        hist_actions = torch.zeros_like(actions[:, :history_size])
        if history_size > 1:
            hist_actions[:, 1:] = actions[:, : history_size - 1]
        future_actions = actions[:, history_size - 1 : history_size - 1 + rollout_steps]
        rollout_pred = _rollout_with_actions(
            model,
            pixels[:, :history_size],
            hist_actions,
            future_actions,
            history_size=history_size,
        )
        rollout_target = emb[:, history_size : history_size + rollout_steps]
        rollout_loss = (rollout_pred - rollout_target).pow(2).mean()

    # Action-contrast loss: with the same visual history but shuffled future
    # actions from elsewhere in the batch, the predicted rollout must come
    # out *further* from the ground-truth latent trajectory than the rollout
    # under the true actions. This directly forces the predictor to make use
    # of the action signal.
    #
    # We use a log-sigmoid surrogate over per-step MSE:
    #   L_act = mean_t softplus(beta * (true_err_t - wrong_err_t))
    # which (a) always gives a non-zero gradient (no hard hinge saturation)
    # and (b) is roughly scale-invariant under a single `beta` because the
    # log term flattens once `true_err << wrong_err`. The optional margin
    # adds a target gap in raw MSE units before the softplus.
    contrast_loss = pred_loss.new_zeros(())
    contrast_margin_violation = pred_loss.new_zeros(())
    if (
        cfg.action_contrast_weight > 0.0
        and rollout_pred is not None
        and rollout_target is not None
        and actions.size(0) > 1
    ):
        ach = min(8, rollout_pred.size(1))
        if ach > 0:
            # Shuffle the future-action sequence across the batch dimension to
            # produce a "wrong action" matched to each visual history.
            batch_size = actions.size(0)
            perm = torch.randperm(batch_size, device=actions.device)
            # Ensure no element is mapped to itself (forces a real shuffle).
            collide = perm == torch.arange(batch_size, device=actions.device)
            if collide.any():
                roll = torch.arange(batch_size, device=actions.device).roll(1)
                perm = torch.where(collide, roll, perm)
            wrong_future = actions[perm, history_size - 1 : history_size - 1 + rollout_steps]
            hist_actions = torch.zeros_like(actions[:, :history_size])
            if history_size > 1:
                hist_actions[:, 1:] = actions[:, : history_size - 1]
            wrong_pred = _rollout_with_actions(
                model,
                pixels[:, :history_size],
                hist_actions,
                wrong_future,
                history_size=history_size,
            )
            # Per-sample, per-step MSE to the ground-truth latent target.
            true_err = (rollout_pred[:, :ach] - rollout_target[:, :ach]).pow(2).mean(dim=-1)
            wrong_err = (wrong_pred[:, :ach] - rollout_target[:, :ach]).pow(2).mean(dim=-1)
            # log(1 + exp(x)) with x = (true - wrong) / scale.
            # Use the mean wrong-action MSE as a per-batch scale so the loss
            # stays in O(1) regardless of where the predictor is in training.
            scale = wrong_err.detach().mean().clamp_min(1e-6)
            contrast_loss = torch.nn.functional.softplus((true_err - wrong_err) / scale).mean()
            contrast_margin_violation = (true_err - wrong_err).mean().detach()

    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = (
        pred_loss
        + cfg.rollout_loss_weight * rollout_loss
        + cfg.action_contrast_weight * contrast_loss
        + cfg.sigreg_weight * sigreg_loss
    )
    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "contrast_loss": contrast_loss.detach(),
        "contrast_signed_gap": contrast_margin_violation,
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
    # Track the lowest sliding-mean prediction loss observed so far so a
    # divergent late phase does not overwrite a healthy intermediate.
    pred_window: list[float] = []
    best_pred_mean = float("inf")
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
                    "contrast_loss": float(losses.get("contrast_loss", torch.zeros(()))),
                    "contrast_signed_gap": float(losses.get("contrast_signed_gap", torch.zeros(()))),
                    "sigreg_loss": float(losses["sigreg_loss"]),
                    "lr": float(optim.param_groups[0]["lr"]),
                }
                append_jsonl(train_log_path, last_log)
                pbar.set_postfix(
                    {
                        "loss": f"{last_log['loss']:.4f}",
                        "pred": f"{last_log['pred_loss']:.4f}",
                        "roll": f"{last_log['rollout_loss']:.4f}",
                        "ctr": f"{last_log['contrast_loss']:.4f}",
                        "gap": f"{last_log['contrast_signed_gap']:+.4f}",
                    }
                )
            # Sliding-mean prediction loss tracked at the same cadence as the
            # log entries so "best" updates without paying for an extra forward.
            if step % train_cfg.log_every == 0 or step == 1:
                pred_window.append(float(losses["pred_loss"].detach()))
                if len(pred_window) > 5:
                    pred_window.pop(0)
                pred_mean = sum(pred_window) / len(pred_window)
                if pred_mean < best_pred_mean and len(pred_window) >= 3:
                    best_pred_mean = pred_mean
                    best_path = ckpt_path.with_name("lewm_best.pt")
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "action_dim": model_cfg.action_dim,
                            "config": hparams,
                            "step": step,
                            "best_pred_mean": best_pred_mean,
                        },
                        best_path,
                    )
            if train_cfg.checkpoint_every > 0 and step % train_cfg.checkpoint_every == 0:
                latest_path = ckpt_path.with_name("lewm_latest.pt")
                step_path = ckpt_path.with_name(f"lewm_step{step:06d}.pt")
                latest_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "state_dict": model.state_dict(),
                    "action_dim": model_cfg.action_dim,
                    "config": hparams,
                    "step": step,
                }
                torch.save(payload, latest_path)
                torch.save(payload, step_path)
    pbar.close()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_path.with_name("lewm_final.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "action_dim": model_cfg.action_dim,
            "config": hparams,
            "step": step,
        },
        final_path,
    )
    # The "shipped" checkpoint (``ckpt_path``, default ``lewm.pt``) is the
    # **best** one we saw during training rather than the final weights.
    # Catastrophic divergence in the last few hundred steps would otherwise
    # silently overwrite a healthy model.
    best_path = ckpt_path.with_name("lewm_best.pt")
    if best_path.exists():
        import shutil as _shutil
        _shutil.copyfile(best_path, ckpt_path)
        print(f"[train] shipped checkpoint = {best_path.name} (best pred_mean) -> {ckpt_path}")
    else:
        # Fallback: no best was recorded (very short training); use the final.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "action_dim": model_cfg.action_dim,
                "config": hparams,
                "step": step,
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
