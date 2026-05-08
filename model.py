"""LeWM for walker locomotion.

LeWM architecture is intentionally kept identical to the upstream
`references/le-wm/jepa.py` + `module.py`:
- ViT-tiny encoder (HuggingFace `ViTModel`, trained from scratch — same as
  `stable_pretraining.backbone.utils.vit_hf` with `pretrained=False`).
- AdaLN-zero AR predictor with action conditioning.
- BatchNorm-MLP projector + pred_proj.
- SIGReg on latents.

Two extensions for goal-trajectory tracking:
- `criterion` aggregates MSE over the **whole** predicted vs goal latent
  trajectory (not just the final frame). The original LeWM only scored the
  last frame; for locomotion we want LeWM-CEM to track an entire reference
  trajectory.
- `cem_rollout_cost` exposes a batched, no-grad rollout that takes a
  `(B, S, T, action_dim)` tensor of action candidates plus a goal trajectory
  and returns per-candidate cost — the inner loop a LeWM-CEM solver needs.

Shape convention:
    pixels        (B, T, C, H, W)
    action        (B, T, action_dim)
    emb           (B, T, D)
    goal_emb      (B, T_goal, D)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

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
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN[-1].weight, 0.0)
        nn.init.constant_(self.adaLN[-1].bias, 0.0)

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
            [_ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
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
        diff = (pred_emb - goal_emb.detach()).pow(2).mean(dim=-1)  # (B, S, T)
        if weights is not None:
            w = weights[:T].to(diff)
            diff = diff * w
        return diff.mean(dim=-1)

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
        history_size: int | None = None,
    ) -> torch.Tensor:
        """Rollout + trajectory cost in one call. Returns (B, S)."""
        pred = self.rollout_latents(history_pixels, history_actions, future_actions, history_size)
        info = {"predicted_emb": pred, "goal_emb": goal_latents}
        if weights is not None:
            info["traj_weights"] = weights
        return self.criterion(info)


# ---------------------------------------------------------------------------
# Loss helper used by the training script.
# ---------------------------------------------------------------------------


def lewm_loss(
    model: JEPA,
    sigreg: SIGReg,
    pixels: torch.Tensor,
    action: torch.Tensor,
    history_size: int,
    num_preds: int,
    sigreg_weight: float = 0.09,
) -> dict:
    """Compute LeWM training loss on a sliding window batch.

    pixels: (B, T, C, H, W) with T == history_size + num_preds.
    action: (B, T, action_dim).
    """
    action = torch.nan_to_num(action, 0.0)
    info = {"pixels": pixels, "action": action}
    info = model.encode(info)

    emb = info["emb"]          # (B, T, D)
    act_emb = info["act_emb"]  # (B, T, D)

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + sigreg_weight * sigreg_loss

    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
    }


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
