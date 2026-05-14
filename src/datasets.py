"""Task-agnostic offline data collection for LeWM training.

The whole module is oracle-free: no task knows what a "good" action is, and the
exploration policies are pure functions of (step, prev_action, RNG). All
policy parameters (frequencies, time constants, segment lengths) are sampled
from the per-episode RNG, so changing the master seed shifts the entire
distribution.

The exported API is:

- ``ExplorationPolicy`` — protocol-style base class for a stateless,
  per-episode action generator.
- A small set of generic policies suitable for locomotion-style continuous
  control: OU noise, sinusoidal mixtures, smoothed piecewise-constant,
  multi-sine. None of them know which task they are running on.
- ``sample_policy(rng, action_dim, episode_steps)`` — picks one policy at
  random and instantiates it.
- ``collect_dataset(...)`` — drives an env with a sampled policy, writes
  per-episode ``.npz`` files, and a ``metadata.json`` summary.

Task adapters only have to provide ``env_fn`` (seed -> dm_control env) and a
``record_step`` callback that turns the live env into per-step metadata such
as ``torso_height`` / ``reward``. The policies and the collection loop never
see the task.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import imageio.v2 as imageio
import numpy as np


# ---------------------------------------------------------------------------
# Exploration policies. Each policy is a closure over per-episode parameters
# sampled from the per-episode RNG; the ``__call__`` signature is uniform.
# ---------------------------------------------------------------------------


@dataclass
class ExplorationPolicy:
    name: str
    params: dict

    def __call__(self, step: int, prev_action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


def _clip(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(a, low), high)


@dataclass
class OUNoisePolicy(ExplorationPolicy):
    """a_t = rho * a_{t-1} + sigma * N(0, 1), with action bounds."""

    low: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __call__(self, step, prev_action, rng):
        rho = self.params["rho"]
        sigma = self.params["sigma"]
        noise = rng.normal(0.0, sigma, size=prev_action.shape).astype(np.float32)
        action = rho * prev_action + noise
        return _clip(action, self.low, self.high)


@dataclass
class SineMixPolicy(ExplorationPolicy):
    """Per-dim sine: amp * sin(2*pi*freq*step*dt + phase)."""

    low: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __call__(self, step, prev_action, rng):
        freqs = self.params["freqs"]
        phases = self.params["phases"]
        amps = self.params["amps"]
        bias = self.params["bias"]
        t = step * self.params["dt"]
        action = amps * np.sin(2.0 * np.pi * freqs * t + phases) + bias
        return _clip(action.astype(np.float32), self.low, self.high)


@dataclass
class SmoothPiecewisePolicy(ExplorationPolicy):
    """Hold a random target for k steps, smooth-transition to next target."""

    low: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def _target_for(self, segment_idx: int, rng_local: np.random.Generator) -> np.ndarray:
        amp = self.params["target_amp"]
        return rng_local.uniform(-amp, amp, size=self.low.shape).astype(np.float32)

    def __call__(self, step, prev_action, rng):
        # We deterministically derive targets from the *episode* seed (params["seed"])
        # rather than the live rng, so the policy is reproducible without state.
        seg_len = int(self.params["segment_len"])
        smooth_steps = int(self.params["smooth_steps"])
        ep_seed = int(self.params["seed"])
        seg_idx = step // seg_len
        within = step % seg_len
        local = np.random.default_rng(ep_seed * 1_000_003 + seg_idx)
        target_curr = self._target_for(seg_idx, local)
        if within < smooth_steps and seg_idx > 0:
            prev_local = np.random.default_rng(ep_seed * 1_000_003 + (seg_idx - 1))
            target_prev = self._target_for(seg_idx - 1, prev_local)
            alpha = (within + 1) / float(smooth_steps + 1)
            target = (1.0 - alpha) * target_prev + alpha * target_curr
        else:
            target = target_curr
        # Low-pass against prev_action for extra continuity.
        beta = self.params["lowpass"]
        action = beta * prev_action + (1.0 - beta) * target
        return _clip(action.astype(np.float32), self.low, self.high)


@dataclass
class MultiSinePolicy(ExplorationPolicy):
    """Sum of K sines per action dimension, all params from the rng."""

    low: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __call__(self, step, prev_action, rng):
        freqs = self.params["freqs"]      # (K, dim)
        phases = self.params["phases"]    # (K, dim)
        amps = self.params["amps"]        # (K, dim)
        bias = self.params["bias"]        # (dim,)
        t = step * self.params["dt"]
        components = amps * np.sin(2.0 * np.pi * freqs * t + phases)
        action = components.sum(axis=0) + bias
        return _clip(action.astype(np.float32), self.low, self.high)


@dataclass
class SineOUMixPolicy(ExplorationPolicy):
    """Sinusoidal rhythm + small OU perturbation around it."""

    low: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __call__(self, step, prev_action, rng):
        freqs = self.params["freqs"]
        phases = self.params["phases"]
        amps = self.params["amps"]
        bias = self.params["bias"]
        t = step * self.params["dt"]
        rhythm = amps * np.sin(2.0 * np.pi * freqs * t + phases) + bias
        # OU perturbation on top, kept small (action_dim units).
        rho = self.params["rho"]
        sigma = self.params["sigma"]
        delta = prev_action - rhythm
        new_delta = rho * delta + rng.normal(0.0, sigma, size=delta.shape).astype(np.float32)
        action = rhythm + new_delta
        return _clip(action.astype(np.float32), self.low, self.high)


def sample_policy(
    rng: np.random.Generator,
    action_dim: int,
    episode_steps: int,
    dt: float,
    low: np.ndarray,
    high: np.ndarray,
    ep_seed: int,
) -> ExplorationPolicy:
    """Pick a generic exploration policy with parameters sampled from rng.

    All policies produce time-coherent actions that keep the agent active
    (no uniform-iid policy that just makes the agent collapse). Parameters
    are biased toward "large, rhythmic, varied" so the agent is forced to
    *do something* every episode.
    """
    kind = rng.choice([
        "ou",
        "sine",
        "smooth_piecewise",
        "multi_sine",
        "sine_ou_mix",
    ])
    amp_scale = (high - low) / 2.0  # half-range per actuator
    centre = (high + low) / 2.0
    # A useful generic prior for locomotion-style tasks: many systems have
    # paired symmetric actuators (left/right hip, knee, ankle). Putting
    # paired dims in **anti-phase** produces step-like rhythm. We don't
    # know which dim is paired with which, so we randomly try one of three
    # task-agnostic pairings per episode:
    #   - "adjacent": (0,1), (2,3), (4,5), ...
    #   - "split":    (0, H), (1, H+1), ... with H = action_dim // 2
    #   - "none":     keep random phases.
    pairing = rng.choice(["adjacent", "split", "none"])
    half = action_dim // 2

    def _maybe_antiphase(phases: np.ndarray) -> np.ndarray:
        if pairing == "none":
            return phases
        out = phases.copy()
        if pairing == "adjacent":
            for k in range(action_dim // 2):
                out[2 * k + 1] = (out[2 * k] + np.pi) % (2 * np.pi)
        elif pairing == "split":
            for k in range(half):
                out[half + k] = (out[k] + np.pi) % (2 * np.pi)
        return out

    if kind == "ou":
        rho = float(rng.uniform(0.75, 0.96))
        sigma = float(rng.uniform(0.3, 0.8)) * amp_scale.mean()
        return OUNoisePolicy(
            name="ou",
            params={"rho": rho, "sigma": sigma},
            low=low, high=high,
        )
    if kind == "sine":
        freqs = rng.uniform(0.5, 3.0, size=action_dim).astype(np.float32)
        phases = _maybe_antiphase(rng.uniform(0.0, 2 * np.pi, size=action_dim)).astype(np.float32)
        amps = (amp_scale * rng.uniform(0.8, 1.4, size=action_dim)).astype(np.float32)
        bias = (centre + amp_scale * rng.uniform(-0.15, 0.15, size=action_dim)).astype(np.float32)
        return SineMixPolicy(
            name="sine",
            params={"freqs": freqs, "phases": phases, "amps": amps, "bias": bias, "dt": dt},
            low=low, high=high,
        )
    if kind == "smooth_piecewise":
        seg_len = int(rng.integers(10, 41))
        smooth_steps = int(rng.integers(2, 6))
        lowpass = float(rng.uniform(0.0, 0.3))
        target_amp = float(rng.uniform(0.8, 1.2))
        return SmoothPiecewisePolicy(
            name="smooth_piecewise",
            params={
                "segment_len": seg_len,
                "smooth_steps": smooth_steps,
                "lowpass": lowpass,
                "target_amp": target_amp,
                "seed": ep_seed,
            },
            low=low, high=high,
        )
    if kind == "multi_sine":
        K = int(rng.integers(2, 4))
        # Make one of the K frequencies share a common base across dims so the
        # whole limb set "breathes" together (still random in absolute value).
        common_freq = float(rng.uniform(0.6, 2.5))
        freqs = rng.uniform(0.4, 3.5, size=(K, action_dim)).astype(np.float32)
        freqs[0] = common_freq
        phases = rng.uniform(0.0, 2 * np.pi, size=(K, action_dim)).astype(np.float32)
        phases[0] = _maybe_antiphase(phases[0]).astype(np.float32)
        weights = rng.uniform(0.3, 1.0, size=(K, action_dim)).astype(np.float32)
        weights = weights / weights.sum(axis=0, keepdims=True)
        amps = (amp_scale * rng.uniform(0.8, 1.4, size=action_dim))[None, :] * weights
        bias = (centre + amp_scale * rng.uniform(-0.15, 0.15, size=action_dim)).astype(np.float32)
        return MultiSinePolicy(
            name="multi_sine",
            params={"freqs": freqs, "phases": phases, "amps": amps.astype(np.float32), "bias": bias, "dt": dt},
            low=low, high=high,
        )
    if kind == "sine_ou_mix":
        freqs = rng.uniform(0.5, 2.5, size=action_dim).astype(np.float32)
        phases = _maybe_antiphase(rng.uniform(0.0, 2 * np.pi, size=action_dim)).astype(np.float32)
        amps = (amp_scale * rng.uniform(0.7, 1.2, size=action_dim)).astype(np.float32)
        bias = (centre + amp_scale * rng.uniform(-0.15, 0.15, size=action_dim)).astype(np.float32)
        rho = float(rng.uniform(0.7, 0.92))
        sigma = float(rng.uniform(0.05, 0.2)) * amp_scale.mean()
        return SineOUMixPolicy(
            name="sine_ou_mix",
            params={"freqs": freqs, "phases": phases, "amps": amps, "bias": bias, "dt": dt, "rho": rho, "sigma": sigma},
            low=low, high=high,
        )
    raise RuntimeError(f"unreachable: {kind}")


# ---------------------------------------------------------------------------
# Task registry.  A *task* exposes (env_fn, render_fn, action_bounds_fn,
# record_fn, dt) under a module-level name so worker processes can re-import
# the module and look up the functions instead of pickling closures across
# the process boundary.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskHandles:
    env_fn: Callable[[int], "object"]
    render_fn: Callable[["object"], np.ndarray]
    action_bounds_fn: Callable[["object"], tuple[np.ndarray, np.ndarray]]
    record_fn: Callable[["object"], dict[str, float]]
    dt: float


_TASK_REGISTRY: dict[str, TaskHandles] = {}


def register_task(name: str, handles: TaskHandles) -> None:
    _TASK_REGISTRY[name] = handles


def _resolve_task(name: str, module_path: str) -> TaskHandles:
    """Import the task module if needed, then look it up in the registry."""
    if name not in _TASK_REGISTRY:
        import importlib
        importlib.import_module(module_path)
    return _TASK_REGISTRY[name]


@dataclass(frozen=True)
class CollectConfig:
    n_episodes: int = 128
    episode_steps: int = 400
    workers: int = 8
    seed: int = 42
    mp_start_method: str = "forkserver"


def _collect_one_episode(handles: TaskHandles, *, ep_idx: int, base_seed: int, episode_steps: int) -> dict:
    """Run a single episode under a random policy. Returns the saved arrays."""
    ep_seed = base_seed + ep_idx
    rng = np.random.default_rng(ep_seed)
    env = handles.env_fn(ep_seed)
    env.reset()
    low, high = handles.action_bounds_fn(env)
    action_dim = int(low.size)
    policy = sample_policy(
        rng=rng,
        action_dim=action_dim,
        episode_steps=episode_steps,
        dt=handles.dt,
        low=low,
        high=high,
        ep_seed=ep_seed,
    )
    prev_action = ((low + high) / 2.0).astype(np.float32)

    pixels = [handles.render_fn(env)]
    actions = []
    rewards = []
    metrics_acc: dict[str, list[float]] = {}

    for step in range(episode_steps):
        action = policy(step, prev_action, rng).astype(np.float32)
        ts = env.step(action)
        actions.append(action)
        rewards.append(float(ts.reward) if ts.reward is not None else 0.0)
        pixels.append(handles.render_fn(env))
        prev_action = action
        for key, value in handles.record_fn(env).items():
            metrics_acc.setdefault(key, []).append(float(value))

    record = {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }
    for key, values in metrics_acc.items():
        record[key] = np.asarray(values, dtype=np.float32)
    record["metadata"] = np.asarray(
        json.dumps({
            "ep_idx": ep_idx,
            "ep_seed": ep_seed,
            "policy": policy.name,
            "episode_steps": episode_steps,
        }),
    )
    return record


def _worker_target(payload: dict) -> tuple[int, dict]:
    handles = _resolve_task(payload["task_name"], payload["task_module"])
    record = _collect_one_episode(
        handles,
        ep_idx=payload["ep_idx"],
        base_seed=payload["base_seed"],
        episode_steps=payload["episode_steps"],
    )
    ep_idx = payload["ep_idx"]
    out_path = Path(payload["out_dir"]) / f"episode_{ep_idx:04d}.npz"
    np.savez_compressed(out_path, **record)
    summary = {
        "ep_idx": ep_idx,
        "policy": json.loads(str(record["metadata"]))["policy"],
        "ep_seed": payload["base_seed"] + ep_idx,
        "return": float(record["rewards"].sum()),
        "mean_reward": float(record["rewards"].mean()),
    }
    for key in ("torso_height", "torso_upright", "horizontal_velocity"):
        if key in record:
            summary[f"mean_{key}"] = float(record[key].mean())
            summary[f"max_{key}"] = float(record[key].max())
            summary[f"min_{key}"] = float(record[key].min())
    return ep_idx, summary


def collect_dataset(
    *,
    out_dir: Path,
    task_name: str,
    task_module: str,
    config: CollectConfig,
    extra_metadata: dict | None = None,
) -> dict:
    """Generate ``config.n_episodes`` episodes in parallel.

    The task is identified by ``task_name`` (key in ``_TASK_REGISTRY``) and
    ``task_module`` (importable Python path). Worker processes resolve the
    task by re-importing the module, so no closures cross the boundary.

    Returns a dict with per-episode summaries; also writes
    ``out_dir/metadata.json``.
    """
    handles = _resolve_task(task_name, task_module)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        {
            "ep_idx": ep_idx,
            "base_seed": config.seed,
            "episode_steps": config.episode_steps,
            "task_name": task_name,
            "task_module": task_module,
            "out_dir": str(out_dir),
        }
        for ep_idx in range(config.n_episodes)
    ]

    summaries: list[dict] = []
    if config.workers <= 1:
        for payload in payloads:
            _, summary = _worker_target(payload)
            summaries.append(summary)
            print(f"[datasets] episode {summary['ep_idx']:03d} policy={summary['policy']:>18s} "
                  f"return={summary['return']:.2f}")
    else:
        ctx = mp.get_context(config.mp_start_method)
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=ctx) as ex:
            for ep_idx, summary in ex.map(_worker_target, payloads):
                summaries.append(summary)
                print(f"[datasets] episode {ep_idx:03d} policy={summary['policy']:>18s} "
                      f"return={summary['return']:.2f}")
    summaries.sort(key=lambda s: s["ep_idx"])

    metadata = {
        "config": {
            "n_episodes": config.n_episodes,
            "episode_steps": config.episode_steps,
            "workers": config.workers,
            "seed": config.seed,
            "mp_start_method": config.mp_start_method,
            "task_name": task_name,
            "task_module": task_module,
            "dt": handles.dt,
        },
        "episodes": summaries,
    }
    if extra_metadata is not None:
        metadata["extra"] = extra_metadata
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------------------
# Goal-manifold rendering (post-hoc goal-frame selection + mp4 + json log).
#
# The collection loop is policy-agnostic; the goal manifold is whatever frames
# the task adapter labels as "goal" via a ``goal_mask_fn``. Rendering the
# resulting frame set as a video makes it easy to eyeball whether the
# automatic selection produced anything resembling the desired behaviour.
# ---------------------------------------------------------------------------


GoalMaskFn = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class GoalManifoldStats:
    total_frames: int
    total_segments: int
    episodes_contributing: int
    per_episode: list[dict]
    metrics_summary: dict[str, dict[str, float]]


def _goal_contiguous_segments(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Return a list of ``(start, end_exclusive)`` for True runs of length >= min_len."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    out = []
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            out.append((int(s), int(e)))
    return out


def render_goal_manifold(
    *,
    dataset_dir: Path,
    out_dir: Path,
    goal_mask_fn: GoalMaskFn,
    video_fps: int = 30,
    min_segment_len: int = 1,
    metric_keys: tuple[str, ...] = ("torso_height", "torso_upright", "horizontal_velocity", "rewards"),
) -> GoalManifoldStats:
    """Write ``goal_manifold.mp4`` and ``goal_manifold_log.json`` summarising
    every contiguous "goal" segment across the dataset. The mp4 is rendered
    at the dataset's native pixel resolution.
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(dataset_dir.glob("episode_*.npz"))
    if not paths:
        raise RuntimeError(f"no episodes in {dataset_dir}")

    log_episodes: list[dict] = []
    metrics_accum: dict[str, list[float]] = {}
    total_frames = 0
    total_segments = 0
    episodes_contributing = 0

    video_path = out_dir / "goal_manifold.mp4"
    log_path = out_dir / "goal_manifold_log.json"
    writer = imageio.get_writer(video_path, fps=video_fps, codec="libx264", quality=8)
    try:
        for path in paths:
            with np.load(path) as ep_npz:
                ep = {k: ep_npz[k] for k in ep_npz.files}
            mask = goal_mask_fn(ep).astype(bool)
            segments = _goal_contiguous_segments(mask, min_len=min_segment_len)
            if not segments:
                continue
            episodes_contributing += 1
            ep_log = {
                "path": str(path),
                "segments": [],
                "total_goal_frames": int(mask.sum()),
            }
            for seg_start, seg_end in segments:
                seg_pixels = ep["pixels"][seg_start:seg_end]
                for frame in seg_pixels:
                    writer.append_data(frame)
                seg_log = {
                    "start": seg_start,
                    "end": seg_end,
                    "length": seg_end - seg_start,
                }
                for key in metric_keys:
                    if key in ep:
                        arr = np.asarray(ep[key])
                        m_start = min(seg_start, len(arr))
                        m_end = min(seg_end, len(arr))
                        if m_end > m_start:
                            slice_ = arr[m_start:m_end]
                            seg_log[f"{key}_mean"] = float(slice_.mean())
                            seg_log[f"{key}_min"] = float(slice_.min())
                            seg_log[f"{key}_max"] = float(slice_.max())
                            metrics_accum.setdefault(key, []).extend(slice_.tolist())
                total_frames += seg_end - seg_start
                total_segments += 1
                ep_log["segments"].append(seg_log)
            log_episodes.append(ep_log)
    finally:
        writer.close()

    metrics_summary: dict[str, dict[str, float]] = {}
    for key, values in metrics_accum.items():
        arr = np.asarray(values, dtype=np.float64)
        metrics_summary[key] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    stats = GoalManifoldStats(
        total_frames=total_frames,
        total_segments=total_segments,
        episodes_contributing=episodes_contributing,
        per_episode=log_episodes,
        metrics_summary=metrics_summary,
    )
    log = {
        "video": str(video_path),
        "dataset_dir": str(dataset_dir),
        "total_frames": total_frames,
        "total_segments": total_segments,
        "episodes_contributing": episodes_contributing,
        "episodes_total": len(paths),
        "metrics_summary": metrics_summary,
        "per_episode": log_episodes,
    }
    log_path.write_text(json.dumps(log, indent=2))
    print(
        f"[goal-manifold] frames={total_frames} segments={total_segments} "
        f"episodes={episodes_contributing}/{len(paths)} "
        f"-> {video_path}"
    )
    return stats


# ---------------------------------------------------------------------------
# Oracle CEM with simulator state-restoring rollouts.
#
# Used **only** to produce a single "goal demo" trajectory that begins from
# the env reset pose and reaches a successful state. The N explore
# episodes that LeWM trains on remain oracle-free; this is the "+1" in the
# "N + 1" dataset. The algorithm is task-agnostic: tasks inject a
# ``score_fn`` callable that scores a candidate action sequence by
# replaying it in a state-restoring oracle env. Everything else (CEM
# optimisation, episode driver, per-step metric recording) lives here.
# ---------------------------------------------------------------------------


# ``score_fn(oracle_env, state0, actions, previous_action) -> float`` — higher is better.
ScoreFn = Callable[[object, np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class OracleCEMConfig:
    plan_horizon: int = 40
    samples: int = 512
    topk: int = 64
    iters: int = 6
    init_std: float = 0.55
    min_std: float = 0.06
    momentum: float = 0.15


def oracle_cem_plan(
    oracle_env,
    state0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    init_mean: np.ndarray,
    previous_action: np.ndarray,
    rng: np.random.Generator,
    score_fn: ScoreFn,
    cfg: OracleCEMConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(best_first_action, refined_mean, best_score)``."""
    action_dim = low.shape[0]
    mean = init_mean.astype(np.float32).copy()
    std = np.full_like(mean, cfg.init_std, dtype=np.float32)
    topk = min(cfg.topk, cfg.samples)
    best_plan: np.ndarray | None = None
    best_score = -float("inf")
    for _ in range(cfg.iters):
        samples = rng.normal(
            loc=mean[None],
            scale=std[None],
            size=(cfg.samples, cfg.plan_horizon, action_dim),
        ).astype(np.float32)
        samples[0] = mean
        samples = np.clip(samples, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
        scores = np.empty(samples.shape[0], dtype=np.float32)
        for i in range(samples.shape[0]):
            scores[i] = score_fn(oracle_env, state0, samples[i], previous_action)
        elite_idx = np.argpartition(-scores, topk - 1)[:topk]
        elites = samples[elite_idx]
        elite_mean = elites.mean(axis=0).astype(np.float32)
        elite_std = np.maximum(elites.std(axis=0), cfg.min_std).astype(np.float32)
        mean = (1.0 - cfg.momentum) * elite_mean + cfg.momentum * mean
        std = elite_std
        i_best = int(np.argmax(scores))
        if float(scores[i_best]) > best_score:
            best_score = float(scores[i_best])
            best_plan = samples[i_best].copy()
    assert best_plan is not None
    return best_plan[0].astype(np.float32), mean.astype(np.float32), best_score


def collect_oracle_goal_demo(
    *,
    env_fn: Callable[[int], object],
    action_bounds_fn: Callable[[object], tuple[np.ndarray, np.ndarray]],
    render_fn: Callable[[object], np.ndarray],
    record_fn: Callable[[object], dict[str, float]],
    score_fn: ScoreFn,
    episode_steps: int,
    seed: int,
    cfg: OracleCEMConfig = OracleCEMConfig(),
    domain: str = "",
    task: str = "",
    desc: str = "oracle goal",
) -> dict[str, np.ndarray]:
    """Run one episode under oracle CEM and return the recorded arrays.

    The returned dict has the same schema as the explore episodes:
    ``pixels``, ``actions``, ``rewards``, plus any per-step scalar metrics
    from ``record_fn`` (e.g. ``torso_height``), and a JSON ``metadata`` blob.
    """
    rng = np.random.default_rng(seed)
    env = env_fn(seed)
    oracle_env = env_fn(seed + 10_000)
    low, high = action_bounds_fn(env)
    action_dim = int(low.shape[0])
    env.reset()
    plan_mean = np.zeros((cfg.plan_horizon, action_dim), dtype=np.float32)
    previous_action = np.zeros(action_dim, dtype=np.float32)

    pixels = [render_fn(env)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    metrics_acc: dict[str, list[float]] = {}

    from tqdm import tqdm  # local import to avoid leaking into module-level api.

    bar = tqdm(range(episode_steps), desc=desc, dynamic_ncols=True)
    for _step in bar:
        state0 = env.physics.get_state().copy()
        action, plan_mean, _score = oracle_cem_plan(
            oracle_env, state0, low, high, plan_mean, previous_action, rng, score_fn, cfg
        )
        plan_mean = np.concatenate(
            [plan_mean[1:], np.zeros((1, action_dim), dtype=np.float32)],
            axis=0,
        )
        ts = env.step(action)
        pixels.append(render_fn(env))
        actions.append(action.copy())
        rewards.append(float(ts.reward or 0.0))
        for key, value in record_fn(env).items():
            metrics_acc.setdefault(key, []).append(float(value))
        previous_action = action
        postfix = {"ret": f"{sum(rewards):.1f}"}
        latest = {k: v[-1] for k, v in metrics_acc.items() if v}
        for k, v in latest.items():
            postfix[k[:8]] = f"{v:.2f}"
        bar.set_postfix(postfix)
        if ts.last():
            break
    bar.close()
    env.close()
    oracle_env.close()

    record: dict[str, np.ndarray] = {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }
    for key, values in metrics_acc.items():
        record[key] = np.asarray(values, dtype=np.float32)
    record["metadata"] = np.asarray(
        json.dumps(
            {
                "kind": "oracle_goal_demo",
                "domain": domain,
                "task": task,
                "seed": seed,
                "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
            }
        )
    )
    return record
