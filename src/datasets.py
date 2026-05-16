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

import gymnasium
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
# SAC-driven dataset producer.
#
# Trains (or loads from cache) a stable-baselines3 SAC policy on the
# dm_control task using the low-dimensional physics state as observation,
# then rolls N episodes with three perturbation recipes (stable /
# action_noise / warmup_random) to produce a visual replay dataset that
# covers stable-walking, mid-rollout recovery, and atypical-start recovery
# in the same .npz schema as the (deleted) random-explore episodes.
# The trained policy is cached at ``out_dir/sac/policy.zip`` so that tuning
# perturbation knobs doesn't require retraining.
# ---------------------------------------------------------------------------


class _DmcGymEnv(gymnasium.Env):
    """Minimal Gymnasium env wrapping a dm_control suite.Env.

    Observation = concatenated qpos + qvel. Reward = the task's env reward.
    SB3 expects ``reset(seed=...) -> (obs, info)`` and
    ``step(action) -> (obs, reward, terminated, truncated, info)``.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, env_fn, seed: int, episode_steps: int):
        super().__init__()
        self._env = env_fn(seed)
        self._env.reset()
        low, high = (
            np.asarray(self._env.action_spec().minimum, dtype=np.float32),
            np.asarray(self._env.action_spec().maximum, dtype=np.float32),
        )
        self.action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        obs = self._obs()
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self._steps = 0
        self._max_steps = int(episode_steps)
        self._seed = seed

    def _obs(self) -> np.ndarray:
        d = self._env.physics.data
        return np.concatenate([d.qpos.ravel(), d.qvel.ravel()]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        self._env.reset()
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        ts = self._env.step(np.asarray(action, dtype=np.float32))
        self._steps += 1
        terminated = bool(ts.last())
        truncated = (not terminated) and (self._steps >= self._max_steps)
        return self._obs(), float(ts.reward or 0.0), terminated, truncated, {}

    def close(self):
        self._env.close()


@dataclass(frozen=True)
class PerturbedRolloutConfig:
    """Severity-driven OU-burst perturbed rollout config.

    Every episode draws ``severity ~ Beta(beta_a, beta_b)`` (default
    Beta(5,2): mean ~0.71, right-skewed) and uses it to scale a single
    OU-correlated noise burst injected mid-rollout, plus an optional
    random-action warmup that drops the walker into a fallen pose for the
    SAC policy to recover from. Reference implementation:
    ``references/noisy_cem_walker_walk.py:86-95, 343-352``.
    """
    n_episodes: int = 128
    severity_beta_a: float = 5.0
    severity_beta_b: float = 2.0
    warmup_steps_max: int = 100
    push_burst_sigma_max: float = 0.7
    push_burst_len: int = 12
    push_burst_window: tuple[int, int] = (60, 200)
    # Burst always starts at >= warmup_steps + buffer, so even high-severity
    # episodes give SAC at least this many steps to take over before the
    # push hits.
    burst_after_warmup_buffer: int = 10
    # OU noise correlation timescale (seconds). rho = exp(-dt/tau).
    ou_tau: float = 0.15


def train_or_load_sac(
    *,
    env_fn: Callable[[int], object],
    episode_steps: int,
    sac_timesteps: int,
    seed: int,
    out_dir: Path,
    desc: str = "SAC",
):
    """Load ``out_dir/sac/policy.zip`` if present, otherwise train and save."""
    from stable_baselines3 import SAC

    out_dir = Path(out_dir)
    policy_path = out_dir / "sac" / "policy.zip"
    if policy_path.exists():
        print(f"[{desc}] loading cached policy <- {policy_path}")
        return SAC.load(policy_path)
    print(f"[{desc}] training SAC for {sac_timesteps} steps...")
    (out_dir / "sac").mkdir(parents=True, exist_ok=True)
    train_env = _DmcGymEnv(env_fn, seed, episode_steps)
    model = SAC("MlpPolicy", train_env, seed=seed, verbose=0)
    model.learn(total_timesteps=int(sac_timesteps), progress_bar=True)
    model.save(policy_path.with_suffix(""))  # SAC.save appends ".zip"
    train_env.close()
    return model


def rollout_sac_episode(
    *,
    model,
    env_fn: Callable[[int], object],
    action_bounds_fn: Callable[[object], tuple[np.ndarray, np.ndarray]],
    render_fn: Callable[[object], np.ndarray],
    record_fn: Callable[[object], dict[str, float]],
    control_timestep: float,
    episode_steps: int,
    seed: int,
    rng: np.random.Generator,
    random_warmup_steps: int = 0,
    push_burst_start: int = 0,
    push_burst_len: int = 0,
    push_burst_sigma: float = 0.0,
    ou_tau: float = 0.15,
    domain: str = "",
    task: str = "",
    kind: str = "sac_rollout",
    severity: float | None = None,
) -> dict[str, np.ndarray]:
    """One SAC-driven episode with optional warmup + OU-correlated burst.

    Behavior:
    * For steps ``t < random_warmup_steps``: action is uniform random in
      ``[low, high]``. Drops the agent into atypical/fallen poses.
    * For steps ``push_burst_start <= t < push_burst_start + push_burst_len``:
      action = SAC(obs) + ``push_burst_sigma * eta`` where ``eta`` evolves as
      ``eta_{t+1} = rho * eta_t + sqrt(1 - rho^2) * N(0, I)`` with
      ``rho = exp(-dt/ou_tau)``. ``eta`` is reset to 0 at burst onset.
      OU correlation makes the burst feel like a sustained push rather
      than canceled white noise.
    * All other steps: deterministic SAC policy.

    NPZ schema matches what ``collect_dataset`` used to produce.
    """
    env = env_fn(seed)
    env.reset()
    low, high = action_bounds_fn(env)
    action_dim = int(low.shape[0])

    ou_rho = float(np.exp(-control_timestep / ou_tau)) if ou_tau > 0 else 0.0
    ou_scale = float(np.sqrt(max(1.0 - ou_rho * ou_rho, 0.0)))
    eta = np.zeros(action_dim, dtype=np.float32)

    pixels = [render_fn(env)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    metrics_acc: dict[str, list[float]] = {}

    for t in range(episode_steps):
        if t < random_warmup_steps:
            action = rng.uniform(low, high).astype(np.float32)
        else:
            obs = np.concatenate(
                [env.physics.data.qpos.ravel(), env.physics.data.qvel.ravel()]
            ).astype(np.float32)
            predicted, _ = model.predict(obs, deterministic=True)
            sac_action = predicted.astype(np.float32)
            in_burst = (
                push_burst_len > 0
                and push_burst_sigma > 0
                and push_burst_start <= t < push_burst_start + push_burst_len
            )
            if in_burst:
                if t == push_burst_start:
                    eta[:] = 0.0
                xi = rng.standard_normal(action_dim).astype(np.float32)
                eta = ou_rho * eta + ou_scale * xi
                action = sac_action + push_burst_sigma * eta
            else:
                eta[:] = 0.0
                action = sac_action
        action = np.clip(action, low, high).astype(np.float32)
        ts = env.step(action)
        actions.append(action.copy())
        rewards.append(float(ts.reward or 0.0))
        pixels.append(render_fn(env))
        for key, value in record_fn(env).items():
            metrics_acc.setdefault(key, []).append(float(value))
        if ts.last():
            break
    env.close()

    record: dict[str, np.ndarray] = {
        "pixels": np.stack(pixels, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }
    for key, values in metrics_acc.items():
        record[key] = np.asarray(values, dtype=np.float32)
    record["metadata"] = np.asarray(
        json.dumps({
            "kind": kind,
            "domain": domain,
            "task": task,
            "seed": int(seed),
            "severity": float(severity) if severity is not None else None,
            "random_warmup_steps": int(random_warmup_steps),
            "push_burst_start": int(push_burst_start),
            "push_burst_len": int(push_burst_len),
            "push_burst_sigma": float(push_burst_sigma),
            "ou_tau": float(ou_tau),
            "ou_rho": float(ou_rho),
            "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
        })
    )
    return record


def collect_sac_perturbed_dataset(
    *,
    env_fn: Callable[[int], object],
    action_bounds_fn: Callable[[object], tuple[np.ndarray, np.ndarray]],
    render_fn: Callable[[object], np.ndarray],
    record_fn: Callable[[object], dict[str, float]],
    control_timestep: float,
    episode_steps: int,
    seed: int,
    out_dir: Path,
    dataset_dir: Path,
    sac_timesteps: int,
    cfg: PerturbedRolloutConfig,
    domain: str = "",
    task: str = "",
) -> dict:
    """Train (or load cached) SAC then roll ``cfg.n_episodes`` perturbed
    episodes. Per-episode: draw ``severity ~ Beta(beta_a, beta_b)`` and use
    it to scale warmup length and OU-burst sigma; burst start is uniform
    over ``[warmup_steps + burst_after_warmup_buffer, push_burst_window[1])``
    so the burst always lands after the SAC policy has taken over.

    All episodes are stacked into a single ``perturbed.npz`` (batch
    shape ``(N, T+1, H, W, 3)`` for pixels, ``(N, T, A)`` for actions,
    etc). Progress is displayed with a rich progress bar matching the
    SAC training bar's style.
    """
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    model = train_or_load_sac(
        env_fn=env_fn,
        episode_steps=episode_steps,
        sac_timesteps=sac_timesteps,
        seed=seed,
        out_dir=out_dir,
        desc=f"SAC[{task or 'task'}]",
    )

    rng = np.random.default_rng(seed)
    _, burst_hi = cfg.push_burst_window
    burst_hi_cap = max(1, min(burst_hi, episode_steps - cfg.push_burst_len - 1))

    pixels_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[np.ndarray] = []
    metric_lists: dict[str, list[np.ndarray]] = {}
    summaries: list[dict] = []

    progress = Progress(
        TextColumn(f"[bold cyan]perturbed[{task or 'task'}][/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("[dim]• {task.fields[ep_info]}[/dim]"),
    )
    with progress:
        task_id = progress.add_task("rollout", total=cfg.n_episodes, ep_info="")
        for i in range(cfg.n_episodes):
            severity = float(rng.beta(cfg.severity_beta_a, cfg.severity_beta_b))
            warmup = int(round(severity * severity * cfg.warmup_steps_max))
            burst_sigma = severity * cfg.push_burst_sigma_max
            burst_lo_eff = max(warmup + cfg.burst_after_warmup_buffer, 0)
            burst_lo_eff = min(burst_lo_eff, burst_hi_cap - 1)
            burst_start = int(rng.integers(burst_lo_eff, burst_hi_cap))

            ep = rollout_sac_episode(
                model=model,
                env_fn=env_fn,
                action_bounds_fn=action_bounds_fn,
                render_fn=render_fn,
                record_fn=record_fn,
                control_timestep=control_timestep,
                episode_steps=episode_steps,
                seed=seed + 1 + i,
                rng=rng,
                random_warmup_steps=warmup,
                push_burst_start=burst_start,
                push_burst_len=cfg.push_burst_len,
                push_burst_sigma=float(burst_sigma),
                ou_tau=cfg.ou_tau,
                domain=domain,
                task=task,
                kind="sac_perturbed",
                severity=severity,
            )

            pixels_list.append(ep["pixels"])
            actions_list.append(ep["actions"])
            rewards_list.append(ep["rewards"])
            for k, v in ep.items():
                if k in ("pixels", "actions", "rewards", "metadata"):
                    continue
                metric_lists.setdefault(k, []).append(np.asarray(v, dtype=np.float32))

            ep_return = float(ep["rewards"].sum())
            summaries.append({
                "ep_idx": i,
                "severity": severity,
                "random_warmup_steps": warmup,
                "push_burst_start": burst_start,
                "push_burst_len": cfg.push_burst_len,
                "push_burst_sigma": float(burst_sigma),
                "return": ep_return,
                "mean_reward": float(ep["rewards"].mean()) if ep["rewards"].size else 0.0,
                "length": int(ep["rewards"].size),
            })
            progress.update(
                task_id, advance=1,
                ep_info=f"sev={severity:.2f} return={ep_return:6.1f}",
            )

    pixels = np.stack(pixels_list, axis=0)
    actions = np.stack(actions_list, axis=0)
    rewards = np.stack(rewards_list, axis=0)
    batched: dict[str, np.ndarray] = {
        "pixels": pixels, "actions": actions, "rewards": rewards,
    }
    for k, vs in metric_lists.items():
        batched[k] = np.stack(vs, axis=0)
    batched["severity"] = np.asarray(
        [s["severity"] for s in summaries], dtype=np.float32
    )
    batched["random_warmup_steps"] = np.asarray(
        [s["random_warmup_steps"] for s in summaries], dtype=np.int32
    )
    batched["push_burst_start"] = np.asarray(
        [s["push_burst_start"] for s in summaries], dtype=np.int32
    )
    batched["push_burst_sigma"] = np.asarray(
        [s["push_burst_sigma"] for s in summaries], dtype=np.float32
    )

    config = {
        "n_episodes": cfg.n_episodes,
        "severity_beta_a": cfg.severity_beta_a,
        "severity_beta_b": cfg.severity_beta_b,
        "warmup_steps_max": cfg.warmup_steps_max,
        "push_burst_sigma_max": cfg.push_burst_sigma_max,
        "push_burst_len": cfg.push_burst_len,
        "push_burst_window": list(cfg.push_burst_window),
        "burst_after_warmup_buffer": cfg.burst_after_warmup_buffer,
        "ou_tau": cfg.ou_tau,
        "episode_steps": int(episode_steps),
        "control_timestep": float(control_timestep),
        "seed": int(seed),
        "sac_timesteps": int(sac_timesteps),
    }
    batched["metadata"] = np.asarray(json.dumps({
        "kind": "sac_perturbed_dataset",
        "domain": domain, "task": task, "config": config,
        "episodes": summaries,
        "action_semantics": "actions[t] advances pixels[t] to pixels[t+1]",
    }))
    np.savez_compressed(dataset_dir / "perturbed.npz", **batched)

    return {"episodes": summaries, "config": config}


def collect_sac_goal_demo(
    *,
    model,
    env_fn: Callable[[int], object],
    action_bounds_fn: Callable[[object], tuple[np.ndarray, np.ndarray]],
    render_fn: Callable[[object], np.ndarray],
    record_fn: Callable[[object], dict[str, float]],
    control_timestep: float,
    episode_steps: int,
    seed: int,
    domain: str = "",
    task: str = "",
) -> dict[str, np.ndarray]:
    """Single deterministic SAC rollout from env reset. The model is passed
    in (typically produced by ``train_or_load_sac`` upstream so the cached
    policy is reused). No perturbation — pure SAC policy."""
    rng = np.random.default_rng(seed)
    return rollout_sac_episode(
        model=model,
        env_fn=env_fn,
        action_bounds_fn=action_bounds_fn,
        render_fn=render_fn,
        control_timestep=control_timestep,
        record_fn=record_fn,
        episode_steps=episode_steps,
        seed=seed,
        rng=rng,
        random_warmup_steps=0,
        push_burst_start=0,
        push_burst_len=0,
        push_burst_sigma=0.0,
        domain=domain,
        task=task,
        kind="sac_goal_demo",
    )
