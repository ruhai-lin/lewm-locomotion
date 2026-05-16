# LeWM-Locomotion

Reward-free visual world-model control with **LeWM** (Latent Embedding World
Model) on `dm_control` locomotion tasks. LeWM learns action-conditioned visual
latent dynamics from raw camera pixels and actions — no reward, no value, no
policy, no inverse dynamics — and a CEM planner searches in that latent space
for action sequences whose predicted futures stay on a **success manifold**
extracted from the same data.

If you like this project, please consider starring ⭐ this repo as it is the easiest and best way to support it.

## Quick Start

```bash
# This project uses uv to manage environment. 
uv sync
# Walker walks (~hour per task on a 3090: dataset + training + eval)
uv run python tasks/walker_walk.py
```

Each task is a one-shot entrypoint. The pipeline auto-detects an existing
dataset and skips collection if it is current; bump the version to invalidate. 
Outputs land under ``outputs/<task_name>/``:

```text
outputs/walker_walk/
├── ckpt/
│   ├── lewm.pt          # shipped checkpoint (= best sliding-mean pred_loss)
│   ├── lewm_best.pt     # safety copy of the same weights
│   └── lewm_final.pt    # final-step weights (kept for record)
├── train_log.jsonl      # per-step training metrics
├── eval_trace.jsonl     # per-step eval metrics + CEM cost breakdown
├── eval_cam.mp4         # eval video
├── goal_trajectory.mp4  # the SAC (or best-explore) goal demo
├── metrics.json         # summary stats
└── diagnostics.json     # goal manifold sizes/scales + planner weights
```

The dataset itself is at ``datasets/<task_name>/`` with the same schema for
all three tasks: ``episode_NNNN.npz`` files (pixels, actions, rewards, plus
task-specific per-step metrics), ``goal_trajectory.npz`` (the SAC demo),
``goal_manifold.mp4`` (post-hoc sweep of "looks like the goal" frames across
the whole dataset, for sanity-checking), and ``metadata.json``.

## Repo layout

```text
src/
├── datasets.py    # exploration policies + parallel data collection +
│                  # goal-manifold rendering + SAC goal demo (SB3)
├── lewm.py        # SIGReg/JEPA/ARPredictor + training + rollout
├── manifold.py    # latent state/delta manifold construction +
│                  # build_goal_manifold helper
├── planner.py     # LatentCEMPlanner
└── eval.py        # task-agnostic CEM eval loop

tasks/
├── walker_walk.py    # 580 lines, mostly constants + 4 task callables
├── hopper_stand.py   # same shape, hopper-specific bits
└── cheetah_run.py    # same shape, cheetah-specific bits
```

A task adapter is fully described by **three callables and a constants block**:

- ``make_env(seed)``, ``render_dataset_frame(env)``, ``render_video_frame(env)``,
  ``diagnostics(env)`` — env wiring.
- ``goal_frame_mask(episode)`` — post-hoc frame labelling for the goal manifold.
- The ``CEM_*`` and goal-threshold constants.

The goal demo is produced by ``datasets.collect_sac_goal_demo`` (a thin
SB3 SAC wrapper, ``SAC_TIMESTEPS`` of dm_control-state training); no
task-specific score function is needed.


If you find this code useful, please cite the following source:
```
@article{lin2026lewm-locomotion,
  title={},
  author={Lin, Ruhai and Eshraghian, Jason K},
  journal={arXiv preprint},
  year={2026}
}
```