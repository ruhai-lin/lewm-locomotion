# LeWM Locomotion

本项目基于 LeWM 做 reward-free 视觉世界模型控制。LeWM 从 raw camera pixels 和 actions 中学习 action-conditioned visual latent dynamics，只负责预测未来 latent，不学习 reward、value、policy 或 inverse dynamics。目标是把原始 `references/lewm` 中的 goal-image planning 扩展到 locomotion：不是直接最大化环境 reward，而是在 LeWM 的 latent rollout 中寻找能回到成功运动 manifold 的动作序列。

当前 pipeline 采用 reference-style 的三阶段流程：先离线生成 `datasets/<task>`，再只从静态数据训练 LeWM，最后做 pure LeWM-CEM eval。MuJoCo oracle 只用于离线 demonstration / recovery 数据生成；训练和评估阶段不使用在线 oracle、reward、intervention 或 hand-coded recovery mode。

第一个 milestone 是 `tasks/walker_walk.py`：基于 clean walking trajectory 的 near-orbit latent tracking 已经证明 LeWM-CEM 可以在接近稳定步态轨道时持续走起来。现在的核心方向是在此基础上推进 latent delta manifold planning：从成功 videos 中构建 goal state manifold、goal delta manifold 和成功 transition/support manifold，让 CEM 用 LeWM rollout 出多步未来 latents / latent deltas，并用纯 latent 几何距离评估“未来运动是否像成功行为、是否回到成功运动 manifold”。near-orbit tracking 和 demo action prior 会保留为低权重稳定器和实验旋钮，但不应重新成为主导目标。

当前结构分为两层：`src/` 是通用第一层方法，包含 reference-aligned LeWM 训练、latent rollout、manifold 构建和 CEM planner；`tasks/` 是第二层 task adapter，只负责 env、离线数据、成功片段筛选、训练入口和输出。walker 的 height/upright/velocity 等规则只能用于 adapter 里的 success segment 清洗，不进入通用 planner。

## Quick Start

```bash
uv sync
uv run python tasks/walker_walk.py
```

输出位于：

```text
outputs/
```

包括 checkpoint、metrics 和 eval video。

## 项目目录

```text
model.py        LeWM / SIGReg architecture, aligned with references/le-wm
src/            task-agnostic LeWM training, latent manifold, CEM planner
tasks/          task adapters and one-command experiment entrypoints
datasets/       static offline datasets
outputs/        checkpoints, metrics, diagnostics, videos
```
