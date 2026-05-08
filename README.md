# LeWM Locomotion

本项目基于 LeWM 做视觉世界模型控制。LeWM 从 raw camera pixels 和 actions 中学习身体动力学，只负责预测未来 latent，不学习 reward、value 或 policy。我们的目标是把原始 references/lewm 中的 goal-image planning 扩展为 locomotion 的 goal-trajectory / goal-manifold tracking：先用 Oracle-CEM 生成稳定站立或行走轨迹，再让 LeWM-CEM 在内部 rollout 中追踪这些目标轨迹。

现有 `oracle_cem_walker_walk.py` 已验证 MuJoCo-oracle CEM 可以稳定控制 walk，是当前最可靠的 teacher / demonstration generator。后续任务是用它生成 demo，再训练 LeWM，并实现 no-oracle 的 LeWM-CEM 部署视频。

## Quick Start

```bash
uv sync
uv run python walker_walk.py
````

输出位于：

```text
outputs/
```

包括 checkpoint、metrics 和 eval video。

## 项目目录

```text
model.py        LeWM、SIGReg
<task>.py       task specific CEM and training

```