# 全身遥操（动捕 + GMR + BFM‑Zero tracking）设计

## 背景与目标
- 目标：基于 `teleop-gmr` 分支实现全身遥操流程：动捕实时输入 → GMR 重定向 → 参考序列 → tracking 推理 z → BFM‑Zero policy 输出动作，先在仿真中验证，再上真机。
- 范围：新增一个“骨骼映射导出工具”与最小链路集成；训练与模型细节不在本次范围。

## 总体架构
1) **动捕输入层**：XING Python SDK（nokovpy）获取 Skeleton/RigidBody 位姿（x,y,z + qx,qy,qz,qw）。
2) **重定向层（GMR）**：将动捕骨骼位姿映射到 G1 29DOF 关节与 root pose。
3) **参考序列缓存**：滑动窗口缓存 50Hz 参考序列。
4) **观测构造**：使用 `rl_policy/offline_gmr/reference_mujoco.py` 的参考 MuJoCo 逻辑，从参考 qpos/root pose 生成 `state + privileged_state` 组成 `next_obs`。
5) **z 推理**：使用 `rl_policy/offline_gmr/tracking_infer.py` 调用 `FBcprAuxModel.tracking_inference(next_obs)` 得到 z 序列（按 `seq_length` 平均并归一化，默认 CUDA）。
6) **策略推理**：`BFMZeroPolicy` 通过 `ctx_override` 注入实时 z；`policy(obs, z)` 输出动作，在仿真中通过 `sim_env` 低层控制通道执行。

## 关键组件与文件
- XING Python SDK：使用gmr conda环境，使用 wheel `/home/chunyu/programs/third_party/nokovpy-3.0.1-py3-none-any.whl` 安装（`pip install`）。
- `tools/xing_dump_skeleton/`：连接动捕服务器，导出骨骼 ID/名称/parentID/offset/quat 的 JSON。
- GMR：使用 `/home/chunyu/programs/GMR`（GMR 重定向运行在 conda 环境 `gmr`，Python 3.10），以 `ik_configs/xrobot_to_g1.json` 做骨骼映射。
- BFM‑Zero 推理：使用仓库内 `bfm_zero_inference_code/` + `rl_policy/offline_gmr/reference_mujoco.py` + `rl_policy/offline_gmr/tracking_infer.py`（tracking z 默认 CUDA）。
- `rl_policy/offline_gmr/`：离线逻辑复用的观测构造/序列读取/推理封装，用于实时链路对齐。
- 仿真执行：沿用 `deploy` 的 `sim_env` + ZMQ 低层控制通道。
- XING SDK：用 Python SDK（nokovpy），wheel 路径 `/home/chunyu/programs/third_party/nokovpy-3.0.1-py3-none-any.whl`。
- Redis：用于 GMR → BFM‑Zero 的 qpos 发布与订阅（Pub/Sub）。

## 仿真端到端实现细化
### 链路与进程划分（最小链路）
- XING Python SDK 与 GMR 同脚本运行（conda `gmr` 环境），读取 Skeleton 帧后直接调用 `retarget()`。
- GMR 输出 `qpos + root pose`，按 50Hz 发布到 Redis 频道 `teleop/qpos`。
- BFM‑Zero 仿真进程订阅 `teleop/qpos`，每个控制周期取最新一帧，不做队列累积。

### Redis Pub/Sub 与消息格式
- 频道：`teleop/qpos`。
- 编码：JSON（便于调试）。
- 字段建议：`ts`（毫秒时间戳）、`seq`（递增序号）、`root_pos`（3）、`root_quat`（4）、`qpos`（29）、`valid`（bool）、`quality`（可选）。
- 订阅端若检测到消息间隔过大（如 > 40ms）标记为 `stale`，用于控制侧降级。

### 坐标/单位转换
- 转换放在 GMR 进程内：mm→m，四元数改为 (qw,qx,qy,qz)。
- 统一使用 `T_xing_to_sim` 做坐标系转换（XING 右手系、Y‑up → 仿真/机器人坐标系）。
- 若 `T_xing_to_sim` 未配置或校验失败，直接拒绝启动并报错，避免方向错误导致动作反向。

### BFM‑Zero 侧推理与控制节奏
- 参考序列以 50Hz 写入滑窗（RingBuffer），长度与 `seq_length` 对齐；不足时用最近帧填充。
- `reference_mujoco.py` 生成 `next_obs(state, privileged_state)` → `tracking_infer.py` 得到 z（默认 CUDA）→ 通过 `ctx_override` 注入 `BFMZeroPolicy`，`policy(obs, z)` 输出动作。
- 控制端 50Hz 更新动作目标；仿真 200Hz 步进下执行 EMA 平滑（α=0.2）。

### 落地步骤（仿真）
1) GMR 侧完成 XING → retarget → Redis 发布最小脚本，离线检查 qpos 维度与范围。
2) 仿真端订阅 `teleop/qpos`，打通参考序列 → z → policy → sim_env 的闭环。
3) 加入 qpos 录制与回放，用于稳定性与时延评估。

## 数据流（仿真阶段）
1. XING Python SDK（nokovpy）读取 Skeleton 帧（mm + qx,qy,qz,qw），wheel 路径 `/home/chunyu/programs/third_party/nokovpy-3.0.1-py3-none-any.whl`（XING：右手系，Y‑up）。
2. 单位换算：mm→m；四元数改为 scalar‑first（qw,qx,qy,qz）；坐标系转换到仿真/机器人坐标系（需明确 X/Z 正方向约定后固化）。
3. GMR `retarget()` 输出机器人 qpos（含 root pose），在 conda 环境 `gmr`（Python 3.10）中运行。
4. 参考序列滑窗（RingBuffer，50Hz，窗口长度与 `seq_length` 对齐）。
5. `reference_mujoco.py` 生成 `next_obs(state, privileged_state)` → `tracking_infer.py` → z（默认 CUDA）。
6. `policy(obs, z)` 输出动作（`ctx_override` 注入实时 z）→ 低层指令做 EMA 平滑（α=0.2）→ 通过 deploy 的控制通道驱动仿真（仿真步进默认 200Hz，对应 `SIMULATE_DT=0.005`）。

## 与 offline 逻辑对齐点
- 观测构造：使用 `rl_policy/offline_gmr/reference_mujoco.py` 生成 `state + privileged_state`。
- z 推理：使用 `rl_policy/offline_gmr/tracking_infer.py`，默认 CUDA。
- z 注入：`BFMZeroPolicy` 支持 `ctx_override` 注入实时 z。
- 序列缓存：滑窗 RingBuffer，与 `seq_length` 对齐。

## 错误处理与鲁棒性
- 连接失败/骨骼缺失：工具与运行时明确报错并退出非 0。
- sim_env 未启动：持续等待 low_state，不设置超时。
- 帧丢失/乱序：根据时间戳做降噪与丢帧补偿；必要时保持上一帧。
- 坐标/单位异常：在日志中标注并拒绝推理。
- 坐标系不一致：明确 XING 为右手系、Y‑up；在转换前校验 X/Z 正方向（前/右/后）并固化转换矩阵。
- qpos/姿态异常：`qpos` 长度不为 29、四元数非单位或出现 NaN 时直接丢弃。
- 四元数顺序：支持 `xyzw/wxyz` 配置，默认 `xyzw`。
- Redis 断连或 `stale`：进入冻结或站立姿态，记录告警日志。
- z 推理异常：回退到上一时刻 z，避免动作突变。
- 低层指令平滑：对关节目标应用一阶低通 `y = (1-α) * y_prev + α * x`，α=0.2。

## 测试计划
- **离线**：用录制骨骼数据重放，验证 GMR → qpos → next_obs → z 的一致性。
- **仿真**：端到端 50Hz 遥操验证动作可控、时延可接受。
- **真机前**：与仿真同数据源对比，确认关节范围与稳定性。

## 里程碑
1) 接入 SDK 与骨骼映射导出工具。
2) 参考序列 → z 推理链路打通（仿真）。
3) 真机接入与安全边界（限幅、急停）。
