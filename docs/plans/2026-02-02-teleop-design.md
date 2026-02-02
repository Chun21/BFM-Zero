# 全身遥操（动捕 + GMR + BFM‑Zero tracking）设计

## 背景与目标
- 目标：基于 `deploy` 分支实现全身遥操流程：动捕实时输入 → GMR 重定向 → 参考序列 → tracking 推理 z → BFM‑Zero policy 输出动作，先在仿真中验证，再上真机。
- 范围：新增一个“骨骼映射导出工具”与最小链路集成；训练与模型细节不在本次范围。

## 总体架构
1) **动捕输入层**：XING SDK 获取 Skeleton/RigidBody 位姿（x,y,z + qx,qy,qz,qw）。
2) **重定向层（GMR）**：将动捕骨骼位姿映射到 G1 29DOF 关节与 root pose。
3) **参考序列缓存**：滑动窗口缓存 50Hz 参考序列。
4) **观测构造**：复用 `minimal_inference/env.py` 的观测构造逻辑，将参考 qpos/root pose 生成 `next_obs`。
5) **z 推理**：调用 `tracking_inference(next_obs)` 得到 z 序列（按 `seq_length` 平均），并归一化。
6) **策略推理**：`policy(obs, z)` 输出动作，在仿真中通过 deploy 的通信/控制桥接执行。

## 关键组件与文件
- `third_party/xing_sdk/`：拷贝 SDK，统一依赖来源。
- `tools/xing_dump_skeleton/`：连接动捕服务器，导出骨骼 ID/名称/parentID/offset/quat 的 JSON。
- GMR：使用 `/home/chunyu/programs/GMR`（GMR 重定向运行在 conda 环境 `gmr`，Python 3.10），以 `ik_configs/xrobot_to_g1.json` 做骨骼映射。
- BFM‑Zero 推理：复用 `minimal_inference` 分支的 z 推理逻辑与 `env.py` 观测构造。
- 仿真执行：沿用 `deploy` 的 `sim_env` + ZMQ 低层控制通道。
- XING SDK：原始 SDK 路径 `/home/chunyu/programs/XING_Linux_C++_SDK_4.1.0.5634`，会完整拷贝到本仓库 `third_party/xing_sdk/` 并以仓库内路径为准。

## 数据流（仿真阶段）
1. XING SDK 读取 Skeleton 帧（mm + qx,qy,qz,qw），SDK 来源为 `/home/chunyu/programs/XING_Linux_C++_SDK_4.1.0.5634` 并拷贝至仓库内 `third_party/xing_sdk/` 使用（XING：右手系，Y‑up）。
2. 单位换算：mm→m；四元数改为 scalar‑first（qw,qx,qy,qz）；坐标系转换到仿真/机器人坐标系（需明确 X/Z 正方向约定后固化）。
3. GMR `retarget()` 输出机器人 qpos（含 root pose），在 conda 环境 `gmr`（Python 3.10）中运行。
4. 参考序列滑窗（控制/推理频率 50Hz，窗口长度与 `seq_length` 对齐）。
5. `env.py` 生成 `next_obs` → `tracking_inference(next_obs)` → z。
6. `policy(obs, z)` 输出动作 → 低层指令做 EMA 平滑（α=0.2）→ 通过 deploy 的控制通道驱动仿真（仿真步进默认 200Hz，对应 `SIMULATE_DT=0.005`）。

## 错误处理与鲁棒性
- 连接失败/骨骼缺失：工具与运行时明确报错并退出非 0。
- 帧丢失/乱序：根据时间戳做降噪与丢帧补偿；必要时保持上一帧。
- 坐标/单位异常：在日志中标注并拒绝推理。
- 坐标系不一致：明确 XING 为右手系、Y‑up；在转换前校验 X/Z 正方向（前/右/后）并固化转换矩阵。
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
