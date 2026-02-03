from __future__ import annotations

import numpy as np

from rl_policy.offline_gmr.reference_mujoco import ReferenceMujoco
from rl_policy.teleop.ring_buffer import RingBuffer


def build_next_obs_from_buffer(
    buffer: RingBuffer,
    ref: ReferenceMujoco,
    dt: float,
    target_len: int | None = None,
) -> dict[str, np.ndarray] | None:
    if len(buffer) == 0:
        return None

    msgs = buffer.values()
    if target_len is not None and len(msgs) < target_len:
        last = msgs[-1]
        msgs = msgs + [last] * (target_len - len(msgs))
    dof_pos = np.stack([m.qpos for m in msgs], axis=0).astype(np.float32)
    root_pos = np.stack([m.root_pos for m in msgs], axis=0).astype(np.float32)
    root_quat = np.stack([m.root_quat for m in msgs], axis=0).astype(np.float32)

    dof_vel = np.zeros_like(dof_pos)
    root_vel = np.zeros_like(root_pos)
    if len(msgs) > 1:
        dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
        root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt

    states = []
    privs = []
    for i in range(len(msgs)):
        state, privileged = ref.compute_obs(
            root_pos[i],
            root_quat[i],
            dof_pos[i],
            dof_vel[i],
            root_vel[i],
            None,
            dt=dt,
        )
        states.append(state)
        privs.append(privileged)

    return {
        "state": np.stack(states, axis=0).astype(np.float32),
        "privileged_state": np.stack(privs, axis=0).astype(np.float32),
    }
