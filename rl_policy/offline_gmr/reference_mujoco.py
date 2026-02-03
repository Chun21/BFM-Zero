from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def quat_mul(a, b, w_last: bool = True):
    if isinstance(a, np.ndarray):
        a = np.asarray(a)
    if isinstance(b, np.ndarray):
        b = np.asarray(b)
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    if w_last:
        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    else:
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    if w_last:
        quat = np.stack([x, y, z, w], axis=-1).reshape(shape)
    else:
        quat = np.stack([w, x, y, z], axis=-1).reshape(shape)
    return quat


def quat_rotate(q, v):
    q = np.asarray(q)
    v = np.asarray(v)
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[:, None]
    b = np.cross(q_vec, v) * (q_w[:, None] * 2.0)
    c = q_vec * (np.sum(q_vec * v, axis=-1)[:, None] * 2.0)
    return a + b + c


def quat_to_tan_norm(q, w_last: bool = True):
    q = np.asarray(q)
    ref_tan = np.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    ref_norm = np.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    if not w_last:
        raise NotImplementedError("w_last=False not supported")
    tan = quat_rotate(q, ref_tan)
    norm = quat_rotate(q, ref_norm)
    return np.concatenate([tan, norm], axis=-1)


def calc_heading_quat_inv(q, w_last: bool = True):
    q = np.asarray(q)
    if w_last:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    axis = np.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    half_angle = -heading / 2.0
    cos_half = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    if w_last:
        heading_q = np.concatenate([sin_half[..., None] * axis, cos_half[..., None]], axis=-1)
    else:
        heading_q = np.concatenate([cos_half[..., None], sin_half[..., None] * axis], axis=-1)
    return heading_q


def calc_angular_velocity(quat_cur, quat_prev, dt):
    quat_cur = np.asarray(quat_cur)
    quat_prev = np.asarray(quat_prev)
    orig_shape = quat_cur.shape
    if quat_cur.ndim == 1:
        quat_cur = quat_cur[None, :]
        quat_prev = quat_prev[None, :]
    quat_cur_xyzw = np.stack([quat_cur[:, 1], quat_cur[:, 2], quat_cur[:, 3], quat_cur[:, 0]], axis=-1)
    quat_prev_xyzw = np.stack([quat_prev[:, 1], quat_prev[:, 2], quat_prev[:, 3], quat_prev[:, 0]], axis=-1)
    rot_cur = Rotation.from_quat(quat_cur_xyzw)
    rot_prev = Rotation.from_quat(quat_prev_xyzw)
    delta_rot = rot_prev.inv() * rot_cur
    rotvec = delta_rot.as_rotvec()
    angular_velocity = rotvec / dt
    if orig_shape == (4,):
        return angular_velocity[0]
    return angular_velocity


class ReferenceMujoco:
    def __init__(self, xml_path: str):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.body_pos_prev = None
        self.body_quat_prev = None
        self.root_quat_prev = None
        self.dt = float(self.mj_model.opt.timestep)

    def _set_state(self, root_pos, root_quat, dof_pos, dof_vel, root_vel, root_ang_vel):
        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        if root_ang_vel is None:
            if self.root_quat_prev is None:
                root_ang_vel = np.zeros(3, dtype=np.float32)
            else:
                root_ang_vel = calc_angular_velocity(root_quat, self.root_quat_prev, self.dt)
            self.root_quat_prev = root_quat.copy()
        local_root_ang_vel = rot.inv().apply(root_ang_vel)
        self.mj_data.qpos[0:3] = root_pos
        self.mj_data.qpos[3:7] = root_quat
        self.mj_data.qpos[7:7 + 29] = dof_pos
        self.mj_data.qvel[0:3] = root_vel
        self.mj_data.qvel[3:6] = local_root_ang_vel
        self.mj_data.qvel[6:6 + 29] = dof_vel
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def compute_obs(
        self,
        root_pos,
        root_quat,
        dof_pos,
        dof_vel,
        root_vel,
        root_ang_vel,
        dt: float | None = None,
    ):
        if dt is not None and dt > 0:
            self.dt = float(dt)
        self._set_state(root_pos, root_quat, dof_pos, dof_vel, root_vel, root_ang_vel)

        dof_pos = self.mj_data.qpos[7:7 + 29].copy()
        dof_vel = self.mj_data.qvel[6:6 + 29].copy()
        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        projected_gravity = rot.inv().apply(np.array([0, 0, -1], dtype=np.float32))
        ang_vel = rot.apply(self.mj_data.qvel[3:6].copy())
        state = np.concatenate([dof_pos, dof_vel, projected_gravity, ang_vel]).astype(np.float32)

        privileged_state = self._get_privileged_state()
        return state, privileged_state

    def _get_privileged_state(self):
        total_bodies = self.mj_model.nbody
        valid_body_indices = []
        body_names_list = []
        head_link_idx = None

        for i in range(total_bodies):
            try:
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    if not (body_name.startswith("dummy") or body_name.endswith("hand") or body_name.startswith("world")):
                        if body_name == "head_link":
                            head_link_idx = i
                        else:
                            valid_body_indices.append(i)
                            body_names_list.append(body_name)
                else:
                    valid_body_indices.append(i)
                    body_names_list.append(f"body_{i}")
            except Exception:
                valid_body_indices.append(i)
                body_names_list.append(f"body_{i}")

        if head_link_idx is not None:
            valid_body_indices.append(head_link_idx)
            body_names_list.append("head_link")

        num_bodies = len(valid_body_indices)
        valid_body_indices = np.array(valid_body_indices)

        body_pos = self.mj_data.xpos[valid_body_indices, :].copy()
        body_quat = self.mj_data.xquat[valid_body_indices, :].copy()  # wxyz

        if self.body_pos_prev is None:
            body_vel = np.zeros((num_bodies, 3))
            body_ang_vel = np.zeros((num_bodies, 3))
            self.body_pos_prev = body_pos
            self.body_quat_prev = body_quat
        else:
            body_vel = (body_pos - self.body_pos_prev) / self.dt
            body_ang_vel = calc_angular_velocity(body_quat, self.body_quat_prev, self.dt)
            self.body_pos_prev = body_pos
            self.body_quat_prev = body_quat

        body_pos_t = body_pos[None, :, :]
        body_rot_t = body_quat[:, [1, 2, 3, 0]][None, :, :]
        body_vel_t = body_vel[None, :, :]
        body_ang_vel_t = body_ang_vel[None, :, :]

        root_pos = body_pos_t[:, 0:1, :]
        root_rot = body_rot_t[:, 0:1, :]

        heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)
        heading_rot_inv_expand = np.repeat(heading_rot_inv, num_bodies, axis=1)
        flat_heading_rot_inv = heading_rot_inv_expand.reshape(-1, 4)

        root_pos_expand = np.repeat(root_pos, num_bodies, axis=1)
        local_body_pos = body_pos_t - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(-1, 3)
        flat_local_body_pos = quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
        local_body_pos_obs = flat_local_body_pos.reshape(1, -1)
        local_body_pos_obs = local_body_pos_obs[..., 3:]

        flat_body_rot = body_rot_t.reshape(-1, 4)
        flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
        flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(1, -1)

        flat_body_vel = body_vel_t.reshape(-1, 3)
        flat_local_body_vel = quat_rotate(flat_heading_rot_inv, flat_body_vel)
        local_body_vel_obs = flat_local_body_vel.reshape(1, -1)

        flat_body_ang_vel = body_ang_vel_t.reshape(-1, 3)
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
        local_body_ang_vel_obs = flat_local_body_ang_vel.reshape(1, -1)

        root_h = root_pos[:, :, 2:3].squeeze(0)

        privileged_state = np.concatenate(
            [
                root_h,
                local_body_pos_obs,
                local_body_rot_obs,
                local_body_vel_obs,
                local_body_ang_vel_obs,
            ],
            axis=-1,
        ).squeeze(0)
        return privileged_state.astype(np.float32)
