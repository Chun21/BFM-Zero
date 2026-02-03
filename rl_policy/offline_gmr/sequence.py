from __future__ import annotations

from dataclasses import dataclass
import pickle
import numpy as np


@dataclass
class GmrSequence:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray  # wxyz
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_vel: np.ndarray
    root_ang_vel: np.ndarray

    @staticmethod
    def load(path: str, quat_order: str = "xyzw") -> "GmrSequence":
        with open(path, "rb") as f:
            data = pickle.load(f)
        fps = float(data["fps"])
        if fps <= 0:
            raise ValueError("fps must be positive")
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot = np.asarray(data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        if quat_order == "xyzw":
            root_rot = root_rot[:, [3, 0, 1, 2]]  # -> wxyz
        elif quat_order != "wxyz":
            raise ValueError("quat_order must be xyzw or wxyz")

        if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
            raise ValueError("sequence length mismatch")

        dt = 1.0 / fps
        dof_vel = np.zeros_like(dof_pos)
        root_vel = np.zeros_like(root_pos)
        root_ang_vel = np.zeros((root_rot.shape[0], 3), dtype=np.float32)
        if len(dof_pos) > 1:
            dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
            root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
        return GmrSequence(
            fps=fps,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
        )
