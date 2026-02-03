import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

mujoco = pytest.importorskip("mujoco")
from rl_policy.offline_gmr.reference_mujoco import ReferenceMujoco


def test_reference_obs_shapes():
    ref = ReferenceMujoco(xml_path="bfm_zero_inference_code/g1_for_reward_inference.xml")
    root_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dof_pos = np.zeros(29, dtype=np.float32)
    dof_vel = np.zeros(29, dtype=np.float32)
    root_vel = np.zeros(3, dtype=np.float32)
    root_ang_vel = np.zeros(3, dtype=np.float32)
    state, privileged = ref.compute_obs(root_pos, root_quat, dof_pos, dof_vel, root_vel, root_ang_vel)
    assert state.shape == (64,)
    assert privileged.shape[0] > 0
