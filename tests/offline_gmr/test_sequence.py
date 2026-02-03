import os
import sys
import pickle
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.offline_gmr.sequence import GmrSequence


def _write_pkl(path):
    data = {
        "fps": 50,
        "root_pos": np.zeros((3, 3), dtype=np.float32),
        "root_rot": np.array([[0, 0, 0, 1]] * 3, dtype=np.float32),  # xyzw
        "dof_pos": np.zeros((3, 29), dtype=np.float32),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_gmr_sequence_basic(tmp_path):
    pkl_path = tmp_path / "seq.pkl"
    _write_pkl(pkl_path)
    seq = GmrSequence.load(str(pkl_path), quat_order="xyzw")
    assert seq.fps == 50
    assert seq.root_pos.shape == (3, 3)
    assert seq.root_rot.shape == (3, 4)
    assert seq.dof_pos.shape == (3, 29)
    assert seq.dof_vel.shape == (3, 29)
