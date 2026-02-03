import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.offline_gmr.tracking_infer import TrackingInfer


def test_tracking_infer_missing_model_path(tmp_path):
    next_obs = {
        "state": np.zeros((2, 64), dtype=np.float32),
        "privileged_state": np.zeros((2, 463), dtype=np.float32),
    }
    with pytest.raises(FileNotFoundError):
        TrackingInfer(model_path=str(tmp_path / "missing")).infer(next_obs)
