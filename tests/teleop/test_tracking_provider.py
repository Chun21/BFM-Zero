import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.teleop.tracking_provider import TrackingZProvider


def test_provider_returns_last_z():
    provider = TrackingZProvider(window_size=2)
    z = np.ones(256, dtype=np.float32)
    provider.update(z)
    assert np.allclose(provider.latest(), z)
