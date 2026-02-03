import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.bfm_zero import BFMZeroPolicy


def test_live_z_hook_present():
    assert hasattr(BFMZeroPolicy, "_get_live_z")
