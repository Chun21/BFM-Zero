import os
import sys
import inspect

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import rl_policy.bfm_zero as bfm_zero


def test_ctx_override_hook_present():
    src = inspect.getsource(bfm_zero.BFMZeroPolicy.__init__)
    assert "ctx_override" in src
