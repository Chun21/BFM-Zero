
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_import_tracking_model():
    from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel  # noqa: F401
