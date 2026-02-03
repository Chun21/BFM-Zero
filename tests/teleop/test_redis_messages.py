import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.teleop.messages import TeleopQposMessage


def test_message_roundtrip():
    msg = TeleopQposMessage(
        ts_ms=123,
        seq=7,
        root_pos=np.zeros(3, dtype=np.float32),
        root_quat=np.array([1, 0, 0, 0], dtype=np.float32),
        qpos=np.zeros(29, dtype=np.float32),
        valid=True,
        quality=None,
    )
    payload = msg.to_json()
    msg2 = TeleopQposMessage.from_json(payload)
    assert msg2.seq == msg.seq
    assert msg2.qpos.shape == (29,)
