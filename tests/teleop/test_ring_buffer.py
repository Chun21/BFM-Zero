import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.teleop.ring_buffer import RingBuffer


def test_ring_buffer_order():
    buf = RingBuffer(maxlen=3)
    for i in range(5):
        buf.append(i)
    assert list(buf.values()) == [2, 3, 4]
