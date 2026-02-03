from __future__ import annotations

from collections import deque


class RingBuffer:
    def __init__(self, maxlen: int):
        self._buf = deque(maxlen=maxlen)

    def append(self, item) -> None:
        self._buf.append(item)

    def values(self):
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
