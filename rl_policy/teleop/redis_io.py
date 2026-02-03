from __future__ import annotations

import redis

from .messages import TeleopQposMessage


class RedisQposSubscriber:
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, channel: str = "teleop/qpos"):
        self._redis = redis.Redis(host=host, port=port, decode_responses=True)
        self._channel = channel
        self._sub = self._redis.pubsub()
        self._sub.subscribe(channel)
        self._latest: TeleopQposMessage | None = None

    def poll_latest(self) -> TeleopQposMessage | None:
        latest = None
        while True:
            msg = self._sub.get_message(ignore_subscribe_messages=True)
            if not msg:
                break
            if isinstance(msg.get("data"), str):
                latest = TeleopQposMessage.from_json(msg["data"])
        if latest is not None:
            self._latest = latest
        return self._latest
