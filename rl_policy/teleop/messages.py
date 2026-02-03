from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np


@dataclass
class TeleopQposMessage:
    ts_ms: int
    seq: int
    root_pos: np.ndarray
    root_quat: np.ndarray
    qpos: np.ndarray
    valid: bool
    quality: float | None = None

    def to_json(self) -> str:
        payload = {
            "ts": int(self.ts_ms),
            "seq": int(self.seq),
            "root_pos": self.root_pos.tolist(),
            "root_quat": self.root_quat.tolist(),
            "qpos": self.qpos.tolist(),
            "valid": bool(self.valid),
            "quality": None if self.quality is None else float(self.quality),
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(payload: str) -> "TeleopQposMessage":
        data = json.loads(payload)
        return TeleopQposMessage(
            ts_ms=int(data["ts"]),
            seq=int(data["seq"]),
            root_pos=np.asarray(data["root_pos"], dtype=np.float32),
            root_quat=np.asarray(data["root_quat"], dtype=np.float32),
            qpos=np.asarray(data["qpos"], dtype=np.float32),
            valid=bool(data.get("valid", True)),
            quality=data.get("quality", None),
        )
