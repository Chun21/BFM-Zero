from __future__ import annotations

import os
import numpy as np
import torch

from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel


class TrackingInfer:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu.")
        self.device = torch.device(device)
        self.model = FBcprAuxModel.load(model_path, device=device)

    def infer(self, next_obs: dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            next_obs_t = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device)
                for k, v in next_obs.items()
            }
            z = self.model.tracking_inference(next_obs_t)
        return z.cpu().numpy()
