#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import yaml
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import torch early to avoid static TLS issues on aarch64 when mujoco loads first.
import torch  # noqa: F401

from rl_policy.offline_gmr.sequence import GmrSequence
from rl_policy.offline_gmr.reference_mujoco import ReferenceMujoco
from rl_policy.offline_gmr.tracking_infer import TrackingInfer
from rl_policy.bfm_zero import BFMZeroPolicy


def build_next_obs(seq: GmrSequence, ref: ReferenceMujoco) -> dict[str, np.ndarray]:
    states = []
    privs = []
    dt = 1.0 / seq.fps
    for i in range(seq.dof_pos.shape[0]):
        state, privileged = ref.compute_obs(
            seq.root_pos[i],
            seq.root_rot[i],
            seq.dof_pos[i],
            seq.dof_vel[i],
            seq.root_vel[i],
            None,
            dt=dt,
        )
        states.append(state)
        privs.append(privileged)
    return {
        "state": np.stack(states, axis=0).astype(np.float32),
        "privileged_state": np.stack(privs, axis=0).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmr_pkl", required=True, help="Path to GMR pkl")
    parser.add_argument("--quat_order", default="xyzw", choices=["xyzw", "wxyz"])
    parser.add_argument("--tracking_model", default="model/checkpoint/model")
    parser.add_argument("--policy_onnx", default="model/exported/FBcprAuxModel.onnx")
    parser.add_argument("--robot_config", default="config/robot/g1.yaml")
    parser.add_argument("--policy_config", default="config/policy/motivo_newG1.yaml")
    parser.add_argument("--ref_xml", default="bfm_zero_inference_code/g1_for_reward_inference.xml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=1)
    args = parser.parse_args()

    seq = GmrSequence.load(args.gmr_pkl, quat_order=args.quat_order)

    ref = ReferenceMujoco(xml_path=args.ref_xml)
    next_obs = build_next_obs(seq, ref)

    tracking = TrackingInfer(model_path=args.tracking_model, device=args.device)
    z_seq = tracking.infer(next_obs).astype(np.float32)

    exp_config = {
        "type": "tracking",
        "start": 0,
        "end": int(z_seq.shape[0]),
        "stop": 0,
        "gamma": float(args.gamma),
        "window_size": int(args.window_size),
        "ctx_override": z_seq,
    }

    with open(args.robot_config) as f:
        robot_config = yaml.safe_load(f)
    with open(args.policy_config) as f:
        policy_config = yaml.safe_load(f)

    policy = BFMZeroPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        exp_config=exp_config,
        model_path=args.policy_onnx,
        rl_rate=int(seq.fps),
    )
    policy.use_policy_action = True
    policy.get_ready_state = False
    policy.start_motion = True
    policy.t = policy.t_start

    time.sleep(0.2)
    policy.run()


if __name__ == "__main__":
    main()
