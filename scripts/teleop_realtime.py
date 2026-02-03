#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
from loguru import logger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_policy.teleop.redis_io import RedisQposSubscriber
from rl_policy.teleop.ring_buffer import RingBuffer
from rl_policy.teleop.tracking_provider import TrackingZProvider


def _validate_msg(msg) -> bool:
    if msg is None or not msg.valid:
        return False
    if msg.root_pos.shape != (3,) or msg.root_quat.shape != (4,) or msg.qpos.shape != (29,):
        return False
    if not np.all(np.isfinite(msg.root_pos)):
        return False
    if not np.all(np.isfinite(msg.root_quat)):
        return False
    if not np.all(np.isfinite(msg.qpos)):
        return False
    norm = np.linalg.norm(msg.root_quat)
    if norm < 1e-6:
        return False
    msg.root_quat = msg.root_quat / norm
    return True


def _teleop_loop(
    subscriber: RedisQposSubscriber,
    buffer: RingBuffer,
    ref,
    tracking,
    provider: TrackingZProvider,
    fps: float,
    seq_length: int,
    stale_ms: int,
) -> None:
    last_seq: Optional[int] = None
    last_ts: Optional[int] = None
    dt = 1.0 / float(fps)
    from rl_policy.teleop.next_obs_builder import build_next_obs_from_buffer

    while True:
        msg = subscriber.poll_latest()
        if msg is None:
            time.sleep(0.001)
            continue
        if last_seq is not None and msg.seq == last_seq:
            time.sleep(0.001)
            continue
        if not _validate_msg(msg):
            continue

        if last_ts is not None and msg.ts_ms - last_ts > stale_ms:
            logger.warning(f"teleop stale: gap {msg.ts_ms - last_ts}ms")
        last_ts = msg.ts_ms
        last_seq = msg.seq

        buffer.append(msg)
        next_obs = build_next_obs_from_buffer(buffer, ref, dt, target_len=seq_length)
        if next_obs is None:
            continue

        try:
            z_seq = tracking.infer(next_obs).astype(np.float32)
        except Exception as exc:
            logger.error(f"tracking inference failed: {exc}")
            continue

        z = z_seq[-1] if z_seq.ndim == 2 else z_seq
        provider.update(z)


def main() -> None:
    import yaml
    from rl_policy.bfm_zero import BFMZeroPolicy
    from rl_policy.offline_gmr.reference_mujoco import ReferenceMujoco
    from rl_policy.offline_gmr.tracking_infer import TrackingInfer

    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_host", default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--channel", default="teleop/qpos")
    parser.add_argument("--tracking_model", default="model/checkpoint/model")
    parser.add_argument("--policy_onnx", default="model/exported/FBcprAuxModel.onnx")
    parser.add_argument("--robot_config", default="config/robot/g1.yaml")
    parser.add_argument("--policy_config", default="config/policy/motivo_newG1.yaml")
    parser.add_argument("--ref_xml", default="bfm_zero_inference_code/g1_for_reward_inference.xml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--rl_rate", type=int, default=50)
    parser.add_argument("--stale_ms", type=int, default=40)
    args = parser.parse_args()

    tracking = TrackingInfer(model_path=args.tracking_model, device=args.device)
    cfg_seq_len = int(getattr(tracking.model.cfg, "seq_length", 1))
    seq_length = int(args.seq_length) if args.seq_length is not None else cfg_seq_len
    z_dim = int(tracking.model.cfg.archi.z_dim)

    provider = TrackingZProvider(window_size=seq_length, z_dim=z_dim)
    subscriber = RedisQposSubscriber(
        host=args.redis_host,
        port=args.redis_port,
        channel=args.channel,
    )
    buffer = RingBuffer(maxlen=seq_length)
    ref = ReferenceMujoco(xml_path=args.ref_xml)

    thread = threading.Thread(
        target=_teleop_loop,
        args=(
            subscriber,
            buffer,
            ref,
            tracking,
            provider,
            args.fps,
            seq_length,
            args.stale_ms,
        ),
        daemon=True,
    )
    thread.start()

    with open(args.robot_config, "r") as f:
        robot_config = yaml.safe_load(f)
    with open(args.policy_config, "r") as f:
        policy_config = yaml.safe_load(f)

    exp_config = {
        "type": "tracking",
        "start": 0,
        "end": 1,
        "stop": 0,
        "gamma": float(args.gamma),
        "window_size": int(args.window_size),
        "ctx_override": np.zeros((1, z_dim), dtype=np.float32),
        "z_provider": provider.latest,
    }

    policy = BFMZeroPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        exp_config=exp_config,
        model_path=args.policy_onnx,
        rl_rate=int(args.rl_rate),
    )
    policy.use_policy_action = True
    policy.get_ready_state = False
    policy.start_motion = True
    policy.t = policy.t_start
    policy.run()


if __name__ == "__main__":
    main()
