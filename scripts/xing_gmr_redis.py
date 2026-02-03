#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from queue import Queue, Full
from threading import Thread
from typing import Dict, Optional, Tuple

import numpy as np
import redis
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GMR_ROOT = os.environ.get("GMR_ROOT", "/home/chunyu/programs/GMR")
if GMR_ROOT not in sys.path:
    sys.path.insert(0, GMR_ROOT)

from rl_policy.teleop.messages import TeleopQposMessage
from utils.strings import unitree_joint_names

try:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
except Exception as import_error:  # pragma: no cover
    raise ImportError(
        "GMR not found. Please ensure /home/chunyu/programs/GMR is available or set GMR_ROOT."
    ) from import_error

try:
    from nokov.nokovsdk import (
        PySDKClient,
        DataDescriptions,
        DataDescriptors,
        POINTER,
    )
except Exception as import_error:  # pragma: no cover
    raise ImportError(
        "nokovpy not found. Please install the wheel: "
        "/home/chunyu/programs/third_party/nokovpy-3.0.1-py3-none-any.whl"
    ) from import_error


def _load_transform(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("xing_to_sim.yaml must be a mapping with rotation/translation")
    rot = np.asarray(data.get("rotation", None), dtype=np.float64)
    trans = np.asarray(data.get("translation", None), dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    if trans.shape != (3,):
        raise ValueError("translation must be a 3-vector")
    ortho = rot.T @ rot
    if not np.allclose(ortho, np.eye(3), atol=1e-3):
        raise ValueError("rotation is not orthonormal")
    det = np.linalg.det(rot)
    if abs(det - 1.0) > 1e-2:
        raise ValueError("rotation determinant must be close to 1.0")
    return rot, trans


def _normalize_names_bvh_nokov(
    frame_map: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    normalized = dict(frame_map)
    if "LeftFootMod" not in normalized and "LeftFoot" in normalized:
        normalized["LeftFootMod"] = normalized["LeftFoot"]
    if "RightFootMod" not in normalized and "RightFoot" in normalized:
        normalized["RightFootMod"] = normalized["RightFoot"]
    if "Spine2" not in normalized:
        if "Spine3" in normalized:
            normalized["Spine2"] = normalized["Spine3"]
        elif "Spine1" in normalized:
            normalized["Spine2"] = normalized["Spine1"]
    return normalized


def _normalize_names_xrobot(
    frame_map: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    alias = {
        "Pelvis": ["Pelvis", "pelvis", "Hips", "Hip", "Root", "ROOT", "root"],
        "Spine3": ["Spine3", "Spine2", "Spine1", "Spine", "Chest", "UpperChest"],
        "Left_Hip": ["Left_Hip", "LeftUpLeg", "LHip", "LeftUpperLeg", "LeftThigh"],
        "Left_Knee": ["Left_Knee", "LeftLeg", "LeftLowerLeg", "LeftShin"],
        "Left_Foot": ["Left_Foot", "LeftFoot", "LeftAnkle", "LeftFootEnd"],
        "Left_Shoulder": ["Left_Shoulder", "LeftShoulder", "LeftArm", "LeftUpperArm"],
        "Left_Elbow": ["Left_Elbow", "LeftForeArm", "LeftLowerArm", "LeftArmLower"],
        "Left_Wrist": ["Left_Wrist", "LeftHand", "LeftHandEnd", "LeftWrist"],
        "Right_Hip": ["Right_Hip", "RightUpLeg", "RHip", "RightUpperLeg", "RightThigh"],
        "Right_Knee": ["Right_Knee", "RightLeg", "RightLowerLeg", "RightShin"],
        "Right_Foot": ["Right_Foot", "RightFoot", "RightAnkle", "RightFootEnd"],
        "Right_Shoulder": ["Right_Shoulder", "RightShoulder", "RightArm", "RightUpperArm"],
        "Right_Elbow": ["Right_Elbow", "RightForeArm", "RightLowerArm", "RightArmLower"],
        "Right_Wrist": ["Right_Wrist", "RightHand", "RightHandEnd", "RightWrist"],
    }
    normalized = dict(frame_map)
    for target_name, candidates in alias.items():
        if target_name in normalized:
            continue
        for candidate in candidates:
            if candidate in frame_map:
                normalized[target_name] = frame_map[candidate]
                break
    return normalized


def _load_joint_limits(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    lower_map = data.get("joint_pos_lower_limit", {})
    upper_map = data.get("joint_pos_upper_limit", {})
    lower = np.full(len(unitree_joint_names), -np.inf, dtype=np.float32)
    upper = np.full(len(unitree_joint_names), np.inf, dtype=np.float32)
    for i, name in enumerate(unitree_joint_names):
        if name in lower_map:
            lower[i] = float(lower_map[name])
        if name in upper_map:
            upper[i] = float(upper_map[name])
    return lower, upper


def _format_frame(
    frame: Dict[str, Tuple[np.ndarray, np.ndarray]], decimals: int = 4
) -> Dict[str, Dict[str, list]]:
    payload: Dict[str, Dict[str, list]] = {}
    for name, (pos, quat) in frame.items():
        payload[name] = {
            "pos": np.round(pos, decimals).tolist(),
            "quat": np.round(quat, decimals).tolist(),
        }
    return payload


class NokovClient:
    def __init__(
        self,
        server_ip: str,
        rot_mat: np.ndarray,
        trans: np.ndarray,
        queue_size: int = 10,
        print_level: int = 0,
        normalize_mode: str = "xrobot",
    ) -> None:
        self.server_ip = server_ip
        self.queue: "Queue[Dict[str, Tuple[np.ndarray, np.ndarray]]]" = Queue(maxsize=queue_size)
        self.latest_frame_number = -1
        self._running = False
        self._reader_thread: Optional[Thread] = None
        self._id_to_name: Dict[int, str] = {}
        self._client = PySDKClient()
        self._print_level = print_level
        self._rot_align = R.from_matrix(rot_mat)
        self._trans = trans.astype(np.float64)
        self._normalize_mode = normalize_mode

    def connected(self) -> bool:
        return self._running and self._reader_thread is not None

    def _log(self, *args) -> None:
        if self._print_level > 0:
            print(*args)

    def _build_id_name_map(self) -> None:
        pdds = POINTER(DataDescriptions)()
        n_defs = 0
        for _ in range(10):
            self._client.PyGetDataDescriptions(pdds)
            try:
                n_defs = int(pdds.contents.nDataDescriptions)
            except Exception:
                n_defs = 0
            if n_defs > 0:
                break
            time.sleep(0.1)

        if n_defs == 0:
            self._id_to_name = {}
            self._log("[NokovClient] Warning: no data descriptions available")
            return

        data_defs = pdds.contents
        id_to_name: Dict[int, str] = {}

        for i_def in range(data_defs.nDataDescriptions):
            data_def = data_defs.arrDataDescriptions[i_def]
            if data_def.type == DataDescriptors.Descriptor_Skeleton.value:
                sk_desc = data_def.Data.SkeletonDescription.contents
                for i_body in range(sk_desc.nRigidBodies):
                    body_def = sk_desc.RigidBodies[i_body]
                    try:
                        name = body_def.szName.decode("utf-8")
                    except Exception:
                        name = f"ID_{body_def.ID}"
                    id_to_name[int(body_def.ID)] = name
            elif data_def.type == DataDescriptors.Descriptor_RigidBody.value:
                rb_desc = data_def.Data.RigidBodyDescription.contents
                try:
                    name = rb_desc.szName.decode("utf-8")
                except Exception:
                    name = f"ID_{rb_desc.ID}"
                id_to_name[int(rb_desc.ID)] = name

        self._id_to_name = id_to_name
        self._log(f"[NokovClient] ID-to-name mapping size: {len(self._id_to_name)}")

    def _normalize_names(
        self, frame_map: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if self._normalize_mode == "bvh_nokov":
            return _normalize_names_bvh_nokov(frame_map)
        if self._normalize_mode == "xrobot":
            return _normalize_names_xrobot(frame_map)
        return frame_map

    def _convert_frame(
        self, frame_ptr
    ) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        if not frame_ptr:
            return None

        frame = frame_ptr.contents
        self.latest_frame_number = int(frame.iFrame)
        frame_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for i_skeleton in range(frame.nSkeletons):
            sk = frame.Skeletons[i_skeleton]
            for i_body in range(sk.nRigidBodies):
                body = sk.RigidBodyData[i_body]
                body_id = int(body.ID)
                name = self._id_to_name.get(body_id)
                if not name:
                    continue

                pos_m = np.array([body.x, body.y, body.z], dtype=np.float64) / 1000.0
                quat_wxyz = np.array(
                    [body.qw, body.qx, body.qy, body.qz], dtype=np.float64
                )

                pos_out = self._rot_align.apply(pos_m) + self._trans
                rot_body = R.from_quat(quat_wxyz, scalar_first=True)
                rot_out = self._rot_align * rot_body
                quat_out = rot_out.as_quat(scalar_first=True)

                frame_map[name] = (pos_out, quat_out)

        if not frame_map:
            return None
        return self._normalize_names(frame_map)

    def _reader_loop(self) -> None:
        while self._running:
            if not self._id_to_name:
                try:
                    self._build_id_name_map()
                except Exception:
                    pass

            frame_ptr = self._client.PyGetLastFrameOfMocapData()
            try:
                if frame_ptr:
                    converted = self._convert_frame(frame_ptr)
                    if converted is not None:
                        try:
                            self.queue.put(converted, block=False)
                        except Full:
                            try:
                                _ = self.queue.get(block=False)
                            except Exception:
                                pass
                            self.queue.put(converted, block=False)
            finally:
                if frame_ptr:
                    self._client.PyNokovFreeFrame(frame_ptr)

            time.sleep(0.0005)

    def run(self) -> bool:
        ret = self._client.Initialize(bytes(self.server_ip, encoding="utf8"))
        if ret != 0:
            print(f"[NokovClient] Failed to connect, return code: {ret}")
            return False
        try:
            self._client.PySetVerbosityLevel(self._print_level)
        except Exception:
            pass
        self._build_id_name_map()
        self._running = True
        self._reader_thread = Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        return True

    def shutdown(self) -> None:
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

    def get_frame(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.queue.get(block=True)

    def get_frame_number(self) -> int:
        return self.latest_frame_number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", help="XING/NOKOV server IP",default="10.1.1.198")
    parser.add_argument("--redis_host", default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--channel", default="teleop/qpos")
    parser.add_argument("--src_human", default="bvh_nokov")
    parser.add_argument("--tgt_robot", default="unitree_g1")
    parser.add_argument("--human_height", type=float, default=1.8)
    parser.add_argument("--offset_to_ground", action="store_true")
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--print_level", type=int, default=0)
    parser.add_argument("--robot_config", default="config/robot/g1.yaml")
    parser.add_argument("--print_frame", action="store_true")
    parser.add_argument("--print_every", type=int, default=60)
    parser.add_argument(
        "--transform_config",
        default="config/teleop/xing_to_sim.yaml",
        help="YAML with rotation/translation for XING->sim",
    )
    args = parser.parse_args()

    rot_mat, trans = _load_transform(args.transform_config)

    client = NokovClient(
        server_ip=args.server_ip,
        rot_mat=rot_mat,
        trans=trans,
        print_level=args.print_level,
        normalize_mode=args.src_human,
    )
    if not client.run():
        raise SystemExit(1)

    retarget = GMR(
        src_human=args.src_human,
        tgt_robot=args.tgt_robot,
        actual_human_height=args.human_height,
        use_velocity_limit=False,
    )

    joint_limits = _load_joint_limits(args.robot_config)
    if joint_limits is None:
        print(f"[XING->GMR] robot_config not found: {args.robot_config}")

    rds = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    seq = 0
    period = 1.0 / float(args.freq)
    last_pub = 0.0
    frame_count = 0

    print(
        f"[XING->GMR] Publish to redis://{args.redis_host}:{args.redis_port} "
        f"channel={args.channel} @~{args.freq:.1f}Hz"
    )

    warned_missing_root = False
    warned_clamp = False
    limits_disabled = False
    while True:
        frame = client.get_frame()
        now = time.time()
        if now - last_pub < period:
            continue
        frame_count += 1
        if args.print_frame and frame_count % max(1, args.print_every) == 0:
            print(f"[XING] frame={frame_count} keys={list(frame.keys())}")
            print(json.dumps(_format_frame(frame), ensure_ascii=False))

        if args.src_human == "xrobot" and "Pelvis" not in frame:
            if not warned_missing_root:
                keys_preview = list(frame.keys())[:20]
                print(
                    "[XING->GMR] missing Pelvis in frame keys, "
                    "请检查骨骼命名或尝试 --src_human bvh_nokov。"
                )
                print(f"[XING->GMR] frame keys sample: {keys_preview}")
                warned_missing_root = True
            continue

        try:
            qpos = retarget.retarget(frame, offset_to_ground=args.offset_to_ground)
        except Exception as exc:
            err_msg = str(exc)
            if "violates configuration limits" in err_msg and not limits_disabled:
                print("[XING->GMR] joint limit hit, disable GMR limits and retry once")
                retarget.ik_limits = []
                limits_disabled = True
                try:
                    qpos = retarget.retarget(frame, offset_to_ground=args.offset_to_ground)
                except Exception as exc2:
                    print(f"[XING->GMR] retarget failed: {exc2}")
                    continue
            else:
                print(f"[XING->GMR] retarget failed: {exc}")
                continue

        if qpos.shape[0] < 7 + 29:
            print(f"[XING->GMR] invalid qpos length: {qpos.shape[0]}")
            continue

        root_pos = qpos[:3].astype(np.float32)
        root_quat = qpos[3:7].astype(np.float32)
        qpos_dof = qpos[7:7 + 29].astype(np.float32)
        if joint_limits is not None:
            lower, upper = joint_limits
            qpos_dof = np.clip(qpos_dof, lower, upper)
            if not warned_clamp:
                print("[XING->GMR] qpos clamped to robot_config limits")
                warned_clamp = True

        if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_quat)):
            print("[XING->GMR] invalid root pose, skip")
            continue
        if not np.all(np.isfinite(qpos_dof)):
            print("[XING->GMR] invalid qpos, skip")
            continue
        norm = np.linalg.norm(root_quat)
        if norm < 1e-6:
            print("[XING->GMR] invalid root quaternion norm")
            continue
        root_quat = root_quat / norm

        msg = TeleopQposMessage(
            ts_ms=int(now * 1000),
            seq=seq,
            root_pos=root_pos,
            root_quat=root_quat,
            qpos=qpos_dof,
            valid=True,
            quality=None,
        )
        rds.publish(args.channel, msg.to_json())
        seq += 1
        last_pub = now


if __name__ == "__main__":
    main()
