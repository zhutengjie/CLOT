"""
Verify motion library CPU offload produces bit-for-bit identical results
compared to the default GPU-resident mode.

Requires a CUDA GPU and the g1 robot asset files.
Run with:
    PYTHONPATH=/root/CLOT-RL-fork pytest tests/test_motion_lib_offload.py -v
"""

import copy
import tempfile
import os

import numpy as np
import pytest
import torch
import joblib
from easydict import EasyDict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSET_ROOT = "humanoidverse/data/robots/g1/"
ASSET_FILE = "g1_23dof_lock_wrist_fitmotionONLY.xml"
NUM_JOINTS = 24
NUM_FRAMES = 60
FPS = 50


def _make_synthetic_motion(num_frames: int, num_joints: int, fps: int) -> dict:
    """Create a minimal synthetic motion clip dict."""
    rng = np.random.RandomState(42)
    return {
        "root_trans_offset": rng.randn(num_frames, 3).astype(np.float64) * 0.1
        + np.array([0.0, 0.0, 0.8]),
        "pose_aa": rng.randn(num_frames, num_joints, 3).astype(np.float32) * 0.05,
        "fps": fps,
    }


def _base_motion_cfg() -> EasyDict:
    """Return a minimal motion config matching the robot YAML structure."""
    return EasyDict(
        {
            "motion_lib_type": "ZTJ",
            "motion_file": "PLACEHOLDER",
            "asset": {
                "assetRoot": ASSET_ROOT,
                "assetFileName": ASSET_FILE,
            },
            "humanoid_type": "g1_23dof_lock_wrist",
            "bias_offset": False,
            "has_self_collision": True,
            "has_mesh": False,
            "has_jt_limit": False,
            "has_dof_subset": True,
            "has_upright_start": True,
            "has_smpl_pd_offset": False,
            "remove_toe": False,
            "motion_sym_loss": False,
            "big_ankle": True,
            "has_shape_obs": False,
            "has_shape_obs_disc": False,
            "has_shape_variation": False,
            "masterfoot": False,
            "freeze_toe": False,
            "freeze_hand": False,
            "box_body": True,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True,
            "nums_extend_bodies": 3,
            "extend_config": [
                {
                    "joint_name": "left_hand_link",
                    "parent_name": "left_elbow_link",
                    "pos": [0.25, 0.0, 0.0],
                    "rot": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "joint_name": "right_hand_link",
                    "parent_name": "right_elbow_link",
                    "pos": [0.25, 0.0, 0.0],
                    "rot": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "joint_name": "head_link",
                    "parent_name": "torso_link",
                    "pos": [0.0, 0.0, 0.42],
                    "rot": [1.0, 0.0, 0.0, 0.0],
                },
            ],
            "body_names": [
                "pelvis",
                "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
                "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
                "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
                "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
                "waist_yaw_link", "waist_roll_link", "torso_link",
                "left_shoulder_pitch_link", "left_shoulder_roll_link",
                "left_shoulder_yaw_link", "left_elbow_link",
                "right_shoulder_pitch_link", "right_shoulder_roll_link",
                "right_shoulder_yaw_link", "right_elbow_link",
            ],
            "dof_names": [
                "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
                "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
                "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
                "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
                "waist_yaw_link", "waist_roll_link", "torso_link",
                "left_shoulder_pitch_link", "left_shoulder_roll_link",
                "left_shoulder_yaw_link", "left_elbow_link",
                "right_shoulder_pitch_link", "right_shoulder_roll_link",
                "right_shoulder_yaw_link", "right_elbow_link",
            ],
            "cpu_offload": False,  # will be overridden per-instance
        }
    )


@pytest.fixture(scope="module")
def motion_pkl(tmp_path_factory):
    """Write a synthetic .pkl motion file and return its path."""
    tmp_dir = tmp_path_factory.mktemp("motions")
    path = str(tmp_dir / "test_motion.pkl")
    motion = _make_synthetic_motion(NUM_FRAMES, NUM_JOINTS, FPS)
    joblib.dump({"clip_0": motion}, path)
    return path


@pytest.fixture(scope="module")
def lib_pair(motion_pkl):
    """Create two MotionLibRobotZTJ instances: gpu-only and cpu-offload."""
    from humanoidverse.utils.motion_lib.motion_lib_robot_ztj import MotionLibRobotZTJ

    device = torch.device("cuda:0")
    num_envs = 8

    cfg_gpu = _base_motion_cfg()
    cfg_gpu.motion_file = motion_pkl
    cfg_gpu.cpu_offload = False

    cfg_off = copy.deepcopy(cfg_gpu)
    cfg_off.cpu_offload = True

    lib_gpu = MotionLibRobotZTJ(cfg_gpu, num_envs=num_envs, device=device)
    lib_off = MotionLibRobotZTJ(cfg_off, num_envs=num_envs, device=device)
    return lib_gpu, lib_off


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_motion_state_identical(lib_pair):
    """get_motion_state must return bit-for-bit identical dicts."""
    lib_gpu, lib_off = lib_pair
    device = lib_gpu._device

    torch.manual_seed(0)
    motion_ids = lib_gpu.sample_motions(8)
    motion_times = lib_gpu.sample_time(motion_ids)

    res_gpu = lib_gpu.get_motion_state(motion_ids, motion_times)
    res_off = lib_off.get_motion_state(motion_ids, motion_times)

    assert set(res_gpu.keys()) == set(res_off.keys()), "Key mismatch"
    for key in res_gpu:
        a, b = res_gpu[key], res_off[key]
        assert a.device == b.device, f"{key}: device mismatch {a.device} vs {b.device}"
        assert torch.equal(a, b), f"{key}: values differ"


def test_get_motion_state_simple_identical(lib_pair):
    """get_motion_state_simple must return bit-for-bit identical dicts."""
    lib_gpu, lib_off = lib_pair

    torch.manual_seed(1)
    motion_ids = lib_gpu.sample_motions(8)
    motion_times = lib_gpu.sample_time(motion_ids)

    res_gpu = lib_gpu.get_motion_state_simple(motion_ids, motion_times)
    res_off = lib_off.get_motion_state_simple(motion_ids, motion_times)

    assert set(res_gpu.keys()) == set(res_off.keys()), "Key mismatch"
    for key in res_gpu:
        a, b = res_gpu[key], res_off[key]
        assert torch.equal(a, b), f"{key}: values differ"


def test_multiple_random_rounds(lib_pair):
    """Run several rounds with random motion_ids and times."""
    lib_gpu, lib_off = lib_pair

    for seed in range(10):
        torch.manual_seed(seed + 100)
        motion_ids = lib_gpu.sample_motions(8)
        motion_times = lib_gpu.sample_time(motion_ids)

        res_gpu = lib_gpu.get_motion_state(motion_ids, motion_times)
        res_off = lib_off.get_motion_state(motion_ids, motion_times)

        for key in res_gpu:
            assert torch.equal(res_gpu[key], res_off[key]), (
                f"Round {seed}, key '{key}': values differ"
            )


def test_gpu_lib_tensors_on_device(lib_pair):
    """With cpu_offload=False, stored tensors should live on GPU."""
    lib_gpu, _ = lib_pair
    for name in ("gts", "grs", "lrs", "gvs", "gavs", "dvs"):
        t = getattr(lib_gpu, name)
        assert t.device.type == "cuda", f"{name} should be on cuda, got {t.device}"


def test_offload_lib_tensors_on_cpu(lib_pair):
    """With cpu_offload=True, stored tensors should be on CPU (pinned)."""
    _, lib_off = lib_pair
    for name in ("gts", "grs", "lrs", "gvs", "gavs", "dvs"):
        t = getattr(lib_off, name)
        assert t.device.type == "cpu", f"{name} should be on cpu, got {t.device}"
        assert t.is_pinned(), f"{name} should be pinned memory"
