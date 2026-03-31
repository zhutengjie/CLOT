from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

import glob
import os.path as osp
import numpy as np
import joblib
import torch
import random
from typing import Dict, Any, Optional

from humanoidverse.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from isaac_utils.rotations import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
    calc_heading_quat_inv
)

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


def _calc_frame_blend(time, len, num_frames, dt):
    time = time.clone()
    phase = time / len
    phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
    time[time < 0] = 0

    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
    
    return frame_idx0, frame_idx1, blend


_BLEND_EPS: float = 1e-5

@torch.jit.script
def _lerp(x0: torch.Tensor, x1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:

    if bool(blend.max() < 1e-5):
        return x0
    return (1.0 - blend) * x0 + blend * x1


def _local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape
    dof_pos = quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)
    

def forbidden(fn):
    def wrapper(*args, **kwargs):
        raise RuntimeError("You are NOT ALLOWED to call it.")
    return wrapper

# @time_prot_cls_dec_mlb
# TimePortion: 6%
class MotionLibBase():
    ############################################################ SETUP ############################################################
    
    def __init__(self, motion_lib_cfg, num_envs, device):

        def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
            self.fix_height = fix_height
            self.multi_thread = multi_thread
            
            #### Termination history
            self._curr_motion_ids = None
            self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
            self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
            self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)
        self.cpu_offload = self.m_cfg.get("cpu_offload", False)

        self.num_envs = num_envs
        self._device = device
        # self.mesh_parsers = None
        self.has_action = False
        self.has_contact_mask: Optional[str] = None
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        
        self.load_data(self.m_cfg.motion_file)
        setup_constants(self, fix_height = False,  multi_thread = False)
        
        # if flags.real_traj:
        #     self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        self.load_motions()
        return
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load  # data_list 是一个字典， keys是文件名
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            # Save original list for directory mode before converting to array
            _motion_data_load_list = list(self._motion_data_load)
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)

        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(_motion_data_load_list[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def _store_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Store tensor on pinned CPU memory (offload) or device."""
        if self.cpu_offload:
            return t.pin_memory()
        return t.to(self._device)

    def _fetch(self, t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Index into a stored tensor, moving result to device if offloaded."""
        if self.cpu_offload:
            return t[idx.cpu()].to(self._device)
        return t[idx]

    def load_motions(self,
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None):
        assert target_heading is None, "Not Allowed to use target_heading!"
        # import ipdb; ipdb.set_trace()

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False
        _motion_actions = []
        _motion_contact_masks = []
        
        # if flags.real_traj:
        #     self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)

        self.curr_motion_keys = self._motion_data_keys
        self.curr_motion_ids = self._motion_data_keys

        motion_data_list = self._motion_data_list

        res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
        
        for f in track(range(len(res_acc)), description="Loading motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.has_contact_mask:
                _motion_contact_masks.append(curr_motion.contact_mask)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
        

        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)

        self._motion_aa = self._store_tensor(torch.from_numpy(np.concatenate(_motion_aa)).float())

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        if self.has_action:
            self._motion_actions = self._store_tensor(torch.cat(_motion_actions, dim=0).float())
        if self.has_contact_mask:
            self._motion_contact_masks = self._store_tensor(torch.from_numpy(np.concatenate(_motion_contact_masks)).float())
        self._num_motions = len(motions)

        self.gts = self._store_tensor(torch.cat([m.global_translation for m in motions], dim=0).float())
        self.grs = self._store_tensor(torch.cat([m.global_rotation for m in motions], dim=0).float())
        self.lrs = self._store_tensor(torch.cat([m.local_rotation for m in motions], dim=0).float())
        self.grvs = self._store_tensor(torch.cat([m.global_root_velocity for m in motions], dim=0).float())
        self.gravs = self._store_tensor(torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float())
        self.gavs = self._store_tensor(torch.cat([m.global_angular_velocity for m in motions], dim=0).float())
        self.gvs = self._store_tensor(torch.cat([m.global_velocity for m in motions], dim=0).float())
        self.dvs = self._store_tensor(torch.cat([m.dof_vels for m in motions], dim=0).float())

        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = self._store_tensor(torch.cat([m.global_translation_extend for m in motions], dim=0).float())
            self.grs_t = self._store_tensor(torch.cat([m.global_rotation_extend for m in motions], dim=0).float())
            self.gvs_t = self._store_tensor(torch.cat([m.global_velocity_extend for m in motions], dim=0).float())
            self.gavs_t = self._store_tensor(torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float())

        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = self._store_tensor(torch.cat([m.dof_pos for m in motions], dim=0).float())
        if flags.real_traj:
            self.q_gts = self._store_tensor(torch.cat(self.q_gts, dim=0).float())
            self.q_grs = self._store_tensor(torch.cat(self.q_grs, dim=0).float())
            self.q_gavs = self._store_tensor(torch.cat(self.q_gavs, dim=0).float())
            self.q_gvs = self._store_tensor(torch.cat(self.q_gvs, dim=0).float())
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints
        
        total_len = self.get_total_length()
        logger.info(f"Loaded {self._num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def load_motion_with_skeleton(self,
                                  motion_data_list: np.ndarray,
                                  fix_height,
                                  target_heading,
                                  max_len):
        @forbidden
        def fix_trans_height(self, pose_aa, trans, fix_height_mode):
            if fix_height_mode == FixHeightMode.no_fix:
                return trans, 0
            with torch.no_grad():
                mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
                height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
                trans[..., 2] -= height_diff
                
                return trans, height_diff
            
            
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file:Dict[str, Any] = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                forbidden(lambda :0)()
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            if False: 
            # if True: 
                print("DEBUG: !!!! MotionLibBase: rebase root_trans_offset & root_rot_offset")
                curr_file['root_trans_offset'][:] = np.array([0, 0, 0.8], dtype=np.float64)
                target_heading = np.array([0, 0, 0, 1.0])
                # curr_file['root_rot']= rebase_yaw(curr_file['root_rot'])
                # breakpoint()
                
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():
                self.has_action = True
            if "contact_mask" in curr_file.keys():
                contact_shape = curr_file['contact_mask'].shape
                assert len(contact_shape) ==2 and contact_shape[0] == seq_len
                self._contact_size = contact_shape[1]
                if contact_shape[1] == 2:
                    self.has_contact_mask = "point"
                else:
                    raise ValueError(f"Contact mask shape {contact_shape} is not supported")
            
            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape

            if not target_heading is None:
                from scipy.spatial.transform import Rotation as sRot
                # forbidden(lambda :0)()
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ]),True))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans.to(torch.float64), torch.from_numpy(heading_delta.as_matrix().squeeze().T))

            if self.mesh_parsers is None:
                logger.error("No mesh parser found")
            # trans, trans_fix = fix_trans_height(self, pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
            curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
            curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
            # add "action" to curr_motion
            if self.has_action:
                curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
            if self.has_contact_mask:
                curr_motion.contact_mask = to_torch(curr_file['contact_mask']).clone()[start:end]
                
            res[f] = (curr_file, curr_motion)
        return res  # res 是字典，键值是数字，返回的值是元祖，数目等于num_envs

    def sample_motions(self, n, start_idx=0, random_sample=True, adaptive_sampling=False, adaptive_prob=None):

        if random_sample:
            motion_ids = torch.multinomial(self._sampling_prob, num_samples=n, replacement=True).to(self._device)
        elif adaptive_sampling:
            motion_ids = torch.multinomial(adaptive_prob, num_samples=n, replacement=True).to(self._device)
        else:
            motion_ids = torch.remainder(torch.arange(n) + start_idx, self._num_unique_motions ).to(self._device)

        return motion_ids

    ############################################################ ACCESS ############################################################
    
    def get_motion_actions(self, motion_ids, motion_times):
        raise RuntimeError("You Should not call it.")
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = _calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action = self._motion_actions[f0l]
        return action

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = _calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        _d = self._device

        if "dof_pos" in self.__dict__:
            local_rot0 = self._fetch(self.dof_pos, f0l)
            local_rot1 = self._fetch(self.dof_pos, f1l)
        else:
            local_rot0 = self._fetch(self.lrs, f0l)
            local_rot1 = self._fetch(self.lrs, f1l)

        body_vel0 = self._fetch(self.gvs, f0l)
        body_vel1 = self._fetch(self.gvs, f1l)

        body_ang_vel0 = self._fetch(self.gavs, f0l)
        body_ang_vel1 = self._fetch(self.gavs, f1l)

        rg_pos0 = self._fetch(self.gts, f0l)
        rg_pos1 = self._fetch(self.gts, f1l)

        dof_vel0 = self._fetch(self.dvs, f0l)
        dof_vel1 = self._fetch(self.dvs, f1l)

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        skip_blend = blend_exp.max() < _BLEND_EPS  # 整个 batch 的 blend 都极小，直接用 f0 帧

        if offset is None:
            rg_pos = _lerp(rg_pos0, rg_pos1, blend_exp)
        else:
            rg_pos = _lerp(rg_pos0, rg_pos1, blend_exp) + offset[..., None, :]

        body_vel = _lerp(body_vel0, body_vel1, blend_exp)
        body_ang_vel = _lerp(body_ang_vel0, body_ang_vel1, blend_exp)

        if "dof_pos" in self.__dict__: # Robot Joints
            dof_vel = _lerp(dof_vel0, dof_vel1, blend)
            dof_pos = _lerp(local_rot0, local_rot1, blend)
        else:
            dof_vel = _lerp(dof_vel0, dof_vel1, blend_exp)
            if skip_blend:
                local_rot = local_rot0
            else:
                local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = _local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self._fetch(self.grs, f0l)
        rb_rot1 = self._fetch(self.grs, f1l)
        rb_rot = rb_rot0 if skip_blend else slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self._fetch(self.gts_t, f0l)
            rg_pos_t1 = self._fetch(self.gts_t, f1l)

            rg_rot_t0 = self._fetch(self.grs_t, f0l)
            rg_rot_t1 = self._fetch(self.grs_t, f1l)

            body_vel_t0 = self._fetch(self.gvs_t, f0l)
            body_vel_t1 = self._fetch(self.gvs_t, f1l)

            body_ang_vel_t0 = self._fetch(self.gavs_t, f0l)
            body_ang_vel_t1 = self._fetch(self.gavs_t, f1l)
            if offset is None:
                rg_pos_t = _lerp(rg_pos_t0, rg_pos_t1, blend_exp)
            else:
                rg_pos_t = _lerp(rg_pos_t0, rg_pos_t1, blend_exp) + offset[..., None, :]
            rg_rot_t = rg_rot_t0 if skip_blend else slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = _lerp(body_vel_t0, body_vel_t1, blend_exp)
            body_ang_vel_t = _lerp(body_ang_vel_t0, body_ang_vel_t1, blend_exp)
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self._fetch(self.q_gavs, f0l), self._fetch(self.q_gavs, f1l)
            q_rb_rot0, q_rb_rot1 = self._fetch(self.q_grs, f0l), self._fetch(self.q_grs, f1l)
            q_rg_pos0, q_rg_pos1 = self._fetch(self.q_gts, f0l), self._fetch(self.q_gts, f1l)
            q_body_vel0, q_body_vel1 = self._fetch(self.q_gvs, f0l), self._fetch(self.q_gvs, f1l)

            q_ang_vel = _lerp(q_body_ang_vel0, q_body_ang_vel1, blend_exp)
            q_rb_rot = q_rb_rot0 if skip_blend else slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = _lerp(q_rg_pos0, q_rg_pos1, blend_exp)
            q_body_vel = _lerp(q_body_vel0, q_body_vel1, blend_exp)
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel


        if self.has_contact_mask:
            contact0, contact1 = self._fetch(self._motion_contact_masks, f0l), self._fetch(self._motion_contact_masks, f1l)
            contact = _lerp(contact0, contact1, blend)
            
            return_dict["contact_mask"] = contact
            

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._fetch(self._motion_aa, f0l),
            "motion_bodies": self._motion_bodies[motion_ids],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
        })
        return return_dict
    
    def get_motion_state_simple(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = _calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        _d = self._device

        if "dof_pos" in self.__dict__:
            local_rot0 = self._fetch(self.dof_pos, f0l)
            local_rot1 = self._fetch(self.dof_pos, f1l)
        else:
            local_rot0 = self._fetch(self.lrs, f0l)
            local_rot1 = self._fetch(self.lrs, f1l)

        body_vel0 = self._fetch(self.gvs, f0l)
        body_vel1 = self._fetch(self.gvs, f1l)

        body_ang_vel0 = self._fetch(self.gavs, f0l)
        body_ang_vel1 = self._fetch(self.gavs, f1l)

        rg_pos0 = self._fetch(self.gts, f0l)
        rg_pos1 = self._fetch(self.gts, f1l)

        dof_vel0 = self._fetch(self.dvs, f0l)
        dof_vel1 = self._fetch(self.dvs, f1l)

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = _lerp(rg_pos0, rg_pos1, blend_exp)
        else:
            rg_pos = _lerp(rg_pos0, rg_pos1, blend_exp) + offset[..., None, :]

        # body_vel = _lerp(body_vel0, body_vel1, blend_exp)
        # body_ang_vel = _lerp(body_ang_vel0, body_ang_vel1, blend_exp)

        # if "dof_pos" in self.__dict__: # Robot Joints
        #     dof_vel = _lerp(dof_vel0, dof_vel1, blend)
        #     dof_pos = _lerp(local_rot0, local_rot1, blend)
        # else:
        #     dof_vel = _lerp(dof_vel0, dof_vel1, blend_exp)
        #     local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        #     dof_pos = _local_rotation_to_dof_smpl(local_rot)

        # rb_rot0 = self.grs[f0l]
        # rb_rot1 = self.grs[f1l]
        # rb_rot = rb_rot0 if blend_exp.max() < _BLEND_EPS else slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self._fetch(self.gts_t, f0l)
            rg_pos_t1 = self._fetch(self.gts_t, f1l)
            
            # rg_rot_t0 = self.grs_t[f0l]
            # rg_rot_t1 = self.grs_t[f1l]
            
            # body_vel_t0 = self.gvs_t[f0l]
            # body_vel_t1 = self.gvs_t[f1l]
            
            # body_ang_vel_t0 = self.gavs_t[f0l]
            # body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = _lerp(rg_pos_t0, rg_pos_t1, blend_exp)
            else:
                rg_pos_t = _lerp(rg_pos_t0, rg_pos_t1, blend_exp) + offset[..., None, :]
            # rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            # body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            # body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            # rg_rot_t = rb_rot
            # body_vel_t = body_vel
            # body_ang_vel_t = body_ang_vel
        
        # if flags.real_traj:
        #     q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
        #     q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
        #     q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
        #     q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

        #     q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
        #     q_rb_rot = slerp(q_rb_rot0, q_rb_rot1, blend_exp)
        #     q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
        #     q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
        #     rg_pos[:, self.track_idx] = q_rg_pos
        #     rb_rot[:, self.track_idx] = q_rb_rot
        #     body_vel[:, self.track_idx] = q_body_vel
        #     body_ang_vel[:, self.track_idx] = q_ang_vel


        # if self.has_contact_mask:
        #     contact0, contact1 = self._motion_contact_masks[f0l], self._motion_contact_masks[f1l]
        #     contact = (1.0 - blend) * contact0 + blend * contact1
            
        #     return_dict["contact_mask"] = contact
            

        return_dict.update({
            # "root_pos": rg_pos[..., 0, :].clone(),
            # "root_rot": rb_rot[..., 0, :].clone(),
            # "dof_pos": dof_pos.clone(),
            # "root_vel": body_vel[..., 0, :].clone(),
            # "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            # "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            # "motion_aa": self._motion_aa[f0l],
            # "motion_bodies": self._motion_bodies[motion_ids],
            # "rg_pos": rg_pos,
            # "rb_rot": rb_rot,
            # "body_vel": body_vel,
            # "body_ang_vel": body_ang_vel,
            "rg_pos_t": rg_pos_t,
            # "rg_rot_t": rg_rot_t,
            # "body_vel_t": body_vel_t,
            # "body_ang_vel_t": body_ang_vel_t,
        })
        return return_dict
    
    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        
        dt = 1.0 / self._sim_fps
        motion_time = torch.floor(motion_time / dt) * dt
        return motion_time.to(self._device)



    
    
class MotionLibRobotZTJ(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        return