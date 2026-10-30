# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """로봇 몸체의 XY 평면 선속도(L2 norm) 감점 함수"""
    asset: Articulation = env.scene[asset_cfg.name]
    # asset.data.root_lin_vel_b 은 (num_envs, 3) 크기의 [vx, vy, vz] 선속도 텐서입니다.
    # index 0, 1이 XY 평면 선속도입니다.
    return torch.square(asset.data.root_lin_vel_b[:, 0]) + torch.square(asset.data.root_lin_vel_b[:, 1])


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """z축(Yaw) 회전 각속도 제곱 페널티 함수"""
    asset = env.scene[asset_cfg.name]
    # asset.data.root_ang_vel_b 은 (num_envs, 3) 크기의 [wx, wy, wz] 각속도 텐서입니다.
    # index 2가 z축(Yaw) 각속도입니다.
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def action_asymmetry_l2(env: ManagerBasedRLEnv, action_name: str = "joint_eff") -> torch.Tensor:
    """좌우 바퀴 Action(토크) 차이 비대칭 페널티 함수"""
    # ActionManager에서 현재 신경망이 낸 action 텐서를 가져옵니다.
    action = env.action_manager.get_term(action_name).raw_actions
    if action.shape[1] >= 2:
        return torch.square(action[:, 0] - action[:, 1])
    return torch.zeros(env.num_envs, device=env.device)


def joint_align_l1(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Base_LeftActiveThigh", "Base_RightActiveThigh"])
) -> torch.Tensor:
    """좌우 허벅지 관절 각도 차이 비대칭 페널티 함수"""
    asset = env.scene[asset_cfg.name]
    # 지정된 관절(예: 좌우 허벅지)의 현재 위치 [num_envs, 2]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    if joint_pos.shape[1] >= 2:
        # 좌우 관절 각도의 차이 절대값 |q_left - q_right|
        return torch.abs(joint_pos[:, 0] - joint_pos[:, 1])
    return torch.zeros(env.num_envs, device=env.device)


def pitch_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """앞뒤 기울기(Pitch, gx)만 감점하는 함수"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 0] 이 gx (Pitch)
    gx = asset.data.projected_gravity_b[:, 0]
    return torch.square(gx)


def roll_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """좌우 기울기(Roll, gy)만 감점하는 함수"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 1] 이 gy (Roll)
    gy = asset.data.projected_gravity_b[:, 1]
    return torch.square(gy)


def base_height_direct(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """로봇 Base의 현재 Z축 높이값(m)을 그대로 반환하여 높을수록 +점수를 주는 함수"""
    robot: Articulation = env.scene[asset_cfg.name]
    # root_pos_w: [num_envs, 3] -> Z축 위치(index 2) 추출
    return robot.data.root_pos_w[:, 2]


def joint_stand(env: ManagerBasedRLEnv, max_angle: float = 1.4835, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """로봇이 서있을수록 +점수를 주는 함수 (좌우 허벅지 관절 각도 평균값을 반환)"""
    robot = env.scene[asset_cfg.name]
    
    # 다리 관절(Thigh)의 현재 각도 추출
    joint_ids, _ = robot.find_joints(["Base_LeftActiveThigh", "Base_RightActiveThigh"])
    thigh_positions = robot.data.joint_pos[:, joint_ids]  # Shape: [num_envs, 2]
    mean_thigh_angle = torch.mean(thigh_positions, dim=-1)  # 양쪽 다리 평균 각도
    
    # 다리를 필수록(각도가 작을수록) +보상, 앉아있을 때 85도(1.4835 rad)
    return max_angle - mean_thigh_angle

def roll_by_cmd_yaw(
    env: ManagerBasedRLEnv, 
    k_roll: float = 0.25,      # 각속도당 Roll 기울임 비율 계수
    max_roll: float = 0.5,     # 최대 허용 Roll 각도 (약 28.6도, 안전 한계)
    sigma: float = 0.15,       # 가우시안 보상 폭
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    커맨드 각속도의 방향(+/-)에 맞추어 Roll 방향(+/-)을 유도하는 보상 함수
    - 좌회전(cmd_yaw > 0) -> Roll > 0 (왼쪽 다리 굽힘)
    - 우회전(cmd_yaw < 0) -> Roll < 0 (오른쪽 다리 굽힘)
    """
    robot = env.scene[asset_cfg.name]
    
    # 현재 로봇의 Roll 각도 (rad) - 부호 유지 (+: 좌측 기울임, -: 우측 기울임)
    roll, _, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    
    # Command Yaw 각속도 (rad/s) 가져오기 (+: 좌회전, -: 우회전)
    cmd_yaw = env.command_manager.get_command("base_velocity")[:, 2]

    # 부호를 반영한 Target Roll 계산 (Max Roll 제한 포함)
    target_roll = -k_roll * cmd_yaw # (부호 반전: 좌회전 -> Roll > 0, 우회전 -> Roll < 0)
    target_roll = torch.clamp(target_roll, min=-max_roll, max=max_roll)

    # 실제 roll과 target_roll의 부호가 일치하는지 평가
    roll_error = torch.square(roll - target_roll)
    gaussian_reward = torch.exp(-roll_error / (sigma ** 2))

    # 부호 일치도 평가 (동일 방향으로 기울이고 있으면 가중 보상)
    directional_alignment = torch.clamp(roll * target_roll, min=0.0)
    
    # 회전 커맨드가 유효할 때만 방향성 보상 활성화
    active_mask = (torch.abs(cmd_yaw) > 0.1).float()
    
    return gaussian_reward + active_mask * directional_alignment