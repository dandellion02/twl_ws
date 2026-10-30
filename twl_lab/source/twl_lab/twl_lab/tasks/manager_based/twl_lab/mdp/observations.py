# source/twl_lab/twl_lab/tasks/manager_based/twl_lab/mdp/observations.py

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def generated_commands_vel_yaw(env, command_name: str):
    """Return linear x velocity and yaw angular velocity command."""
    command = env.command_manager.get_command(command_name)

    # command = [vx, vy, wz]
    return command[:, [0, 2]]


def material_properties(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """로봇 타겟 바디/지면의 마찰계수(Static/Dynamic Friction) 정보 추출"""
    asset = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    # 텐서가 GPU(env.device)에 위치하도록 .to(env.device) 명시
    return materials.reshape(env.num_envs, -1).to(env.device)


def robot_physical_properties(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """차체의 질량(Mass) 및 CoM 오차 정보 추출"""
    asset = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    coms = asset.root_physx_view.get_coms()
    
    # 텐서 결합 및 GPU device 명시
    res = torch.cat([masses.reshape(env.num_envs, -1), coms.reshape(env.num_envs, -1)], dim=-1)
    return res.to(env.device)


def net_external_wrench(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """외부에서 가해진 순수 힘/토크 (Net External Force & Torque)"""
    asset = env.scene[asset_cfg.name]
    
    if hasattr(asset.data, "computed_external_force_wrench") and asset.data.computed_external_force_wrench is not None:
        return asset.data.computed_external_force_wrench[:, asset_cfg.body_ids[0], :].to(env.device)
    
    # device=env.device를 확실히 지정하여 GPU 텐서로 생성!
    return torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)