import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg,EventTermCfg
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

def reset_and_fold_legs(
    env, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_angle: float = 1.4835, # 85도 (rad)
    warmup_steps: int = 10      # 접어 올릴 시뮬레이션 스텝 수
):
    """4-bar linkage 찢어짐 방지: 0도 스폰 후 물리 연산을 거치며 85도로 접는 함수"""
    robot = env.scene[asset_cfg.name]
    
    # 1. 안전한 Default State로 텔레포트
    robot.reset(env_ids)
    
    # 2. 관절 인덱스 가져오기 (Base_LeftActiveThigh, Base_RightActiveThigh)
    joint_ids, _ = robot.find_joints(["Base_LeftActiveThigh", "Base_RightActiveThigh"])
    
    # 3. PhysX 연산을 돌리며 Target Angle을 0 -> 85도로 부드럽게 증가
    for step in range(1, warmup_steps + 1):
        current_target = (step / warmup_steps) * target_angle
        
        targets = torch.full(
            (len(env_ids), len(joint_ids)), 
            current_target, 
            device=env.device
        )
        
        # 💥 [수정 완료] joint_ids=[...] 대신 변수 joint_ids 전달 💥
        robot.set_joint_position_target(targets, joint_ids=joint_ids, env_ids=env_ids)
        
        # PhysX Solver 실행하여 Closed-Loop Constraint 계산
        env.sim.step()

