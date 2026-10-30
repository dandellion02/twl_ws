# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from twl_lab.assets.twl_robot_cfg import TWL_ROBOT_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from . import mdp

##
# Pre-defined configs
##

#from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##


@configclass
class TwlLabSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0)),
    )

    # robot
    robot: ArticulationCfg = TWL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

#@configclass
#class CommandsCfg:
#    """Command terms for the MDP."""
#    base_velocity = mdp.DiscreteVelocityCommandCfg(
#        asset_name="robot",
#        resampling_time_range=(1000.0, 1000.0), # 에피소드 리셋 시에만 바뀌도록 설정
#    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

#    # 다리는 위치 제어(Position)
#    joint_pos = mdp.JointPositionToLimitsActionCfg(
#        asset_name="robot", 
#        joint_names=["Base_LeftActiveThigh", "Base_RightActiveThigh"],
#        rescale_to_limits=True,
#    )
    
    # 바퀴는 토크 제어(Effort/Torque)
#    joint_eff = mdp.DelayedJointEffortActionCfg(
#        asset_name="robot", 
#        joint_names=["LeftCalf_LeftWheel", "RightCalf_RightWheel"],
#        scale=2.5,
#        preserve_order=True,  # joint_names 순서대로 action값 적용
#        delay_steps=1,  # action delay step 수 (physics step 기준)
#    )   
   

    wheel_vel = mdp.DelayedJointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["LeftCalf_LeftWheel", "RightCalf_RightWheel"],
        scale=20.0,
        use_default_offset=False,  # 기본 offset값(0 rad/s) 사용 여부
        preserve_order=True,  # joint_names 순서대로 action값 적용
        delay_steps=1,  # action delay step 수 (physics step 기준)
        lpf_alpha=0.9,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    ## Policy 관측치 (신경망 입력)
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 로봇 차체 각속도 (Roll, Pitch, Yaw 속도)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=UniformNoiseCfg(
                n_min = -0.03,
                n_max = 0.03
            ),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # 로봇 몸체 기울기 (중력 벡터 투영: 직립 균형 감지에 필수)
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(
                n_min = -0.003,
                n_max = 0.003
            ),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
#        # 다리 각도
#        joint_pos = ObsTerm(
#            func=mdp.joint_pos_rel, 
#            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Base_LeftActiveThigh", "Base_RightActiveThigh"])}
#        )
        
        # 바퀴 각속도
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(
                n_min = -0.03,
                n_max = 0.03
            ), 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LeftCalf_LeftWheel", "RightCalf_RightWheel"])}
        )

        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # 이전 Action 기록
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True  # 관측치에 노이즈 추가 여부
            self.concatenate_terms = True
    
    # Critic 관측치 (시뮬레이션 내부의 privileged information 포함)
    @configclass
    class CriticCfg(ObsGroup):
        
        # --- Actor 관측치와 동일한 항목 ---
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})        
        proj_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        #joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Base_LeftActiveThigh", "Base_RightActiveThigh"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LeftCalf_LeftWheel", "RightCalf_RightWheel"])})
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=mdp.last_action)

        # --- Privileged Information ---
        
        # 로봇 차체 선속도 (x, y축 이동 속도)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        # 지면 물리 재질 속도/마찰력 (마찰계수, 반발계수 등)
        material_properties = ObsTerm(func=mdp.material_properties, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # 차체 질량 및 무게중심 오차 (Rigid Body의 물리 특성)
        robot_physical_properties = ObsTerm(func=mdp.robot_physical_properties, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})

        # 외부 외란 힘/토크 (push_robot 등으로 가해진 순간 외란)
        external_torques = ObsTerm(func=mdp.net_external_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class HistoryCfg(PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            # 최근 20스텝(20 * 8ms = 0.16s) 동안의 Policy 관측치를 링 버퍼로 저장
            self.history_length = 20
            # [num_envs, history_length, obs_dim] -> [num_envs, history_length * obs_dim]로 직렬화(1D 평탄화)
            self.flatten_history_dim = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    history: HistoryCfg = HistoryCfg()


@configclass
class EventCfg:
    #-------Domain randomization events -----------------
    
    # 마찰력 랜덤화 (타일 ~ 카펫 수준)
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.3, 1.2),  
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    # 로봇 링크 질량(Mass) 오차 부여 (배터리/센서 탑재 오차 대비)
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.1), # +-10% 질량 변동
            "operation": "scale",
        },
    )

    # 로봇 무게중심(CoM: Center of Mass) 위치 오차 부여
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            # (x, y, z) 오차 범위 (단위: 미터)
            "com_range": {
                "x": (-0.015, 0.015),
                "y": (-0.015, 0.015),
                "z": (-0.015, 0.015),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # 주기적인 외란 충격 (Push Robot) - 일정 시간마다 밀기
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 5.0), # 4~5초마다 한 번씩 툭 밀기
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}, # 순간 선속도 변화 1.5m/s 이하
        },
    )

    
    #-------Configuration for events (Reset state)-------
    
    # 에피소드 리셋 시 로봇의 초기 위치, 기울기(Orientation), 속도에 오차 부여
    reset_base_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),      
                "pitch": (-0.26, 0.26),  # 약 +-15도 기울기 오차 (앞뒤 - 바퀴 로봇 균형 학습 핵심)
                "yaw": (0.0, 0.0),    
            },
            "velocity_range": {
                "x": (-0.1, 0.1),        # 초기 x축 선속도 오차 (m/s)
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),     
                "pitch": (-0.2, 0.2),    # 초기 앞뒤 흔들림 각속도 오차
                "yaw": (0.0, 0.0),      # 초기 좌우 흔들림 각속도 오차
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 4-bar linkage 부드러운 접기 이벤트 등록(서있는 상태에서 학습시킬 경우 주석처리 할 것)
    #reset_and_fold_legs_event = EventTerm(
    #    func=mdp.reset_and_fold_legs,
    #    mode="reset",  # 에피소드 리셋 시 실행
    #    params={
    #        "target_angle": 1.4835,  # 85도
    #        "warmup_steps": 5,    # 5스텝에 걸쳐 물리 연산하며 접음
    #    },
    #)
    
def override_value(env, env_ids, data, value, num_steps):
        """외란 속도 범위를 동적으로 변경하는 함수"""
        if env.common_step_counter > num_steps:
            return value
        return mdp.modify_term_cfg.NO_CHANGE  # 변경 없음


@configclass
class CurriculumsCfg:
    """Curriculum settings for the MDP."""
    
    # 외란 속도 범위 변경 (x축 방향)
    modify_disturbance_velocity_1st = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.push_robot.params.velocity_range.x",
            "modify_fn": override_value,
            "modify_params": {"value": (-0.5, 0.5), "num_steps": 200*16}, # 200 iterations 이후 (4096*16 = 65536 steps)
        }
    )

    modify_disturbance_velocity_2nd = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.push_robot.params.velocity_range.x",
            "modify_fn": override_value,
            "modify_params": {"value": (-1.0, 1.0), "num_steps": 500*16}, # 500 iterations 이후
        }
    )                                

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 생존 보상 (오래 서 있을수록 +점수)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # 쓰러짐 감점 (종료 조건 도달 시 마이너스 점수)
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # 직립 유지 보상: 로봇 상체가 z축(하늘)을 똑바로 바라보도록 유도
    upright = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 로봇 차체의 선운동 속도(x, y축 이동) 억제
    lin_vel_xy = RewTerm(
        func=mdp.lin_vel_xy_l2,  # Isaac Lab 기본 제공: base의 xy 선속도 제곱 페널티
        weight=-2.0,            # 이동하면 감점!
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 로봇 차체의 z축 회전운동(Yaw) 억제
    ang_vel_z = RewTerm(
        func=mdp.ang_vel_z_l2, # z축 회전 각속도 페널티
        weight=-5.0,            # 회전하면 감점!
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Roll 및 Pitch 회전 각속도 페널티 (차체가 앞뒤/좌우로 갸우뚱흔들리는 속도 억제)
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
#--------wheel motor 관련 ---------------------#

    # 과도한 토크/에너지 사용 감점 (부드러운 제어 유도)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
    )

#    # 양쪽 바퀴 토크(Action) 출력이 다를수록 페널티 (좌우 비대칭 구동 억제)
#    wheel_action_asymmetry = RewTerm(
#        func=mdp.action_asymmetry_l2,
#        weight=-0.5,
#        params={
#            "action_name": "joint_eff"  # ActionsCfg에 정의된 joint_eff 항 지정
#        },
#    )

#------------joint 관련---------------#

#    # 로봇 base의 높이가 높을수록 보상
#    base_height = RewTerm(
#        func=mdp.base_height_direct,
#        weight=-3.0,
#        params={"asset_cfg": SceneEntityCfg("robot")},
#    )   

#    joint_stand = RewTerm(
#        func=mdp.joint_stand,
#        weight=-2.0,
#        params={"asset_cfg": SceneEntityCfg("robot")}, 
#    )

#    # 양쪽 Leg 관절(허벅지) 각도가 다를수록 페널티 (좌우 대칭 유도)
#    leg_alignment = RewTerm(
#        func=mdp.joint_align_l1,
#        weight=-1.0,
#        params={
#            "asset_cfg": SceneEntityCfg(
#                "robot", 
#                joint_names=["Base_LeftActiveThigh", "Base_RightActiveThigh"]
#            )
#        },
#    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) 에피소드 시간 초과 (기본 10초)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) 로봇 넘어짐 판정: 중력 투영 z값이 낮아지면(너무 많이 기울어지면) 에피소드 종료
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.04}, # 약 60도 이상 기울어지면 사망
    )


##
# Environment configuration
##


@configclass
class TwlLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TwlLabSceneCfg = TwlLabSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    
    #commands : CommandsCfg = CommandsCfg()
    
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2  # 신경망(RL Agent)이 행동 명령을 내리는 주기를 결정하는 비율: 250Hz / 2 = 125Hz, 8ms 제어 주기
        self.episode_length_s = 10.0
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 0.004 # PhysX의 연산 주기 : 4ms (250Hz)
        self.sim.render_interval = self.decimation # 화면 렌더링 주기