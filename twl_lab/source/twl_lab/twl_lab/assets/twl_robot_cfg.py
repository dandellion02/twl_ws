import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# assets 폴더 내의 USD 파일 경로 지정
MY_ROBOT_USD_PATH = f"source/twl_lab/twl_lab/assets/twl_robot.usd"  # 👈 로봇 USD 파일명으로 변경하세요!

TWL_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MY_ROBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.275),  # 초기 위치 높이
        rot=(1.0, 0.0, 0.0, 0.0),
        # 초기 관절 위치 지정
        joint_pos={
            "Base_LeftActiveThigh": 0.0,
            "Base_RightActiveThigh": 0.0,
            ".*Wheel": 0.0,
        },
    ),
    actuators={
        # 1. 능동 다리 모터 관절 (2개만 지정)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Base_LeftActiveThigh", "Base_RightActiveThigh"], 
            effort_limit=40.0,   # 최대 토크 (Nm)
            velocity_limit=10.0, # 최대 속도 (rad/s)
            stiffness=40.0,      # P Gain
            damping=1.0,         # D Gain
        ),
        # 2. 바퀴 모터 관절 (2개 지정)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["LeftCalf_LeftWheel", "RightCalf_RightWheel"], 
            effort_limit=2.5,
            velocity_limit=33.5,
            stiffness=0.0,
            damping=0.20,
            armature=0.00202,
            friction=0.0,
        ),
    },
)