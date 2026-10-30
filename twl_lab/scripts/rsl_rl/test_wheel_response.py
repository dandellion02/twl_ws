import argparse
import torch

from isaaclab.app import AppLauncher


# --------------------------------------------------------
# Isaac Sim 실행 설정
# --------------------------------------------------------

parser = argparse.ArgumentParser(description="TWL wheel velocity step response test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --------------------------------------------------------
# Isaac Lab imports
# --------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from twl_lab.assets.twl_robot_cfg import TWL_ROBOT_CFG


# ========================================================
# 설정값
# ========================================================

SIM_DT = 0.004             # 4 ms = 250 Hz physics
TARGET_VEL = 20.0          # [rad/s]

SETTLE_TIME = 0.2          # step 전 0 rad/s 유지
TEST_TIME = 0.3            # step 이후 기록 시간


def main():

    # ----------------------------------------------------
    # SimulationContext
    # ----------------------------------------------------

    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args_cli.device,
    )

    sim = sim_utils.SimulationContext(sim_cfg)

    # 카메라 위치
    sim.set_camera_view(
        eye=(2.0, 2.0, 1.5),
        target=(0.0, 0.0, 0.5),
    )

    # ----------------------------------------------------
    # Ground
    # ----------------------------------------------------

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(
        "/World/Ground",
        ground_cfg,
    )

    # ----------------------------------------------------
    # Robot spawn
    # ----------------------------------------------------

    robot_cfg = TWL_ROBOT_CFG.replace(
        prim_path="/World/Robot"
    )

    robot = Articulation(robot_cfg)

    # ----------------------------------------------------
    # Light
    # ----------------------------------------------------

    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0
    )

    light_cfg.func(
        "/World/Light",
        light_cfg,
    )

    # ----------------------------------------------------
    # Simulation initialize
    # ----------------------------------------------------

    sim.reset()

    print("---------------------------------------")
    print("Simulation initialized")
    print("---------------------------------------")

    # ----------------------------------------------------
    # Wheel joint IDs 찾기
    # ----------------------------------------------------

    left_ids, _ = robot.find_joints(
        "LeftCalf_LeftWheel"
    )

    right_ids, _ = robot.find_joints(
        "RightCalf_RightWheel"
    )

    left_id = left_ids[0]
    right_id = right_ids[0]

    print(f"Left wheel joint id  : {left_id}")
    print(f"Right wheel joint id : {right_id}")

    print("\n========= RUNTIME DOF PROPERTIES =========")

    print("max velocities:")
    print(robot.root_physx_view.get_dof_max_velocities())
    
    print("max forces:")
    print(robot.root_physx_view.get_dof_max_forces())
    
    print("stiffness:")
    print(robot.root_physx_view.get_dof_stiffnesses())
    
    print("damping:")
    print(robot.root_physx_view.get_dof_dampings())
    
    print("friction:")
    print(robot.root_physx_view.get_dof_friction_coefficients())
    
    print("armature:")
    print(robot.root_physx_view.get_dof_armatures())
    
    print("==========================================\n")

    # ----------------------------------------------------
    # 테스트 시간 계산
    # ----------------------------------------------------

    total_time = SETTLE_TIME + TEST_TIME
    total_steps = int(total_time / SIM_DT)

    settle_steps = int(SETTLE_TIME / SIM_DT)

    # ----------------------------------------------------
    # CSV header
    # ----------------------------------------------------

    print()
    print("time,target,left_vel,right_vel,left_tau,right_tau")
    
    # ----------------------------------------------------
    # Simulation loop
    # ----------------------------------------------------

    for step in range(total_steps):

        current_time = step * SIM_DT

        # ----------------------------------------------
        # Step command
        # ----------------------------------------------

        if step < settle_steps:
            target_vel = 0.0
        else:
            target_vel = TARGET_VEL

        # joint velocity target 생성
        velocity_target = torch.tensor(
            [[target_vel, target_vel]],
            device=sim.device,
        )

        # 두 wheel에 목표 속도 입력
        robot.set_joint_velocity_target(
            velocity_target,
            joint_ids=[left_id, right_id],
        )

        # command를 PhysX에 전달
        robot.write_data_to_sim()

        # physics 1 step
        sim.step()

        # PhysX 결과 읽기
        robot.update(SIM_DT)

        # ----------------------------------------------
        # 실제 wheel velocity
        # ----------------------------------------------

        left_vel = robot.data.joint_vel[0, left_id].item()
        right_vel = robot.data.joint_vel[0, right_id].item()
        left_tau = robot.data.applied_torque[0, left_id].item()
        right_tau = robot.data.applied_torque[0, right_id].item()

        # ----------------------------------------------
        # log
        # ----------------------------------------------

        print(
            f"{current_time:.6f},"
            f"{target_vel:.6f},"
            f"{left_vel:.6f},"
            f"{right_vel:.6f},"
            f"{left_tau:.6f},"
            f"{right_tau:.6f}"
        )

if __name__ == "__main__":

    main()

    simulation_app.close()