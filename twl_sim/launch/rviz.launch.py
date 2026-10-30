import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 패키지 경로 및 URDF 파일 경로 설정 (자신의 경로에 맞게 수정 필수!)
    package_name = 'twl_sim' # 자신의 패키지 이름으로 수정
    urdf_file_name = 'model_original.urdf'

    rviz_config_path = os.path.join(get_package_share_directory(package_name), 'rviz', 'robot.rviz')

    
    urdf_path = os.path.join(
        get_package_share_directory(package_name),
        'modeling',
        urdf_file_name
    )

    # 2. 로봇 Description 파일을 읽어서 문자열로 변환
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    # 3. 필요한 노드들 정의
    # robot_state_publisher: URDF를 /robot_description 토픽으로 퍼블리시
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': False
        }]
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': False}]
    )

    # joint_state_publisher_gui 노드 추가
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': False}]
    )

    # rviz2: RViz 실행 및 미리 저장된 설정 파일 로드 (선택사항)
    # rviz_config_file = os.path.join(get_package_share_directory(package_name), 'rviz', 'robot.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path], # 설정 파일이 있다면 주석 해제
        parameters=[{'use_sim_time': False}]
    )

    # 4. LaunchDescription에 노드 추가 및 반환
    return LaunchDescription([
        robot_state_publisher_node,
        rviz_node,
        #joint_state_publisher_node,
        joint_state_publisher_gui_node
    ])