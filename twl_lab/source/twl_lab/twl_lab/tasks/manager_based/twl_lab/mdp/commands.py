# commands.py
import torch
from dataclasses import MISSING
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


@configclass
class DiscreteVelocityCommandCfg(CommandTermCfg):
    """Discrete Velocity Command 설정을 위한 Config"""

    class_type: type = None  # 아래 클래스 정의 후 지정

    asset_name: str = "robot"
    """대상이 되는 로봇 asset 이름"""

    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
    """자동 리샘플링을 막기 위한 긴 주기로 설정"""


class DiscreteVelocityCommand(CommandTerm):
    """5가지 사전 정의된 속도 커맨드를 커리큘럼에 따라 생성하는 Command Term"""

    cfg: DiscreteVelocityCommandCfg

    def __init__(self, cfg: DiscreteVelocityCommandCfg, env):
        super().__init__(cfg, env)

        # 5가지 사전 정의된 커맨드 [lin_x, lin_y, ang_z]
        self.predefined_commands = torch.tensor(
            [
                [0.0, 0.0, 0.0],   # 0: 정지 (STOP)
                [0.5, 0.0, 0.0],   # 1: 직진 (FORWARD)
                [-0.3, 0.0, 0.0],  # 2: 후진 (BACKWARD)
                [0.0, 0.0, 0.8],   # 3: 좌회전 (LEFT)
                [0.0, 0.0, -0.8],  # 4: 우회전 (RIGHT)
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # 관측치 반환용 버퍼 생성 [num_envs, 3]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """관측치(Observations) 및 Policy에 전달되는 커맨드 텐서"""
        return self.vel_command_b

    def _update_command(self):
        """매 step 실행되는 함수 (리샘플링 주기 도달 시 _resample 호출)"""
        pass

    def _resample_command(self, env_ids: torch.Tensor):
        """에피소드 리셋 시 지정된 환경(env_ids)에 새로운 Discrete Command 주입"""
        is_training = getattr(self._env.unwrapped, "training", True) if hasattr(self._env, "unwrapped") else True

        if is_training:
            num_steps_per_env = getattr(self._env.unwrapped.cfg, "num_steps_per_env", 16) if hasattr(self._env, "unwrapped") else 16
            current_iter = self._env.common_step_counter // num_steps_per_env

            if current_iter < 500:
                weights = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
            elif current_iter < 1500:
                weights = torch.tensor([0.50, 0.25, 0.25, 0.0, 0.0], device=self.device)
            else:
                weights = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20], device=self.device)
        else:
            weights = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20], device=self.device)

        # 가중치 기반 커맨드 인덱스 샘플링
        cmd_indices = torch.multinomial(weights, num_samples=len(env_ids), replacement=True)
        
        # 선택된 커맨드를 버퍼에 저장
        self.vel_command_b[env_ids] = self.predefined_commands[cmd_indices]

    def _update_metrics(self):
        """Tensorboard/Wandb 등 로깅용 메트릭 update (필요시 작성)"""
        pass


# Config와 Class 연결
DiscreteVelocityCommandCfg.class_type = DiscreteVelocityCommand