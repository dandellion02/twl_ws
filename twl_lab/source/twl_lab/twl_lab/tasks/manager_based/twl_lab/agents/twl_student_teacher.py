import torch
import torch.nn as nn
from torch.distributions import Normal


class AdaptationNetwork(nn.Module):
    """학생 신경망 (Student Network / ESP32 배포용)"""
    def __init__(self, history_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(history_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, history_obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(history_obs)


class EnvironmentFactorEncoder(nn.Module):
    """교사 인코더 (Teacher Network / 시뮬레이션 학습 전용)"""
    def __init__(self, privileged_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(privileged_dim, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(privileged_obs)


class TWLStudentTeacher(nn.Module):
    """RSL-RL 호환 Asymmetric Actor-Critic + RMA (Student-Teacher) 신경망 통합 모듈"""
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [128, 128],
        critic_hidden_dims: list = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = 8,
        history_dim: int = 50 * 18,  # (예시) history_length * policy_obs_dim 차원 지정
        **kwargs
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # 활성화 함수 선택
        activations = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}
        act_func = activations.get(activation.lower(), nn.ELU())

        # 1. Adaptation Module (Student & Teacher) - __init__에서 정적 생성 (Optimizer 정상 등록)
        self.adaptation_net = AdaptationNetwork(history_dim, self.latent_dim)
        self.env_factor_encoder = EnvironmentFactorEncoder(num_critic_obs, self.latent_dim)

        # 2. Actor Policy Network (입력 = Policy 센서 관측치 + Latent z)
        actor_input_dim = num_actor_obs + latent_dim
        actor_layers = []
        curr_dim = actor_input_dim
        for h_dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(curr_dim, h_dim), act_func])
            curr_dim = h_dim
        actor_layers.append(nn.Linear(curr_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # 3. Critic Network (입력 = Critic 관측치 / Privileged Information 포함)
        critic_layers = []
        curr_dim = num_critic_obs
        for h_dim in critic_hidden_dims:
            critic_layers.extend([nn.Linear(curr_dim, h_dim), act_func])
            curr_dim = h_dim
        critic_layers.append(nn.Linear(curr_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # 4. Action Distribution (PPO 분산 - Log Std 형태로 가중치 안정화)
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None

        # Disable args validation for speed
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: dict):
        """관측치를 받아 분포(Distribution) 업데이트"""
        policy_obs = observations["policy"]
        history_obs = observations.get("history", None)
        critic_obs = observations.get("critic", policy_obs)

        # 💡 학습 중에는 Teacher z 사용, 추론/배포 시에는 Student z_hat 사용
        if self.training:
            z = self.env_factor_encoder(critic_obs)
        elif history_obs is not None:
            z = self.adaptation_net(history_obs)
        else:
            z = torch.zeros(policy_obs.shape[0], self.latent_dim, device=policy_obs.device)

        actor_input = torch.cat([policy_obs, z], dim=-1)
        mean = self.actor(actor_input)
        
        # log_std를 exp 처리하여 표준편차 적용
        self.distribution = Normal(mean, self.log_std.exp())

    def act(self, observations: dict, **kwargs) -> torch.Tensor:
        """PPO Rollout 중 행동(Action) 샘플링"""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: dict) -> torch.Tensor:
        """실제 배포 / Test 평가 시 결정론적(Deterministic) 행동 반환"""
        policy_obs = observations["policy"]
        history_obs = observations["history"]

        # 100% Student Network만 사용하여 z 추론
        z_hat = self.adaptation_net(history_obs)
        actor_input = torch.cat([policy_obs, z_hat], dim=-1)
        return self.actor(actor_input)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        """Critic 평가 (Value 산출)"""
        critic_obs = observations.get("critic", observations["policy"])
        return self.critic(critic_obs)

    def compute_adaptation_loss(self, observations: dict) -> torch.Tensor:
        """Adaptation Network 학습용 MSE Loss 계산"""
        if "history" not in observations:
            return torch.tensor(0.0, device=observations["policy"].device)

        history_obs = observations["history"]
        critic_obs = observations["critic"]

        # Teacher는 Gradient가 흐르지 않도록 detach() 적용
        z_teacher = self.env_factor_encoder(critic_obs).detach()
        z_student = self.adaptation_net(history_obs)

        return nn.MSELoss()(z_student, z_teacher)
    
    
# twl_student_teacher.py 최하단

import rsl_rl.modules

# rsl_rl 내부 모듈 딕셔너리 또는 모듈 객체에 TWLStudentTeacher 강제 등록
# 1) rsl_rl.modules 패키지 자체 속성으로 등록
setattr(rsl_rl.modules, "TWLStudentTeacher", TWLStudentTeacher)

# 2) 만약 rsl_rl 내부에 __all__ 이나 modules 딕셔너리가 있는 경우 대응
if hasattr(rsl_rl.modules, "__all__"):
    rsl_rl.modules.__all__.append("TWLStudentTeacher")