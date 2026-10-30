# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

#import twl_lab.tasks.manager_based.twl_lab.agents.twl_student_teacher

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16   # gradient 업데이트 전 병렬환경 하나 당 수집하는 step 수
    max_iterations = 3000    # 전체 gradient 업데이트 횟수
    save_interval = 50       # gradient 업데이트 횟수마다 모델 저장
    save_best_only = True        # True: 가장 성능 좋은 모델만 저장, False: save_interval 마다 모델 저장
    #clip_actions = 1.0        # action clipping 범위 (1.0: -1 ~ 1)
    experiment_name = "twl_lab_ppo"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,             # Gaussian Policy의 초기 action 표준편차, Actor가 출력한 평균 action 주위에서 exploration을 위한 action sampling 범위를 결정
        actor_obs_normalization=True,  # 관측값 정규화 여부 (True: 정규화, False: 정규화하지 않음)
        critic_obs_normalization=True, 
        actor_hidden_dims=[128, 64],     # actor : 로봇의 state를 입력으로 받아서 action을 출력하는 신경망(2개의 은닉층, 각 층 128, 64개의 뉴런) 
        critic_hidden_dims=[256, 128],    # critic : 로봇의 state를 입력으로 받아서 value function을 출력하는 신경망(2개의 은닉층, 각 층 256, 128개의 뉴런) 
        activation="elu",               # 신경망 활성화 함수 (elu, relu, tanh 등)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,            # 가치 함수 손실 가중치
        use_clipped_value_loss=True,    # 가치 함수 손실에 클리핑 적용 여부(업데이트 폭 제한)
        clip_param=0.2,                 # 정책 업데이트 시 클리핑 범위 (0.2: 20% 이상 변화 제한)
        entropy_coef=0.005,             # 정책 탐험 장려를 위한 엔트로피 보상 가중치 (0.005: 엔트로피 보상 0.5% 반영)
        num_learning_epochs=5,          # gradient 업데이트 시 mini-batch 학습 반복 횟수
        num_mini_batches=4,            # mini-batch 학습 시 나누는 배치 수 (num_steps_per_env * num_envs / num_mini_batches)
        #num_mini_batches=8,             # <--num_envs=8192 기준
        learning_rate=1.0e-3,           # 학습률 (1.0e-3: 0.001) , 가중치 업데이트 시 얼마나 크게 변화시킬지 결정   
        schedule="adaptive",            # 학습률 스케줄링 방식 (adaptive: KL divergence 변화량에 따라 학습률 조정, linear: 선형 감소, constant: 고정)
        gamma=0.99,                     # 보상 할인율 (0.99: 1 step 뒤 보상은 99% 반영, 2 step 뒤 보상은 98% 반영, ...) 먼 미래일수록 현재 행동에 대한 영향력이 작아짐
        lam=0.95,                       # GAE(Generalized Advantage Estimation) Critic이 예측한 가치 함수와 실제 보상 간의 차이를 얼마나 반영할지 결정하는 파라미터 (0.95: 95% 반영, 5% 무시)
        desired_kl=0.01,                # KL divergence 목표값 (정책 업데이트 시 KL divergence가 desired_kl 이상이면 학습률 감소)
        max_grad_norm=1.0,              # gradient 폭주를 막기 위한 Gradient Clipping 최댓값
    )                                   # KL divergence : 두 확률분포 간의 차이를 측정하는 지표, 정책 업데이트 시 KL divergence가 desired_kl 이상이면 학습률 감소