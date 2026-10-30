# twl_lab/mdp/actions.py

import torch

from isaaclab.envs.mdp.actions import JointVelocityAction, JointEffortAction
from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg, JointEffortActionCfg
from isaaclab.utils import configclass


class DelayedJointEffortAction(JointEffortAction):
    cfg: "DelayedJointEffortActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.delay_steps = cfg.delay_steps

        # 이번 policy step에서 새로 들어온 torque command
        self._pending_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        # 실제 actuator에 적용 중인 torque command
        self._applied_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        # 새 action을 받은 뒤 지난 physics step 수
        self._delay_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        # 이번 policy action이 이미 적용되었는지
        self._action_updated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor):
        """
        Policy step마다 호출.
        현재 설정에서는 8 ms마다 호출됨.
        """

        # Isaac Lab 기본 processing 수행
        #
        # 예:
        # raw action = 0.5
        # scale = 2.5
        #
        # processed action = 1.25 Nm
        super().process_actions(actions)

        # 새 torque target 저장
        self._pending_actions[:] = self.processed_actions

        # delay 다시 시작
        self._delay_counter.zero_()
        self._action_updated.zero_()

    def apply_actions(self):
        """
        Physics step마다 호출.
        sim.dt = 0.004 -> 4 ms마다 호출
        """

        update_mask = (
            (self._delay_counter >= self.delay_steps)
            & (~self._action_updated)
        )

        # delay 끝난 env만 새로운 torque 적용
        self._applied_actions[update_mask] = \
            self._pending_actions[update_mask]

        self._action_updated[update_mask] = True

        # 실제 joint effort target 적용
        self._asset.set_joint_effort_target(
            self._applied_actions,
            joint_ids=self._joint_ids,
        )

        self._delay_counter += 1

    def reset(self, env_ids=None):
        super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.device,
            )

        self._pending_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0
        self._delay_counter[env_ids] = 0
        self._action_updated[env_ids] = False


class DelayedJointVelocityAction(JointVelocityAction):
    cfg: "DelayedJointVelocityActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.delay_steps = cfg.delay_steps
        self.lpf_alpha = cfg.lpf_alpha

        self._filtered_raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        self._pending_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        self._applied_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        self._delay_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self._action_updated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )


    def process_actions(self, actions: torch.Tensor):

        # ==========================================
        # ESP32와 동일한 Action LPF
        # 125 Hz = 8 ms마다 딱 한 번 실행
        # ==========================================

        self._filtered_raw_actions[:] = (
            self.lpf_alpha * self._filtered_raw_actions
            + (1.0 - self.lpf_alpha) * actions
        )

        # ==========================================
        # LPF 이후 기존 Isaac Lab action processing
        # 여기서 scale=20 등이 적용됨
        # ==========================================

        super().process_actions(
            self._filtered_raw_actions
        )

        # delay에 넣을 새로운 target
        self._pending_actions[:] = self.processed_actions

        self._delay_counter.zero_()
        self._action_updated.zero_()


    def apply_actions(self):

        # ==========================================
        # physics dt = 4 ms delay
        # ==========================================

        update_mask = (
            (self._delay_counter >= self.delay_steps)
            & (~self._action_updated)
        )

        self._applied_actions[update_mask] = (
            self._pending_actions[update_mask]
        )

        self._action_updated[update_mask] = True

        self._asset.set_joint_velocity_target(
            self._applied_actions,
            joint_ids=self._joint_ids,
        )

        self._delay_counter += 1


    def reset(self, env_ids=None):

        super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.device,
            )

        self._filtered_raw_actions[env_ids] = 0.0
        self._pending_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0

        self._delay_counter[env_ids] = 0
        self._action_updated[env_ids] = False

@configclass
class DelayedJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = DelayedJointVelocityAction

    delay_steps: int = 1
    lpf_alpha: float = 0.9