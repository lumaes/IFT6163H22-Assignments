import numpy as np

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.td3_critic import TD3Critic
import copy

from ift6163.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)

        self.q_fun = TD3Critic(self.actor,
                               agent_params,
                               self.optimizer_spec)
        self.td3_policy_delay = agent_params['td3_policy_delay']

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and (self.t % self.learning_freq) == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            for critic_update in range(self.agent_params["num_critic_updates_per_agent_update"]):
                # TODO fill in the call to the update function using the appropriate tensors
                log['Critic'] = self.q_fun.update(ob_no, ac_na, next_ob_no, re_n, terminal_n, low=self.env.action_space.low, high=self.env.action_space.high)

            # TD3 update actor and target networks
            if (self.t % self.td3_policy_delay) == 0:
                log['Actor'] = self.actor.update(ob_no, copy.deepcopy(self.q_fun))
                self.q_fun.update_target_network()

        self.t += 1
        return log
