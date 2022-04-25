from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.MLP_policy import ConcatMLP


class TD3Critic(DDPGCritic):

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n, low=None, high=None):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        noise = (torch.randn(ac_na.shape) * self.td3_noise).clamp(-self.td3_noise_clip, self.td3_noise_clip)
        noise = noise.to(ptu.device)

        next_ac_na = self.actor_target(next_ob_no) + noise
        # Elementwise clipping
        low = ptu.from_numpy(low)
        high = ptu.from_numpy(high)
        next_ac_na = torch.max(torch.min(next_ac_na, high), low)


        # TODO compute the Q-values from the target network
        ## Hint: you will need to use the target policy
        qa_tp1_values = self.q_net_target(next_ob_no, next_ac_na)

        if self.td3_two_q_net:
            qa_2_values = self.q2_net_target(next_ob_no, next_ac_na)
            # select worst prediction between the two target q networks
            qa_tp1_values = torch.min(qa_tp1_values, qa_2_values)

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
        #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        qa_tp1_values = qa_tp1_values.squeeze()
        target = reward_n + self.gamma * qa_tp1_values * (1-terminal_n)
        target = target.detach()

        q_t_values = self.q_net(ob_no, ac_na).squeeze()

        assert q_t_values.shape == target.shape

        if self.td3_two_q_net:
            q2_t_values = self.q2_net(ob_no, ac_na).squeeze()
            loss = self.loss(q_t_values, target) + self.loss(q2_t_values, target)
        else:
            loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        if self.td3_two_q_net:
            utils.clip_grad_value_(self.q2_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        #self.learning_rate_scheduler.step()
        return {
            "Training Loss": ptu.to_numpy(loss),
            "Q Predictions": ptu.to_numpy(q_t_values),
            "Q Targets": ptu.to_numpy(target),
            "Policy Actions": ptu.to_numpy(ac_na),
            "Actor Actions": ptu.to_numpy(self.actor(ob_no))
        }

    def update_target_network(self):
        for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            ## Perform Polyak averaging
            target_param.data.copy_(self.polyak_avg * param + (1-self.polyak_avg) * target_param)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            ## Perform Polyak averaging for the target policy
            target_param.data.copy_(self.polyak_avg * param + (1-self.polyak_avg) * target_param)

        if self.td3_two_q_net:
            for target_param, param in zip(self.q2_net_target.parameters(), self.q2_net.parameters()):
                ## Perform Polyak averaging
                target_param.data.copy_(self.polyak_avg * param + (1-self.polyak_avg) * target_param)
