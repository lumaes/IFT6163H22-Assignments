import re
from collections import OrderedDict
from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MLP_policy import MLPPolicyPG, MLPPolicyAC
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *

import random

class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']
        self.gamma = self.agent_params['discount']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy

        self.actor = MLPPolicyAC(
             self.agent_params['ac_dim'],
             self.agent_params['ob_dim'],
             self.agent_params['n_layers'],
             self.agent_params['size'],
             self.agent_params['discrete'],
             self.agent_params['learning_rate'],
        )

        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            # Copy this from previous homework

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_ens variable defined above useful

            lower_bound = i * num_data_per_ens
            upper_bound = lower_bound + num_data_per_ens 
            
            observations = ob_no[lower_bound:upper_bound] # DONE(Q1)
            actions = ac_na[lower_bound:upper_bound] # DONE(Q1)
            next_observations = next_ob_no[lower_bound: upper_bound] # DONE(Q1)
    
            # use datapoints to update one of the dyn_models
            model = self.dyn_models[i] # DONE(Q1)
            log = model.update(observations, actions, next_observations, self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)
            
        # DONE Pick a model at random
        random_model = random.choice(self.dyn_models)

        # DONE Use that model to generate one additional next_ob_no for every state in ob_no (using the policy distribution) 
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy

        # extract actions
        actions = self.actor.get_action(ob_no)

        # prediction of the next state
        preds_next_state = random_model.get_prediction(ob_no, actions, self.data_statistics)

        # prediction of the next reward
        preds_rewards, preds_terminals = self.env.get_reward(ob_no, actions)

        path = {"observation" : ob_no,
                      "image_obs" : [],
                      "reward" : preds_rewards,
                      "action" : actions,
                      "next_observation": preds_next_state,
                      "terminal": preds_terminals}

        self.add_to_replay_buffer([path])

        # DONE add this generated data to the real data
        final_rewards = np.append(re_n, preds_rewards, axis=0)    
        final_actions = np.append(ac_na, actions, axis=0)
        final_observations = np.append(ob_no, ob_no, axis=0)
        final_next_state = np.append(next_ob_no,preds_next_state, axis=0)
        final_terminals = np.append(terminal_n, preds_terminals, axis=0)

        #DONE Perform a policy gradient update 
        loss = OrderedDict()

        critic_loss = None
        actor_loss = None

        # Hint: Should the critic be trained with this generated data? Try with and without and include your findings in the report.
        for critic_update in range(self.agent_params["num_critic_updates_per_agent_update"]):
            #critic_loss = self.critic.update(final_observations, final_actions, final_next_state, final_rewards, final_terminals)
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        advantage = self.estimate_advantage(final_observations, final_next_state, final_rewards, final_terminals)
        for actor_update in range(self.agent_params["num_actor_updates_per_agent_update"]):
            actor_loss = self.actor.update(final_observations, final_actions, adv_n=advantage)

        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['FD_Loss'] = np.mean(losses)

        return loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # DONE Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        values = self.critic.forward_np(ob_no)
        values_prime = self.critic.forward_np(next_ob_no)

        q_values = re_n + self.gamma * values_prime * (1-terminal_n)

        adv_n = q_values - values

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)

    def save(self, path):
        return self.actor.save(path)