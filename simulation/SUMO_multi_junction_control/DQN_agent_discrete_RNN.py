import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from collections import namedtuple, deque
# from itertools import count
import random
import os
import cv2
from datetime import datetime
from typing import Dict, List, Tuple
import collections

from sumo_environment_discrete import SumoEnvironmentDiscrete
from arguments import get_args


class RNN(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, n_obs, hidden_size, n_actions):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(n_obs, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x, hidden):
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        rnn_out, new_hidden = self.rnn(x, hidden)
        x = F.relu(self.fc1(rnn_out))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))

        # print(x.shape)
        # print(hidden.shape)
        # print(rnn_out.shape)
        # print(new_hidden.shape)
        predictions = self.head(x)
        # print(predictions.shape)

        return predictions, new_hidden


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                       max_epi_num=100, max_epi_len=100,
                       lookup_step=None):
        self.random_update = random_update # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.lookup_step = lookup_step

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self, batch_size):
        if self.random_update is False and batch_size > 1:
            sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update: # Random upodate
            sampled_episodes = random.choices(self.memory, k=batch_size)

            # check_flag = True # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []

    def put(self, obs, actions, rewards, next_obs):
        self.obs.append(obs)
        self.action.append(actions)
        self.reward.append(rewards)
        self.next_obs.append(next_obs)

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs)

    def __len__(self) -> int:
        return len(self.obs)


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
#
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = deque([],maxlen=capacity)
#         self.position = 0
#
#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


class DQN_agent_discrete_RNN:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        self.n_obs = env.get_obs_space()
        self.num_junctions = env.get_num_traffic_lights()
        print('Environment observation space: {}'.format(self.n_obs))
        print('Environment action space: {}'.format(self.n_actions))
        print('Number of junctions: {}'.format(self.num_junctions))

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'DQN_Grid3by3')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        if self.args.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hidden_size = self.n_obs
        self.hidden = torch.zeros([self.num_junctions,1,1,self.hidden_size]).to(self.device)

        self.policy_net = []
        self.target_net = []
        self.optimizer = []
        # self.buffer = []
        self.episode_memory = []
        for i in range(self.num_junctions):
            self.policy_net.append(RNN(self.n_obs, self.hidden_size, self.n_actions).to(self.device))
            self.target_net.append(RNN(self.n_obs, self.hidden_size, self.n_actions).to(self.device))

            self.target_net[i].load_state_dict(self.policy_net[i].state_dict())
            self.target_net[i].eval()

            # if self.args.cuda:
            #     self.policy_net[i].cuda()
            #     self.target_net[i].cuda()

            # self.optimizer.append(torch.optim.Adam(self.policy_net[i].parameters(), lr = self.args.lr))
            self.optimizer.append(torch.optim.RMSprop(self.policy_net[i].parameters(), lr = self.args.lr))

            # self.buffer.append(ReplayBuffer(self.args.buffer_size))
            self.episode_memory.append(EpisodeMemory(random_update=True, max_epi_num=100, max_epi_len=100, lookup_step=2))

            self.random_eps = self.args.random_eps

    def learn(self):
        # env.reset()
        episode_return_all = []
        episode_avg_traffic_load_all = []
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()

            # print('Env reset')
            rewards_sum = np.zeros([self.num_junctions])
            # rewards_sum = 0
            traffic_load_sum = 0

            # initialise hidden layers
            self.hidden = torch.zeros([self.num_junctions,1,1,self.hidden_size]).to(self.device)

            episode_record = []
            for i in range(self.num_junctions):
                episode_record.append(EpisodeBuffer())

            for t in range(self.args.episode_length):
                actions_all = []
                for junction_id in range(self.num_junctions):
                    obs_junction_tensor = self._preproc_inputs(obs[junction_id])
                    # obs_junction_tensor = self._preproc_inputs(obs)

                    action_junction = self._select_action(obs_junction_tensor, junction_id)
                    actions_all.append(action_junction)
                actions_all = np.array(actions_all)

                # env.render()
                obs_new, reward, done, info = self.env.step(actions_all)
                rewards_sum += reward
                traffic_load_sum += info['traffic_load']

                reward_overall = reward.sum()

                # Store episode data into the buffers
                for junction_id in range(self.num_junctions):
                    # save the timestep transitions into the replay buffer
                    episode_record[junction_id].put(obs[junction_id], actions_all[junction_id], reward[junction_id], obs_new[junction_id])

                last_action = obs[:,0]
                last_action = last_action.flatten()
                print('[{}] Episode {}, Timestep {}, Last action: {}, Action: {}, Rewards: {}, Traffic load: {}'.format(datetime.now(), episode, t, last_action, actions_all, reward, info['traffic_load']))

                obs = obs_new

                # if done:
                #     break

            for junction_id in range(self.num_junctions):
                self.episode_memory[junction_id].put(episode_record[junction_id])

            # self.random_eps = max(0.001, 0.995*self.random_eps) #Linear annealing

            # Train the networks
            #print('Optimizing starts')
            # if len(self.buffer[0]) >= self.args.batch_size:
            if len(self.episode_memory[0]) >= 1:
                # print('Updating policy network')
                for _ in range(self.args.episode_length):
                    for i in range(self.num_junctions):
                        self._optimize_model(self.args.batch_size, i)
                #print('Optimizing finishes')

                # Update the target network, copying all weights and biases in DQN
                if episode >= self.args.target_update_step and episode % self.args.target_update_step == 0:
                    print('Updating target networks')
                    for i in range(self.num_junctions):
                        self.target_net[i].load_state_dict(self.policy_net[i].state_dict())

            print('[{}] Episode {} finished. Episode return: {}'.format(datetime.now(), episode, rewards_sum))
            # # print('[{}] Episode {} finished. Reward total J1: {}, J2: {}, J3: {}, J4: {}, traffic_load: {}' \
            # #       .format(datetime.now(), episode, reward_sum_J1, reward_sum_J2, reward_sum_J3, reward_sum_J4, traffic_load_sum))
            episode_return_all.append(rewards_sum)
            episode_avg_traffic_load_all.append(traffic_load_sum/self.args.episode_length)
            np.save(self.save_path + '/DiscreteRL_RNN_episode_return_queue_reward_flowprob0.5.npy', episode_return_all)
            np.save(self.save_path + '/DiscreteRL_RNN_avg_traffic_load_queue_reward_flowprob0.5.npy', episode_avg_traffic_load_all)

            for i in range(self.num_junctions):
                torch.save(self.policy_net[i].state_dict(), self.save_path + '/DiscreteRL_RNN_models/DiscreteRL_LSTM_policy_net_junction' + str(i) + '_flowprob0.5.pt')

        print('Learning process finished')

    # def obs_transfer_full(self, obs):
    #     # obs_combined = obs.flatten().copy()
    #     # # print(obs_combined)
    #     # obs_full = []
    #     # for junction_id in range(self.num_junctions):
    #     #     obs_full.append(obs_combined)
    #     # obs_full = np.array(obs_full)
    #     # # print(obs_full)
    #     # return obs_full
    #
    #     obs_full = np.zeros([self.num_junctions, self.num_junctions*self.n_obs], dtype=int)
    #     for junction_id in range(self.num_junctions):
    #         obs_full[junction_id, junction_id*self.n_obs:(junction_id+1)*self.n_obs] = obs[junction_id].copy()
    #     # print(obs_full)
    #     # print(obs_full.shape)
    #     return obs_full

    def _preproc_inputs(self, input):
        input_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
        # if self.args.cuda:
        #     input_tensor = input_tensor.cuda()
        return input_tensor

    def _select_action(self, state, junction_id):
        # global steps_done
        state = state.unsqueeze(0).unsqueeze(0)
        Q_value, self.hidden[junction_id]= self.policy_net[junction_id](state, self.hidden[junction_id])
        # Q_value, (self.hidden_h[junction_id], self.hidden_c[junction_id]) = self.policy_net[junction_id](state, (self.hidden_h[junction_id], self.hidden_c[junction_id]))

        sample = random.random()
        # eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
        #     math.exp(-1. * steps_done / self.args.eps_decay)
        # steps_done += 1
        if sample > self.random_eps:
            with torch.no_grad():
                # print(Q_values)
                Q_value = Q_value.squeeze(0).squeeze(0)
                # take the Q_value index with the largest expected return
                # action_tensor = Q_values.max(1)[1].view(1, 1)
                action_tensor = torch.argmax(Q_value)
                # print(action_tensor)
                action = action_tensor.detach().cpu().numpy().squeeze()
        else:
            action = random.randrange(self.n_actions)
        return action

    # def _select_action_random(self):
    #     return random.randrange(self.n_actions)
    #
    # def _select_action_inorder(self, timestep):
    #     action = (timestep % (self.n_actions))
    #     return action

    def _optimize_model(self, batch_size, junction_id):
        samples, seq_len = self.episode_memory[junction_id].sample(batch_size)

        observations = []
        actions = []
        rewards = []
        next_observations = []

        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)

        observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(self.device)
        actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(self.device)
        rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(self.device)
        next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(self.device)

        # if self.args.cuda:
        #     observations = observations.cuda()
        #     next_observations = next_observations.cuda()
        #     rewards = rewards.cuda()
        #     actions = actions.cuda()


        # initialise hidden layers
        train_hidden = torch.zeros([1,batch_size,self.hidden_size]).to(self.device)

        q_target,_ = self.target_net[junction_id](next_observations, train_hidden)
        next_state_values = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
        expected_values = rewards + self.args.gamma * next_state_values

        # initialise hidden layers
        train_hidden = torch.zeros([1,batch_size,self.hidden_size]).to(self.device)
        q_policy,_ = self.policy_net[junction_id](observations, train_hidden)
        predicted_values = q_policy.gather(2, actions)

        # Compute loss
        loss = F.smooth_l1_loss(predicted_values, expected_values)

        # calculate loss
        self.optimizer[junction_id].zero_grad()
        loss.backward()
        self.optimizer[junction_id].step()
        return

if __name__ == '__main__':
    args = get_args()
    env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3.rou.xml')
    # env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3_flow0.05.rou.xml')
    # env = SumoEnvironmentDiscrete('Grid3by3/Grid3by3_time_variant.rou.xml')

    # env = SumoEnvironmentDiscrete('Grid2by2/Grid2by2.rou.xml')
    # env = SumoEnvironmentDiscrete('Grid2by2/Grid2by2_time_variant.rou.xml')

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = DQN_agent_discrete_RNN(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()
