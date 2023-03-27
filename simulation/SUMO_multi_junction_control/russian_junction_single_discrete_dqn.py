import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from itertools import count
import random
import os
import cv2
from datetime import datetime

from sumo_environment_discrete import SumoEnvironmentDiscrete
from arguments import get_args

from visualization import Visualization 
from utils import import_train_configuration, set_sumo, set_train_path, set_top_path

class DQN_Linear(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN_Linear, self).__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_agent_discrete:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        self.n_obs = env.get_obs_space()
        self.num_junctions = env.get_num_traffic_lights()
        self.episode_return_all = []
        self.episode_avg_traffic_load_all = []
        print('Environment observation space: {}'.format(self.n_obs))
        print('Environment action space: {}'.format(self.n_actions))
        print('Number of junctions: {}'.format(self.num_junctions))

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'RussianJunctionModels')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.policy_net = []
        self.target_net = []
        self.optimizer = []
        self.buffer = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        for i in range(self.num_junctions):
            if self.args.fullobs:
                self.policy_net.append(DQN_Linear(self.n_obs * self.num_junctions, self.n_actions))
                self.target_net.append(DQN_Linear(self.n_obs * self.num_junctions, self.n_actions))
            else:
                self.policy_net.append(DQN_Linear(self.n_obs, self.n_actions))
                self.target_net.append(DQN_Linear(self.n_obs, self.n_actions))

            self.target_net[i].load_state_dict(self.policy_net[i].state_dict())
            self.target_net[i].eval()

            if self.args.cuda:
                self.policy_net[i].cuda()
                self.target_net[i].cuda()

            self.optimizer.append(torch.optim.RMSprop(self.policy_net[i].parameters(), lr = self.args.lr))

            self.buffer.append(ReplayBuffer(self.args.buffer_size))

    def learn(self):
        # env.reset()
        self.episode_return_all = []
        self.episode_avg_traffic_load_all = []
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()
            if self.args.fullobs:
                obs = self.obs_transfer_full(obs)

            # print('Env reset')
            rewards_sum = np.zeros([self.num_junctions])
            # rewards_sum = 0
            traffic_load_sum = 0
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
                if self.args.fullobs:
                    obs_new = self.obs_transfer_full(obs_new)

                # Store episode data into the buffers
                for junction_id in range(self.num_junctions):
                    # save the timestep transitions into the replay buffer
                    self.buffer[junction_id].push(obs[junction_id], actions_all[junction_id], obs_new[junction_id], reward[junction_id])

                if self.args.fullobs:
                    index = np.arange(0, self.num_junctions, 1, dtype=int)
                    last_action = obs[index, index * self.n_obs]
                else:
                    last_action = obs[:,0]
                    last_action = last_action.flatten()
                print('[{}] Episode {}, Timestep {}, Last action: {}, Action: {}, Rewards: {}, Traffic load: {}'.format(datetime.now(), episode, t, last_action, actions_all, reward, info['traffic_load']))

                obs = obs_new

                # Train the networks
                #print('Optimizing starts')
                if len(self.buffer[0]) >= self.args.batch_size:
                    # print('Updating policy network')
                    for i in range(self.num_junctions):
                        self._optimize_model(self.args.batch_size, i)
                #print('Optimizing finishes')

                # Update the target network, copying all weights and biases in DQN
                if t >= self.args.target_update_step and t % self.args.target_update_step == 0:
                    print('Updating target networks')
                    for i in range(self.num_junctions):
                        self.target_net[i].load_state_dict(self.policy_net[i].state_dict())

                # if done:
                #     break

            print('[{}] Episode {} finished. Episode return: {}'.format(datetime.now(), episode, rewards_sum))
            # # print('[{}] Episode {} finished. Reward total J1: {}, J2: {}, J3: {}, J4: {}, traffic_load: {}' \
            # #       .format(datetime.now(), episode, reward_sum_J1, reward_sum_J2, reward_sum_J3, reward_sum_J4, traffic_load_sum))
            self.episode_return_all.append(rewards_sum)
            self.episode_avg_traffic_load_all.append(traffic_load_sum/self.args.episode_length)
            np.save(self.save_path + '/DiscreteRL_episode_return_queue_reward_flowprob0.5.npy', self.episode_return_all)
            np.save(self.save_path + '/DiscreteRL_avg_traffic_load_queue_reward_flowprob0.5.npy', self.episode_avg_traffic_load_all)

            for i in range(self.num_junctions):
                torch.save(self.policy_net[i].state_dict(), self.save_path + '/DiscreteRL_models/DiscreteRL_policy_net_junction' + str(i) + '_flowprob0.5.pt')

        print('Learning process finished')

    def obs_transfer_full(self, obs):
        obs_full = np.zeros([self.num_junctions, self.num_junctions*self.n_obs], dtype=int)
        for junction_id in range(self.num_junctions):
            obs_full[junction_id, junction_id*self.n_obs:(junction_id+1)*self.n_obs] = obs[junction_id].copy()
        return obs_full

    def _preproc_inputs(self, input):
        input_tensor = torch.tensor(input, dtype=torch.float32)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor

    def _select_action(self, state, junction_id):
        # global steps_done
        sample = random.random()
        # eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
        #     math.exp(-1. * steps_done / self.args.eps_decay)
        # steps_done += 1
        if sample > self.args.random_eps:
            with torch.no_grad():
                Q_values = self.policy_net[junction_id](state)
                # print(Q_values)
                # take the Q_value index with the largest expected return
                # action_tensor = Q_values.max(1)[1].view(1, 1)
                action_tensor = torch.argmax(Q_values)
                # print(action_tensor)
                action = action_tensor.detach().cpu().numpy().squeeze()
                return action
        else:
            return random.randrange(self.n_actions)

    def _optimize_model(self, batch_size, junction_id):
        states, actions, next_states, rewards = Transition(*zip(*self.buffer[junction_id].sample(batch_size)))

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        if self.args.cuda:
            states = states.cuda()
            next_states = next_states.cuda()
            rewards = rewards.cuda()
            actions = actions.cuda()

        predicted_values = torch.gather(self.policy_net[junction_id](states), 1, actions.long().unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net[junction_id](next_states).max(1)[0].detach()
        expected_values = rewards + self.args.gamma * next_state_values

        # Compute loss
        loss = F.smooth_l1_loss(predicted_values, expected_values)

        # calculate loss
        self.optimizer[junction_id].zero_grad()
        loss.backward()
        self.optimizer[junction_id].step()
        return

if __name__ == '__main__':
    args = get_args()
    env = SumoEnvironmentDiscrete(rou_file='RussianJunction/osm.rou.xml', 
                                  net_file='RussianJunction/osm.net.xml',
                                    cfg='RussianJunction/osm.sumocfg')
    
    path = set_top_path("yingyi_simulation_results", ['DQN'])

    visualization = Visualization(
        path, 
        dpi=96
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = DQN_agent_discrete(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()

    visualization.plot_single_agent(data=trainer.episode_return_all,
                    filename='reward',
                    xlabel='Episode',
                    ylabel='Episode return value',
                    agent="DQN")
    
    visualization.plot_single_agent(data=trainer.episode_avg_traffic_load_all,
                    filename='avg_traffic_load',
                    xlabel='Episode',
                    ylabel='Average traffic load for junction',
                    agent="DQN")
