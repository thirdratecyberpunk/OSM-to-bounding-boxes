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

from sumo_environment_centralise import SumoEnvironmentCentralize
from sumo_environment_discrete import SumoEnvironmentDiscrete
from arguments import get_args


class DQN_Linear_Joint(nn.Module):
    def __init__(self, n_obs, n_actions, n_junctions):
        super(DQN_Linear_Joint, self).__init__()
        self.fc1 = nn.Linear(n_obs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.head = nn.Linear(512, n_actions ** n_junctions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

class DQN_Linear_Discrete(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN_Linear_Discrete, self).__init__()
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


class DQN_LfD_agent_joint:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        self.n_obs = env.get_obs_space()
        self.num_junctions = env.get_num_traffic_lights()
        print('Environment observation space: {}'.format(self.n_obs))
        print('Environment action space: {}'.format(self.n_actions))
        print('Number of junctions: {}'.format(self.num_junctions))

        self.policy_net = DQN_Linear_Joint(self.n_obs * self.num_junctions, self.n_actions, self.num_junctions)
        self.target_net = DQN_Linear_Joint(self.n_obs * self.num_junctions, self.n_actions, self.num_junctions)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr = self.args.lr)
        self.buffer = ReplayBuffer(self.args.buffer_size)

        # if args.load_path != None:
        #     load_policy_model = torch.load(self.args.load_path, map_location=lambda storage, loc: storage)
        #     self.policy_net.load_state_dict(load_policy_model)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.args.cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'DQN_Grid2by2')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # load pre-trained discrete networks
        self.demo_policy_net = []
        for i in range(self.num_junctions):
            self.demo_policy_net.append(DQN_Linear_Discrete(self.n_obs, self.n_actions))

            load_demo_policy_model = torch.load(self.save_path + '/DiscreteRL_policy_net_junction' + str(i) + '.pt', map_location=lambda storage, loc: storage)
            self.demo_policy_net[i].load_state_dict(load_demo_policy_model)

            if self.args.cuda:
                self.demo_policy_net[i].cuda()


    def learn(self):
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()
            reward_sum = 0
            for t in range(self.args.episode_length):
                actions_all = []
                for junction_id in range(self.num_junctions):
                    obs_junction_tensor = self._preproc_inputs(obs[junction_id])
                    action_junction = self._select_action_discrete(obs_junction_tensor, junction_id)
                    actions_all.append(action_junction)
                actions_all = np.array(actions_all)
                # env.render()
                obs_new, reward, done, info = self.env.step(actions_all)
                reward_sum += reward

                # Calculate the action and obs for joint learning
                action_joint = self._action_transfer(actions_all)
                obs_joint = self._obs_transfer(obs)
                obs_new_joint = self._obs_transfer(obs_new)
                reward_joint = reward.sum()
                # print(actions_all)
                # print(action_joint)
                # print(obs)
                # print(obs_joint)
                # Store episode data into the buffers
                self.buffer.push(obs_joint, action_joint, obs_new_joint, reward_joint)

                print('[{}] Episode {}, Timestep {}, Obs: {}, Action: {}, Rewards: {}, Traffic load: {}'.format(datetime.now(), episode, t, obs_joint[:self.num_junctions], actions_all, reward, info['traffic_load']))

                obs = obs_new

                # Train the networks
                #print('Optimizing starts')
                # if len(self.buffer) >= self.args.batch_size:
                if len(self.buffer) >= 100:
                    # print('Updating policy network')
                    loss = self._behaviour_clone(self.args.batch_size)
                    print('Loss is: {}'.format(loss))

                 # Update the target network, copying all weights and biases in DQN
                if t >= self.args.target_update_step and t % self.args.target_update_step == 0:
                    print('Updating target networks')
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # if done:
                #     break

            print('[{}] Episode {} finished. Episode return: {}'.format(datetime.now(), episode, reward_sum))

            torch.save(self.policy_net.state_dict(), self.save_path + '/Centralized_LfD_policy_net.pt')

        print('Learning process finished')

    def _preproc_inputs(self, input):
        input_tensor = torch.tensor(input, dtype=torch.float32)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor

    def _select_action_discrete(self, state, junction_id):

        with torch.no_grad():
            Q_values = self.demo_policy_net[junction_id](state)
            # print(Q_values)
            # take the Q_value index with the largest expected return
            # action_tensor = Q_values.max(1)[1].view(1, 1)
            action_tensor = torch.argmax(Q_values)
            # print(action_tensor)
            action = action_tensor.detach().cpu().numpy().squeeze()
            return action

    def _action_transfer(self, action_all):
        action_joint = 0
        for junction_id in range(self.num_junctions):
            action_joint += action_all[junction_id] * (self.num_junctions ** junction_id)

        return action_joint

    def _obs_transfer(self, obs_all):
        obs_joint = []
        # obs_phase = obs_all[:self.num_junctions]
        # obs_lane_queue_length = pbs_all[self.num_junctions:]
        for junction_id in range(self.num_junctions):
            obs_joint.append(obs_all[junction_id, 0])
        for junction_id in range(self.num_junctions):
            for i in range(12):
                obs_joint.append(obs_all[junction_id, 1+i])
        obs_joint = np.array(obs_joint)
        return obs_joint



    # def _optimize_model(self, batch_size):
    #     states, actions, next_states, rewards = Transition(*zip(*self.buffer.sample(batch_size)))
    #
    #     states = torch.tensor(states, dtype=torch.float32)
    #     next_states = torch.tensor(next_states, dtype=torch.float32)
    #     rewards = torch.tensor(rewards, dtype=torch.float32)
    #     actions = torch.tensor(actions, dtype=torch.float32)
    #
    #     if self.args.cuda:
    #         states = states.cuda()
    #         next_states = next_states.cuda()
    #         rewards = rewards.cuda()
    #         actions = actions.cuda()
    #
    #
    #     predicted_values = torch.gather(self.policy_net(states), 1, actions.long().unsqueeze(-1)).squeeze(-1)
    #     next_state_values = self.target_net(next_states).max(1)[0].detach()
    #     expected_values = rewards + self.args.gamma * next_state_values
    #
    #     # Compute loss
    #     loss = F.smooth_l1_loss(predicted_values, expected_values)
    #
    #     # calculate loss
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss


    def _behaviour_clone(self, batch_size):
        states, actions, next_states, rewards = Transition(*zip(*self.buffer.sample(batch_size)))

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        if self.args.cuda:
            states = states.cuda()
            actions = actions.cuda()

        # Q_values = self.policy_net(states)
        # Q_predicted = Q_values.max(1)[0].detach()
        # print(Q_predicted)
        Q_actions = torch.gather(self.policy_net(states), 1, actions.long().unsqueeze(-1)).squeeze(-1)
        print(Q_actions)
        # loss = - F.smooth_l1_loss(Q_actions, Q_predicted)
        loss = Q_actions.mean()
        print(loss)

        # calculate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


if __name__ == '__main__':
    args = get_args()
    env = SumoEnvironmentDiscrete()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = DQN_LfD_agent_joint(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()
