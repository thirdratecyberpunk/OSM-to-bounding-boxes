# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:13:18 2021

@author: Lewis
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_features=80, num_actions=4):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def vector_to_tensor(vector):
    return torch.from_numpy(vector).float()

# defining an agent who utilises DEEP Q-learning
# rather than utilise a Q-table to store all state-reward pairs
# uses a neural network to learn a distribution
# takes state as input, generates Q-value for all possible actions as output
class DeepQLearningAgent():
    def __init__(self, epsilon=0.05, alpha=0.1, gamma=1, batch_size=128, in_features=26, possible_actions=8):
        # checking CUDA support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # % chance that the agent will perform a random exploratory action
        self.epsilon = epsilon
        # learning rate -> how much the difference between new values is changed
        self.alpha = alpha
        # discount factor -> used to balance future/immediate reward
        self.gamma = gamma
        self.in_features = in_features
        # neural network outputs Q(state, action) for all possible actions
        self.possible_actions = possible_actions
        # policy network
        self.policy_qnn = DQN(in_features, num_actions=self.possible_actions).to(self.device)
        # neural network that acts as the Q function approximator
        self.target_qnn = DQN(in_features, num_actions=self.possible_actions).to(self.device)
        # loss function
        self.loss_function = nn.MSELoss()
        # loss value
        self.loss = 0
        # optimiser
        # self.optimiser = optim.SGD(self.policy_qnn.parameters(),lr=0.001,momentum=0.9)
        self.optimiser = optim.Adam(self.policy_qnn.parameters())
        # size of a batch of replays to sample from
        self.batch_size = batch_size

    def get_q_values_from_policy_network(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        return self.policy_qnn.forward(current_state_tensor)

    def get_q_values_from_target_network(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        return self.target_qnn.forward(current_state_tensor)

    def choose_action(self, current_state):
        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        """Returns the optimal action for the state from the Q value as predicted by the neural network"""
        if np.random.uniform(0,1) < self.epsilon:
            # chooses a random exploratory action if chance is under epsilon
            action = np.random.choice(range(self.possible_actions))
        else:
            # gets the values associated with action states from the neural network
            q_values = self.policy_qnn.forward(current_state_tensor)
            print(q_values)
            q_values_for_states = dict(zip(range(self.possible_actions), (x.item() for x in q_values)))
            # chooses the action with the best known 
            action = sorted(q_values_for_states.items(), key=lambda x: x[1])[0][0]
        return action

    def learn_batch(self, states, q_values):
        # calculate expected Q values for given states
        states_tensor = torch.tensor(states).float().to(self.device)
        q_values_tensor = torch.tensor(q_values).float().to(self.device)
        old_state_q_values = self.policy_qnn(states_tensor).to(self.device)
        q_values = torch.tensor(q_values).to(self.device)
        # calculate loss based on difference between expected and actual values
        self.optimiser.zero_grad()
        self.loss = self.loss_function(old_state_q_values, q_values_tensor)
        self.loss.backward()
        self.optimiser.step()

    def finish_episode(self):
        """
        Updates the weights of the target network

        Returns
        -------
        None.

        """
        print("Updating target network weights...")
        self.target_qnn.load_state_dict(self.policy_qnn.state_dict())
    
    def save_agent(self, agent_name, epoch):
        torch.save({
            'epoch': epoch,
            'policy_model_state_dict': self.policy_qnn.state_dict(),
            'target_model_state_dict': self.target_qnn.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'loss': self.loss,
            'epoch' : epoch
            }, agent_name)

    def load_agent(self, path):
        print("Before loading:")
        for var_name in self.optimiser.state_dict():
            print(var_name, "\t", self.optimiser.state_dict()[var_name])
        
        
        checkpoint = torch.load(path)
        self.policy_qnn.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_qnn.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.loss = checkpoint["loss"]
        
        print("After loading:")
        for var_name in self.optimiser.state_dict():
            print(var_name, "\t", self.optimiser.state_dict()[var_name])
        
        return checkpoint["epoch"]

class Memory:
    def __init__(self, size_max, size_min):
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min


    def add_sample(self, sample):
        """
        Add a sample into the memory
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element


    def get_samples(self, n):
        """
        Get n samples randomly from the memory
        """
        if self._size_now() < self._size_min:
            return []

        if n > self._size_now():
            return random.sample(self._samples, self._size_now())  # get all the samples
        else:
            return random.sample(self._samples, n)  # get "batch size" number of samples


    def _size_now(self):
        """
        Check how full the memory is
        """
        return len(self._samples)

class TrainModel:
    def __init__(self, batch_size, learning_rate, input_dim, output_dim, epsilon, alpha, gamma):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(epsilon, alpha, gamma, input_dim, output_dim)
        self._memory = Memory(600,50000)

    def _build_model(self, epsilon, alpha, gamma, input_dim, output_dim):
        """
        Build and compile a fully connected deep neural network
        """
        print(epsilon, alpha, gamma, input_dim, output_dim)
        agent = DeepQLearningAgent(epsilon=epsilon, alpha=alpha, gamma=gamma, in_features=input_dim, possible_actions=output_dim)
        return agent

    def choose_action(self, current_state):
        return self._model.choose_action(current_state)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.get_q_values_from_policy_network(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.get_q_values_from_policy_network(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.learn_batch(states, q_sa)


    def save_model(self, path, episode):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save_agent(f"{path}/DeepQLearningAgent_{episode}", episode)
       
    def after_action(self, old_state, old_action, reward, current_state):
        """

        Parameters
        ----------
        old_state : TYPE
            DESCRIPTION.
        old_action : TYPE
            DESCRIPTION.
        reward : TYPE
            DESCRIPTION.
        current_state : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._memory.add_sample((old_state, old_action, reward, current_state))
     
        
    def finish_episode(self, training_epochs):
        for _ in range(training_epochs):
            self._replay()
        self._model.finish_episode()
        
    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._memory.get_samples(self._model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._model.in_features))
            y = np.zeros((len(batch), self._model.possible_actions + 1))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._model.gamma * torch.max(q_s_a_d[i])  # update Q(state, action)
                x[i] = state.astype(np.float)
                y[i] = current_q.detach().cpu().numpy() # Q(state) that includes the updated action value

            self._model.learn_batch(x,y)
    

    @property
    def agent_name(self):
        return type(self._model).__name__

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, output_dim,  model_path):
        self._input_dim = input_dim
        self._model = DeepQLearningAgent(self._input_dim, output_dim)
        self._model.load_agent(model_path)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.get_q_values_from_policy_network(state)

    @property
    def input_dim(self):
        return self._input_dim