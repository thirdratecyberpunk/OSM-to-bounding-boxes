# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:13:18 2021

@author: Lewis
"""
import numpy as np
import torch
from agents.memory import Memory

from agents.DeepQLearningAgent import DeepQLearningAgent

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, memory):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, output_dim)
        self._memory = memory


    def _build_model(self, num_layers, output_dim):
        """
        Build and compile a fully connected deep neural network
        """
        agent = DeepQLearningAgent(self._input_dim, output_dim)
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