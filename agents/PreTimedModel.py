# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:13:18 2021

Simplistic pre-RL model for TLCS
Rotates between

@author: Lewis
"""
import numpy as np

class PreTimedModel:
    def __init__(self, actions, duration):
        self._actions = (actions - 1)
        self._chosen_action = 0
        self._duration = duration
        self._elapsed_duration = 0

    def _build_model(self, output_dim):
        pass
    
    def choose_action(self, current_state):
        return self.predict_one(current_state)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        In this case, simply rotates between actions after a certain
        amount of time
        """
        if self._elapsed_duration == self._duration:
            self._chosen_action = (self._chosen_action + 1) % self._actions
            self._elapsed_duration = 0
        self._elapsed_duration += 1
        return self._chosen_action


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        pass


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        pass


    def save_model(self, path, episode):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        pass
       
    def after_action(self, old_state, old_action, reward, current_state):
        pass
     
        
    def finish_episode(self, training_epochs):
        pass

    @property
    def agent_name(self):
        return "PreTimedModel"

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
    def __init__(self, actions, duration):
        self._actions = actions
        self._chosen_action = actions[0]
        self._duration = duration
        self._elapsed_duration = 0

    def predict_one(self, state):
        if self._elapsed_duration == self._duration:
            self._chosen_action = self._chosen_action + 1 % self._actions
            self._elapsed_duration = 0
        self._elapsed_duration += 1
        return self._chosen_action
    @property
    def input_dim(self):
        return self._input_dim