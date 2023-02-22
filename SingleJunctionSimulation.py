# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 00:07:19 2021

@author: Lewis
"""
import traci
import numpy as np
import random
import timeit
import os
import torch

from agents import VPGAgent

class SingleJunctionSimulation:
    def __init__(self, Model, TrafficLightController, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._TrafficLightController = TrafficLightController
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            # current_state = self._TrafficLightController._get_state()
            # print(f"Yingyi style state: {self._TrafficLightController._yingyi_state()}")
            current_state = self._TrafficLightController._yingyi_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait
            
            # saving the results of the PREVIOUS action to the memory
            if self._step != 0:
                self._Model.after_action(old_state, old_action, reward, current_state)

            # choose the light phase to activate, based on the current state of the intersection
            # TODO: rewrite the agent architecture so it's not doing Model._model, this is awkward
            # action = self._Model._model.choose_action(current_state)
            action = self._Model.choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._TrafficLightController._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._TrafficLightController._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        self._Model.finish_episode(100)
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._TrafficLightController._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_length == waited_seconds


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = self._TrafficLightController._get_incoming_roads()
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

