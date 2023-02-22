# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:50:20 2021

@author: Lewis
Class representing an agent controlling a traffic light
"""
import traci
import numpy as np

class TrafficLightController:
    # def __init__(self, possible_phases, tlid, lanes, incoming_roads, outgoing_roads, num_states, edges_to_action):
    def __init__(self, possible_phases, tlid, lanes, incoming_roads, outgoing_roads, edges_to_action):
        # ID of the traffic light this class controls
        self.tlid = tlid
        # list of all possible traffic light states
        self.possible_phases = possible_phases
        # number of possible phases (i.e. traffic light states)
        self.num_phases = len(self.possible_phases)
        # dictionary of all lanes that can be observed by this class
        # key is id, value is group value
        self.lanes = lanes

        # list of all lanes that are controlled by this traffic light
        # self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tlid)

        # array of all incoming edges
        self.incoming_roads = incoming_roads
        # array of all outgoing edges
        self.outgoing_roads = outgoing_roads
        # number of possible states
        # self.num_states = num_states
        # mapping of an edge to its action
        self.edges_to_action = edges_to_action
        
    def _get_incoming_roads(self):
        return self.incoming_roads
    
    # TODO: make state activation genericised
    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        # print(f"{self.tlid} chose action {action_number}, phase {self.possible_phases[action_number]}")
        traci.trafficlight.setPhase(self.tlid, action_number)
    
    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        # print(f"Activated yellow phase on action {old_action}, phase {self.possible_phases[old_action]}")
        # TODO: check if the yellow phase code is ALWAYS the next value up
        # not 100% convinced this is how it should be handled
        action_number = old_action + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        # print(f"{self.tlid} chose action {action_number}, phase {self.possible_phases[action_number]}")
        traci.trafficlight.setPhase(self.tlid, action_number)
        
    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        return sum([traci.edge.getLastStepHaltingNumber(incoming) for incoming in self.incoming_roads])
    
    def _calculate_pressure(self, road_to_calculate):
        """
        Calculates pressure for a given edge
        Pressure of an intersection. The intersection pressure 𝑝𝑖 is
        defined as the difference between the number of all vehicles enter-
        ing the lane and the average queue situation on the exit lane or at
        the adjacent intersection. 
        Pressure = # of incoming cars - # of outgoing cars
        Returns
        -------
        None.

        """
        return traci.edge.getLastStepHaltingNumber(self.incoming_roads[road_to_calculate]) - traci.edge.getLastStepHaltingNumber(self.outgoing_roads[road_to_calculate])   
    
    def _get_highest_pressure_corresponding_action(self):
        """
        Returns the action corresponding to the highest pressure lane
        TODO: find a better way of doing this, this assumes parity between
        incoming and outgoing roads
        """
        highest_pressure = 0
        highest_pressure_num = 0
        for count, value in enumerate(self.incoming_roads):
            if (self._calculate_pressure(count) > highest_pressure):
                highest_pressure = self._calculate_pressure(count)
                highest_pressure_num = count
        return highest_pressure_num
    
    def _get_greedy_queue_length_action(self):
        """
        Returns the action corresponding to the greedy action for # of cars
        in lane

        Returns
        -------
        None.

        """
        highest_queue_length = 0
        highest_queue_length_action = 0
        for count, value in enumerate(self.incoming_roads):
            road_to_check = self.incoming_roads[count]
            queue_len = traci.edge.getLastStepVehicleNumber(road_to_check)
            if (queue_len > highest_queue_length):
                highest_queue_length = queue_len
                highest_queue_length_action = self.edges_to_action[road_to_check]
        return highest_queue_length_action
    
    def _get_greedy_total_wait_time_action(self):
        """
        Returns the action corresponding to the greedy action for sum of waiting
        times in edge

        Returns
        -------
        None.

        """
        highest_wait_time = 0
        highest_wait_time_action = 0
        highest_wait_time_road = ""
        for count, value in enumerate(self.incoming_roads):
            road_to_check = self.incoming_roads[count]
            wait_time = traci.edge.getWaitingTime(road_to_check)
            if (wait_time > highest_wait_time):
                highest_wait_time = wait_time
                highest_wait_time_action = self.edges_to_action[road_to_check]
        return highest_wait_time_action
    
    
    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        waiting_times = {}
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in self.incoming_roads:  # consider only the waiting times of cars in incoming roads
                waiting_times[car_id] = wait_time
            else:
                if car_id in waiting_times: # a car that was tracked has cleared the intersection
                    del waiting_times[car_id] 
        total_waiting_time = sum(waiting_times.values())
        return total_waiting_time
    
    """
    Returns a representation of the state as an array of 1D arrays containing:
    1) the current phase code for the traffic light
    2) the value for getLastStepHaltingNumber for each lane controlled by this traffic light
    """
    def _yingyi_state(self):
        # gets a list of all the lanes controlled by this traffic light
        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tlid)
        obs = []
        # appends the current phase code for this traffic light
        obs_tl = []
        obs_tl.append(traci.trafficlight.getPhase(self.tlid))

        # gets the number of stationary cars in each lane observable by this traffic light
        for lane in self.controlled_lanes:
            obs_tl.append(traci.lane.getLastStepHaltingNumber(lane))
        # converts the observation to a numpy array
        obs_tl = np.array(obs_tl)
        # appends this observation to the array of observations
        obs.append(obs_tl)
        obs = np.array(obs)

        print(f'obs: {obs}')
        return obs

    """
    Returns the size of the state space for initialising neural networks
    """
    def get_state_space(self):
        # yingyi's example: return self.num_lanes_per_direction * self.num_directions_per_tl + 1
        # in our case, we count the number of lanes controlled by this traffic light
        return len(traci.trafficlight.getControlledLanes(self.tlid)) + 1
        
    