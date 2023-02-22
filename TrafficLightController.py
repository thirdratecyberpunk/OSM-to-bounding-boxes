# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:50:20 2021

@author: Lewis
Class representing an agent controlling a traffic light
"""
import traci
import numpy as np

class TrafficLightController:
    def __init__(self, possible_phases, tlid, lanes, incoming_roads, outgoing_roads, num_states, edges_to_action):
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
        self.num_states = num_states
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
    Returns an array of all edges for a given 
    """
    def getLanesForEdge(self, edge_id):
        # gets the number of lanes this edge is supposed to have
        num_lanes = traci.edge.getLaneNumber(edge_id)
        return [f"{edge_id}_{str(lane_count)}" for lane_count in range(num_lanes)]

    # TODO: check if this state representation works for other junctions
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    def _get_state(self):
        state = np.zeros(self.num_states)
        
        # get all vehicles in the simulation
        car_list = traci.vehicle.getIDList()
        
        # if that vehicle is in a lane that can be seen by the traffic light,
        # calculate its number in the state representation
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            
            lane_group = self.lanes.get(lane_id, -1)

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
        return state
        
    