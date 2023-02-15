# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:50:20 2021

@author: Lewis
Class representing an agent controlling a traffic light
Provides methods for 
"""
import traci
import numpy as np

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class TrafficLightController:
    # def __init__(self, phase_codes, tlid, lanes, incoming_roads, outgoing_roads, num_states):
    def __init__(self, tlid, lanes, incoming_roads, outgoing_roads, num_states):
        # ID of the traffic light this class controls
        self.tlid = tlid
        # dictionary of all phase codes
        # self.phase_codes = phase_codes
        # dictionary of all lanes that can be observed by this class
        # key is id, value is group value
        self.lanes = lanes
        # array of all incoming lanes
        self.incoming_roads = incoming_roads
        # array of all outgoing lanes
        self.outgoing_roads = outgoing_roads
        # number of possible states
        self.num_states = num_states
        
    def _get_incoming_roads(self):
        return self.incoming_roads
    
    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase(self.tlid, PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(self.tlid, PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase(self.tlid, PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase(self.tlid, PHASE_EWL_GREEN)
    
    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase(self.tlid, yellow_phase_code)
        
    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        return sum([traci.edge.getLastStepHaltingNumber(incoming) for incoming in self.incoming_roads])
    
    def _calculate_pressure(self, road_to_calculate):
        """
        Calculates pressure for a given lane
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
            queue_len = traci.edge.getLastStepVehicleNumber(self.incoming_roads[count])
            if (queue_len > highest_queue_length):
                highest_queue_length = queue_len
                highest_queue_length_action = count
        return highest_queue_length_action
    
    def _get_greedy_total_wait_time_action(self):
        """
        Returns the action corresponding to the greedy action for # of cars
        in lane

        Returns
        -------
        None.

        """
        highest_wait_time = 0
        highest_wait_time_action = 0
        for count, value in enumerate(self.incoming_roads):
            wait_time = traci.edge.getWaitingTime(self.incoming_roads[count])
            if (wait_time > highest_wait_time):
                highest_wait_time = wait_time
                highest_queue_length_action = count
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
    
    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
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
        
    