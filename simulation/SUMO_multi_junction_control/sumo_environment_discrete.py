import numpy as np
from datetime import datetime
import sumolib
import traci


class SumoEnvironmentDiscrete:
    def __init__(self, rou_file='Grid3by3/Grid3by3.rou.xml', net_file='Grid3by3/Grid3by3.net.xml', cfg='Grid3by3/Grid3by3.sumocfg'):
        self.run = 0
        self.sumo = None
        self._net = net_file
        self._rou = rou_file
        self._cfg = cfg

        self.label = 'default'

        self.max_timesteps = 100
        self.sumo_step_length = 20
        self.env_timeout = 0

        self.num_tls = 0
        self.num_lanes_per_direction = 3
        self.num_directions_per_tl = 4
        self.num_lanes_per_tl = self.num_lanes_per_direction * self.num_directions_per_tl
        self.tl_IDs = []
        self.lane_IDs = []

        self.lane_IDS_by_tl_IDs = {}

        self.last_measure = np.zeros([self.num_tls])
        self.vehicles = dict()

        self._start_sumo(use_gui=False)
        self._close_sumo()


    def _start_sumo(self, use_gui):
        if use_gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')
        sumo_cmd = [sumo_binary,
                     '-n', self._net,
                     '-r', self._rou,
                     '-c', self._cfg]
        sumo_cmd.append('--random')
        # sumo_cmd.append('--no-warnings')
        if use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])

        # Enable and setup virtual display
        # sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
        # from pyvirtualdisplay.smartdisplay import SmartDisplay
        # print("Creating a virtual display.")
        # self.disp = SmartDisplay(size=self.virtual_display)
        # self.disp.start()
        # print("Virtual display started.")

        traci.start(sumo_cmd, label=self.label)
        self.sumo = traci.getConnection(self.label)
        if use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        self.num_tls = self.sumo.trafficlight.getIDCount()
        self.tl_IDs = self.sumo.trafficlight.getIDList()
        lane_IDs = []
        for tl in range(self.num_tls):
            lane_IDs.append(self.sumo.trafficlight.getControlledLanes(self.tl_IDs[tl]))
            self.lane_IDS_by_tl_IDs[self.tl_IDs[tl]] = self.sumo.trafficlight.getControlledLanes(self.tl_IDs[tl])
        lane_IDs = np.array(lane_IDs)
        self.lane_IDs = lane_IDs.flatten()
        self.last_measure = np.zeros([self.num_tls])

    def _close_sumo(self):
        # close current sumo simulation
        # traci.switch(self.label)
        if self.sumo is None:
            return
        traci.close()
        self.sumo = None


    def _get_obs(self):
        obs = []
        for tl in range(self.num_tls):
            obs_tl = []
            obs_tl.append(self.sumo.trafficlight.getPhase(self.tl_IDs[tl]))

            for lane in range(self.num_lanes_per_tl):
                obs_tl.append(self.sumo.lane.getLastStepHaltingNumber(self.lane_IDs[tl*self.num_lanes_per_tl + lane]))
            obs_tl = np.array(obs_tl)
            obs.append(obs_tl)
        obs = np.array(obs)

        # print('timestep: {}, obs: {}'.format(self.env_timeout, obs))
        return obs

    def _apply_action(self, actions):
        actions = np.array(actions)
        # print('timestep: {}, actions: {}'.format(self.env_timeout, actions))
        for tl in range(self.num_tls):
            self.sumo.trafficlight.setPhase(self.tl_IDs[tl], actions[tl])
        # print("action sent")

    def _get_reward(self):
        reward = self._queue_reward_discrete()
        # reward = self._average_waiting_time_reward_discrete()
        # reward = self._average_speed_reward_discrete()
        # print('timestep: {}, reward: {}'.format(self.env_timeout, reward))
        return reward

    def _queue_reward_overall(self):
        # return -sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lane_IDs)
        reward = 0
        for tl in range(self.num_tls):
            for lane in range(self.num_lanes_per_tl):
                reward += self.sumo.lane.getLastStepHaltingNumber(self.lane_IDs[tl*self.num_lanes_per_tl + lane])
        return -reward

    def _queue_reward_discrete(self):
        reward = []
        for tl in range(self.num_tls):
            queue_length_tl = 0
            for lane in range(self.num_lanes_per_tl):
                queue_length_tl += self.sumo.lane.getLastStepHaltingNumber(self.lane_IDs[tl*self.num_lanes_per_tl + lane])
            reward.append(queue_length_tl)
        reward = np.array(reward)
        return -reward

    def _get_highest_queue_size_action(self, tl_num=0):
        """
        Returns the action corresponding to the queue with the highest size
        given a traffic light by number 
        TODO: generalise this to return a list of highest queue size actions for multijunction problem
        """
        highest_queue_length = 0
        highest_queue_length_action = 0

        print(self.tl_IDs)
        print(self.lane_IDs)
        print(tl_num)

        # for each direction of lights
        for direction in range(self.num_directions_per_tl):
            queue_length = 0
            for lane in range(self.num_lanes_per_direction):
                # summate the queue length of all lanes 
                lane_to_check = self.lane_IDs[tl_num + direction + lane]
                print(lane_to_check)
                queue_length += traci.lane.getLastStepVehicleNumber(lane_to_check)
            # if that combined queue length is the longest, return THAT action number
            if queue_length > highest_queue_length:
                highest_queue_length_action = direction
                highest_queue_length = queue_length
            print("_____________________")
        print(f"Direction {highest_queue_length_action} has length {highest_queue_length}")
        return highest_queue_length_action


        for lane in range(self.num_lanes_per_tl):
            road_to_check = self.lane_IDs[tl_num*self.num_lanes_per_tl + lane]
            queue_len = traci.edge.getLastStepVehicleNumber(road_to_check)
            if (queue_len > highest_queue_length):
                highest_queue_length = queue_len
                highest_queue_length_action = self.edges_to_action[road_to_check]
        return highest_queue_length_action

    def _average_waiting_time_reward_discrete(self):
        reward = []
        for tl in range(self.num_tls):
            wait_time_tl = 0
            for lane in range(self.num_lanes_per_tl):
                 veh_list = self.sumo.lane.getLastStepVehicleIDs(self.lane_IDs[tl*self.num_lanes_per_tl + lane])
                 if len(veh_list) > 0:
                     wait_time_tl += (sum(self.sumo.vehicle.getWaitingTime(veh) for veh in veh_list) / len(veh_list))
            reward.append(-wait_time_tl/self.num_lanes_per_tl)
        reward = np.array(reward)
        return reward

    def _average_speed_reward_discrete(self):
        reward = []
        for tl in range(self.num_tls):
            avg_speed = 0.0
            veh_list = []
            for lane in range(self.num_lanes_per_tl):
                veh_list = self.sumo.lane.getLastStepVehicleIDs(self.lane_IDs[tl*self.num_lanes_per_tl + lane])
                if len(veh_list) > 0:
                    avg_speed += (sum(self.sumo.vehicle.getSpeed(veh) for veh in veh_list)/ len(veh_list))
            reward.append(avg_speed / self.num_lanes_per_tl)
        reward = np.array(reward)
        return reward

    def get_num_traffic_lights(self):
        return self.num_tls

    def get_obs_space(self):
        return self.num_lanes_per_direction * self.num_directions_per_tl + 1
        # return (self.num_lanes_per_direction * self.num_directions_per_tl + 1) * self.num_tls

    def get_action_space(self):
        return self.num_directions_per_tl

    def reset(self):

        self._close_sumo()
        self.env_timeout = 0

        # restart a new simulation
        self._start_sumo(True)
        # self._start_sumo(True)

        # # set the phases of all traffic lights to red
        # for tl in range(self.num_tls):
        #     self.sumo.trafficlight.setRedYellowGreenState(self.tl_IDs[tl], 'rrrrrrrrrrrr')

        obs = self._get_obs()
        return obs

    def step(self, action):

        # apply the action on the traffic lights and run sumo simulation for defined steps
        self._apply_action(action)
        for _ in range(self.sumo_step_length):
            self.sumo.simulationStep()

        # collect the new observation, rewards and information
        obs = self._get_obs()
        rewards = self._get_reward()
        done = False
        info_queue = self.sumo.vehicle.getIDCount()
        info = {'traffic_load':info_queue}

        # check if current episode is finished
        self.env_timeout += 1
        if self.env_timeout >= self.max_timesteps:
            done = True


        return obs, rewards, done, info
