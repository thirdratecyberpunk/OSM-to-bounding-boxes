from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from SingleJunctionSimulation import SingleJunctionSimulation
from agents.DQNModel import TrainModel as DQNModel
from agents.pytorch_vpg_model import TrainModel as VPGModel
from agents.PreTimedModel import PreTimedModel
from agents.GreedyQueueSizeModel import GreedyQueueSizeModel
from agents.HighestPressureModel import HighestPressureModel
from agents.GreedyWaitingTimeModel import GreedyWaitingTimeModel
from TrafficLightController import TrafficLightController
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path, set_top_path

if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])    
    
    TrafficLightController0 = TrafficLightController(
        tlid= config['tl'],
        lanes={
            "610375444#0_0":0,
            "610375444#0_1":0,
            "610375444#0_2":0,
            "610375447#1_0":1,
            "610375447#1_1":1,
            "610375447#1_2":1,
            "610375443#0_0":2,
            "610375443#0_1":2,
            "610375443#0_2":2,
            "360779398#1_0":3,
            "360779398#1_1":3,
            "610375443#1_0":4,
            "610375443#1_1":4,
            "610375443#1_2":4,
        },
        # outgoing_roads=["-610375444#0", "610375447#1", "610375443#0", "360779398#1"],
        # incoming_roads=["610375444#0", "-610375447#1", "-610375443#0", "-360779398#1"],
        outgoing_roads=["-610375444#0", "360779398#1","610375443#0", "-610375447#1"],
        incoming_roads=["610375444#0", "-360779398#1", "-610375443#0", "610375447#1"],
        num_states=config['num_states'],
        possible_phases=["rrrGGrrrrrGGGggrrrrrGGGgg",
        "rrryyrrrrryyyggrrrrryyygg",
        "rrrGGrrrrrrrrGGrrrrrrrrGG",
        "rrryyrrrrrrrryyrrrrrrrryy",
        "GGGGGGGGggrrrrrGGGggrrrrr",
        "yyyyyyyyggrrrrryyyggrrrrr",
        "rrGGGrrrGGrrrrrrrrGGrrrrr",
        "rryyyrrryyrrrrrrrryyrrrrr"
        ],
        edges_to_action={
            # L/R
            "610375444#0": 4,
            "-610375443#0": 4,
            # U/D
            "610375447#1": 0,
            "-360779398#1": 0
        }
    )

    # non-RL benchmarks
    Model = PreTimedModel(duration=1, actions=8)
    # Model = HighestPressureModel(tlc=TrafficLightController0)
    # Model = GreedyQueueSizeModel(tlc=TrafficLightController0)
    # Model = GreedyWaitingTimeModel(tlc=TrafficLightController0)
    # Model = HighestPressureModel(tlc=TrafficLightController0)

    # RL benchmarks
    # Model = VPGModel(
    #     input_dim=config['num_states'], 
    #     output_dim=config['num_actions'],
    #     gamma = config['gamma'],
    #     epsilon = config['epsilon'],
    #     alpha = config['alpha']
    # )

    # Model = DQNModel(
    #     config['num_layers'], 
    #     config['width_layers'], 
    #     config['batch_size'], 
    #     config['learning_rate'], 
    #     input_dim=config['num_states'], 
    #     output_dim=config['num_actions']
    # )

    if config['save']:
        path = set_top_path(config['models_path_name'], [Model])

        Visualization = Visualization(
            path, 
            dpi=96
        )
    
    print(config['training_epochs'])
    
    Simulation = SingleJunctionSimulation(
        Model,
        TrafficLightController0,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())

    if (config['save']):
        print("----- Session info saved at:", path)

        Model.save_model(path, episode)

        copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

        Visualization.plot_single_agent(data=Simulation.reward_store,
                            filename='reward',
                            xlabel='Episode',
                            ylabel='Cumulative local negative reward',
                            agent=Model.agent_name)
        Visualization.plot_single_agent(data=Simulation.cumulative_wait_store,
                            filename='delay',
                            xlabel='Episode',
                            ylabel='Cumulative delay(s)',
                            agent=Model.agent_name)
        Visualization.plot_single_agent(data=Simulation.avg_queue_length_store,
                                filename='queue',
                                xlabel='Episode',
                                ylabel='Average queue length (vehicles)', 
                                agent=Model.agent_name)