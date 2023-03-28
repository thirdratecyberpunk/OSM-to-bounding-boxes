import configparser
from sumolib import checkBinary
import os
import sys
import datetime
import subprocess

def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['epsilon'] = content['agent'].getfloat('epsilon')
    config['alpha'] = content['agent'].getfloat('alpha')
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['tl'] = content['junction']['tl']
    config['junction'] = content['junction']['junction']
    config['save'] = content['results'].getboolean('save')
    
    return config


def import_test_configuration(config_file):
    """
    Read the config file regarding the testing and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['models_path_name'] = content['dir']['models_path_name']
    config['model_to_test'] = content['dir'].getint('model_to_test') 
    return config

def set_top_path(models_path_name, agents):
    """
    Returns a model path for an experiment

    Parameters
    ----------
    models_path_name : TYPE
        DESCRIPTION.
    agents : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # generates the path for the experiment folder based on the date 
    # and number/type of agents
    
    full_name = 'experiment_{date:%Y-%m-%d_%H_%M_%S}_'.format(date=datetime.datetime.now())
    full_name += f"{len(agents)}_agents"
    for agent in agents:
        full_name += '_' + agent
    
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    models_path = os.path.join(models_path, full_name,'')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)
    
    # create a directory for each agent to store models and results in
    print(agents)
    # TODO: why isn't iter working here?
    count = 0
    for agent in agents:
        new_model_path = os.path.join(models_path, f"agent_{count}",'')
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        count = count + 1

    return models_path

def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 


def set_test_path(models_path_name, model_n):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):    
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", sumocfg_file_name, "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "--quit-on-end"]
    sumo_cmd.append('--no-warnings')

    return sumo_cmd

def generate_new_route_and_flow(seed):
    """
    Generates new flow and route files by calling the SUMO tool
    """
    os.system(f"""
    python3 {(os.environ.get('SUMO_HOME'))}/tools/randomTrips.py -n junctions/2023-01-13-15-51-50/osm.net.xml.gz -o junctions/2023-01-13-15-51-50/newTrip.trips.xml -r junctions/2023-01-13-15-51-50/newRoutes.rou.xml --flows 10 --period 3 --binomial 1 --seed {seed}""")