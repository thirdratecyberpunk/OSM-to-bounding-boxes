import configparser
from sumolib import checkBinary
import os
import sys
import datetime
import subprocess


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

    return sumo_cmd

def generate_new_route_and_flow(seed):
    """
    Generates new flow and route files by calling the SUMO tool
    """
    os.system(f"""
    python3 {(os.environ.get('SUMO_HOME'))}/tools/randomTrips.py -n junctions/2023-01-13-15-51-50/osm.net.xml.gz -o junctions/2023-01-13-15-51-50/newTrip.trips.xml -r junctions/2023-01-13-15-51-50/newRoutes.rou.xml --flows 10 --period 3 --binomial 1 --seed {seed}""")