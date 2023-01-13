# generates screenshots of junction states through simulations 

import traci
import argparse

from utils import set_sumo

# load in arguments from CL
parser = argparse.ArgumentParser(description="Generates screenshots of junction states through simulations .")
parser.add_argument('--file', type = str, help="Filename of the SUMO config to load in.")
parser.add_argument('--timestep', type = int, default = 1)
parser.add_argument('--image_timestep', type = int, default = 1, help="Timestep to generate image for.")

args = parser.parse_args()
sumocfg_file_name = args.file
timestep = args.timestep
image_timestep = args.image_timestep
# have to be running GUI mode to take a screenshot
gui = True

sumo_cmd = set_sumo(gui, sumocfg_file_name, timestep)

# start a simulation in SUMO and take screenshots every x timesteps
traci.start(sumo_cmd)
for i in range(timestep):
    traci.simulationStep() # repeat 0...n
    if (i % image_timestep == 0):
        traci.gui.screenshot(traci.gui.DEFAULT_VIEW , f"./screenshots/junction_timestep_{i}.png")
traci.close()
