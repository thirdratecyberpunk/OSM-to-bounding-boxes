# Generates screenshots and videos of junction states through simulations in SUMO

import traci
import argparse
import ffmpeg
from datetime import datetime
import os

from utils import set_sumo

# load in arguments from CL
parser = argparse.ArgumentParser(description="Generates screenshots and videos of junction states through simulations in SUMO.")
parser.add_argument('--file', type = str, default='junctions/2023-01-13-15-51-50/osm.sumocfg', help="Filename of the SUMO config to load in.")
parser.add_argument('--timestep', type = int, default = 1000)
parser.add_argument('--image_timestep', type = int, default = 1, help="Timestep to generate image for.")
parser.add_argument('--generate_movie', default=False, action='store_true', help="Flag for if you want an mp4 of the simulation runthrough.")

args = parser.parse_args()
sumocfg_file_name = args.file
timestep = args.timestep
image_timestep = args.image_timestep
generate_movie = args.generate_movie

# create folder for images of current run
screenshots_dir = f"./screenshots/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

# have to be running GUI mode to take a screenshot
gui = True

sumo_cmd = set_sumo(gui, sumocfg_file_name, timestep)

# start a simulation in SUMO and take screenshots every x timesteps
traci.start(sumo_cmd)
for i in range(timestep):
    traci.simulationStep() # repeat 0...n
    if (i % image_timestep == 0):
        padded_i = str(i).zfill(len(str(timestep)))
        traci.gui.screenshot(traci.gui.DEFAULT_VIEW , f"{screenshots_dir}/junction_timestep_{padded_i}.png")
traci.close()

# creating a video via ffmpeg of results
if generate_movie:
    (
        ffmpeg
        .input(f'{screenshots_dir}/*.png', pattern_type='glob', framerate=25)
        .output(f'{screenshots_dir}/movie.mp4')
        .run()
    )