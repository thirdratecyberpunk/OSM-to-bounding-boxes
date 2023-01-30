# Generates screenshots and videos of junction states through simulations in SUMO

import traci
import argparse
import ffmpeg
from datetime import datetime
import os
import csv
import random

from utils import set_sumo, generate_new_route_and_flow

# load in arguments from CL
parser = argparse.ArgumentParser(description="Generates screenshots and videos of junction states through simulations in SUMO.")
parser.add_argument('--junction', type = str, default='junctions/2023-01-13-15-51-50/', help="Directory of the junction to simulate")
parser.add_argument('--file', type = str, default='junctions/2023-01-13-15-51-50/osm.sumocfg', help="Filename of the SUMO config to load in.")
parser.add_argument('--timestep', type = int, default = 100, help='Number of timesteps you want to run the simulation for.')
parser.add_argument('--image_timestep', type = int, default = 1, help="Timestep to generate image for.")
parser.add_argument('--generate_movie', default=False, action='store_true', help="Flag for if you want an mp4 of the simulation runthrough.")
parser.add_argument('--new_flow', default=True, action='store_true', help="Flag for if you want to generate a new route/flow pair.")
parser.add_argument('--seed', type=int, default=-1, help='Seed for generating flow/route files, defaults to -1 which chooses a random seed if not using a specific seed.')

args = parser.parse_args()
seed = args.seed
junction = args.junction
new_flow = args.new_flow
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

# create a new route/flow file if needed 
if new_flow:
    if (seed == -1):
        seed = random.randint(1,1000)
    print(f"Generating new route and flow file for this run with seed {seed}")
    generate_new_route_and_flow(seed)

sumo_cmd = set_sumo(gui, sumocfg_file_name, timestep)

# start a simulation in SUMO and take screenshots every x timesteps
traci.start(sumo_cmd)
for i in range(timestep):
    # sanitising number for filenames
    padded_i = str(i).zfill(len(str(timestep)))

    traci.simulationStep() # repeat 0...n
    # create a .csv file for a timestep containing the x/y/angle positions of all vehicles currently in the simulation
    with open(f'{screenshots_dir}/junction_timestep_{padded_i}.csv', 'w', newline='') as csvfile:
        fieldnames = ['vehicle', 'x', 'y', 'lon', 'lat',  'angle']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        # get a list of all the vehicles
        vehicles = traci.vehicle.getIDList()
        # for each vehicle in the simulation, write a .csv row containing the vehicles
        for vehicle in vehicles:
            x,y = traci.vehicle.getPosition(vehicle)
            lon, lat = traci.simulation.convertGeo(x, y)
            angle = traci.vehicle.getAngle(vehicle)
            csvwriter.writerow({'vehicle': vehicle, 'x' : x, 'y': y, 'angle': angle,'lon': lon, 'lat': lat})

    if (i % image_timestep == 0):
        traci.gui.screenshot(traci.gui.DEFAULT_VIEW , f"{screenshots_dir}/junction_timestep_{padded_i}.png")
traci.close()

# creating a video via ffmpeg of results
if generate_movie:
    print("Generating movie...")
    (
        ffmpeg
        .input(f'{screenshots_dir}/*.png', pattern_type='glob', framerate=25)
        .output(f'{screenshots_dir}/movie.mp4')
        .run()
    )