# Generates screenshots and videos of junction states through simulations in SUMO

import traci
import argparse
import ffmpeg
from datetime import datetime
import os
import csv
import random

from skimage import transform, io
from skimage.io import imread, imsave

from transforming import FootageTransformation

import numpy as np
from utils import set_sumo, generate_new_route_and_flow

def pad_timestep(i):
    return str(i).zfill(len(str(timestep)))

# load in arguments from CL
parser = argparse.ArgumentParser(description="Generates screenshots and videos of junction states through simulations in SUMO.")
parser.add_argument('--junction', type = str, default='junctions/2023-01-13-15-51-50/', help="Directory of the junction to simulate")
parser.add_argument('--file', type = str, default='junctions/2023-01-13-15-51-50/osm.sumocfg', help="Filename of the SUMO config to load in.")
parser.add_argument('--timestep', type = int, default = 100, help='Number of timesteps you want to run the simulation for.')
parser.add_argument('--image_timestep', type = int, default = 1, help="Timestep to generate image for.")
parser.add_argument('--generate_movie', default=False, action='store_true', help="Flag for if you want an mp4 of the simulation runthrough.")
parser.add_argument('--new_flow', default=True, action='store_true', help="Flag for if you want to generate a new route/flow pair.")
parser.add_argument('--seed', type=int, default=-1, help='Seed for generating flow/route files, defaults to -1 which chooses a random seed if not using a specific seed.')
parser.add_argument('--image_w', type=int, default=1838, help='Width of the screenshot to generate in pixels.')
parser.add_argument('--image_h', type=int, default=828, help='Height of the screenshot to generate in pixels.')

args = parser.parse_args()
seed = args.seed
junction = args.junction
new_flow = args.new_flow
sumocfg_file_name = args.file
timestep = args.timestep
image_timestep = args.image_timestep
generate_movie = args.generate_movie
image_w = args.image_w
image_h = args.image_h

# calculating image ratios for normalisation
# TODO: parameterise this, not sure if you can get the size of a viewport in pixels 
# through TRACI yet
# using default values, 70px = 10m
# image defaults to w=1838 px h = 828px
metre_in_pixels = (10/70)

# create folders for results of current run
results_dir = f"./results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    os.makedirs(f"{results_dir}/images")
    os.makedirs(f"{results_dir}/homographed_images")
    os.makedirs(f"{results_dir}/csvs")

# saving generation configuration
with open (f"{results_dir}/setup.txt", 'w') as f:
    f.write(f"Seed: {seed}")

# have to be running GUI mode to take a screenshot
gui = True

# create a new route/flow file if needed 
if new_flow:
    if (seed == -1):
        seed = random.randint(1,1000)
    print(f"Generating new route and flow file for this run with seed {seed}")
    generate_new_route_and_flow(seed)

sumo_cmd = set_sumo(gui, sumocfg_file_name, timestep)

ft = FootageTransformation()
ft.show_homography_plot()

# start a simulation in SUMO and take screenshots every x timesteps
traci.start(sumo_cmd)
for i in range(timestep):
    # sanitising number for filenames
    padded_i = pad_timestep(i)

    traci.simulationStep() # repeat 0...n
    # create a .csv file for a timestep containing the x/y/angle positions of all vehicles currently in the simulation
    with open(f'{results_dir}/csvs/junction_timestep_{padded_i}.csv', 'w', newline='') as csvfile:
        fieldnames = ['vehicle', 'vclass', 'x_metres', 'y_metres', 'width_metres', 'height_metres', 'bb_x_metres', 'bb_y_metres', 'bb_x_homographed', 'bb_y_homographed', 'bb_x_homographed_normalised', 'bb_y_homographed_normalised', 'width_normalised', 'height_normalised', 'bb_x_normalised', 'bb_y_normalised', 'lon', 'lat',  'angle', 'color']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        # get a list of all the vehicles
        vehicles = traci.vehicle.getIDList()
        # for each vehicle in the simulation, write a .csv row containing the vehicles
        for vehicle in vehicles:
            x_metres,y_metres = traci.vehicle.getPosition(vehicle)
            # x_metres/y_metres refers to position in the CENTRE of the front bumper
            # therefore position of centre for bounding box
            # x_metres + (width_metres / 2), y_metres + (height_metres / 2)
            width_metres = traci.vehicle.getWidth(vehicle)
            height_metres = traci.vehicle.getHeight(vehicle)
            bb_x_metres = x_metres + (width_metres / 2)
            bb_y_metres = y_metres + (height_metres / 2)
            to_transform = np.array([[((bb_x_metres, bb_y_metres))]])
            translation_result = np.squeeze(ft.translate_coordinates(to_transform=to_transform))
            bb_x_homographed, bb_y_homographed = translation_result[0], translation_result[1]

            # normalising the values
            # in this case, turning the values from m to ratio of pixels
            width_normalised = (width_metres * metre_in_pixels) / image_w
            height_normalised = (height_metres * metre_in_pixels) / image_h
            bb_x_normalised = (bb_x_metres * metre_in_pixels) / image_w
            bb_y_normalised = (bb_y_metres * metre_in_pixels) / image_h
            bb_x_homographed_normalised = (bb_x_homographed * metre_in_pixels) / image_w
            bb_y_homographed_normalised = (bb_y_homographed * metre_in_pixels) / image_h
            lon, lat = traci.simulation.convertGeo(x_metres, y_metres)
            angle = traci.vehicle.getAngle(vehicle)
            color = traci.vehicle.getColor(vehicle)
            vclass = traci.vehicle.getVehicleClass(vehicle)
            # TODO: convert vclass values into expected format for Daniel's .txt input
            csvwriter.writerow({
                'vehicle': vehicle,
                # 'vclass': vclass,
                #  this is hardcoded as 0 for Daniel's .csv reader for now
                'vclass': 0,
                'x_metres' : x_metres, 
                'y_metres': y_metres,
                'width_metres': width_metres,
                'height_metres': height_metres,
                'bb_x_metres' : bb_x_metres, 
                'bb_y_metres': bb_y_metres,
                'bb_x_homographed': bb_x_homographed,
                'bb_y_homographed': bb_y_homographed,
                'bb_x_homographed_normalised': bb_x_homographed_normalised,
                'bb_y_homographed_normalised': bb_y_homographed_normalised,
                'width_normalised': width_normalised,
                'height_normalised' : height_normalised, 
                'bb_x_normalised' : bb_x_normalised,
                'bb_y_normalised': bb_y_normalised, 
                'angle': angle,
                'lon': lon, 
                'lat': lat, 
                'color': color
            })
    if (i % image_timestep == 0):
        traci.gui.screenshot(traci.gui.DEFAULT_VIEW , f"{results_dir}/images/junction_timestep_{padded_i}.png", width=image_w, height=image_h)
    # screenshot saves the image at the NEXT call to simulationStep, so warp acts on the previous screenshot
    if (i > 0):
        # warping image results
        to_transform = imread(f"{results_dir}/images/junction_timestep_{pad_timestep(i - 1)}.png")
        warped = ft.translate_image(to_transform)
        io.imsave(f"{results_dir}/homographed_images/junction_timestep_{padded_i}.png", warped)
traci.close()

# creating a video via ffmpeg of results
if generate_movie:
    print("Generating movies...")
    (
        ffmpeg
        .input(f'{results_dir}/images/*.png', pattern_type='glob', framerate=25)
        .output(f'{results_dir}/movie.mp4')
        .run()
    )
    (
        ffmpeg
        .input(f'{results_dir}/homographed_images/*.png', pattern_type='glob', framerate=25)
        .output(f'{results_dir}/homographed_movie.mp4')
        .run()
    )