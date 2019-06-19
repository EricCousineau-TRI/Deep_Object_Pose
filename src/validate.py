#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
YWARARAWR
"""

from __future__ import print_function
from os.path import dirname, realpath, join
import sys

import numpy as np
import cv2
import yaml

from tqdm import tqdm
import pyassimp

import torchvision.transforms as transforms

from pyrr import Matrix33
from pysixd.pose_error import add as add_metric

# Import DOPE code
g_path2package = dirname(dirname(realpath(__file__)))
sys.path.append(join(g_path2package, "src/inference"))
sys.path.append(join(g_path2package, "src/training"))
sys.path.append("{}/src/inference".format(g_path2package))
weights_dir = "/home/eacousineau/Downloads/dope/weights"

from cuboid import *
from detector import *
from train import MultipleVertexJson


def get_mesh_file(model):
    return "/home/eacousineau/proj/tri/proj/perception/Dataset_Utilities/nvdu/data/ycb/aligned_cm/{}/google_16k/textured.obj".format(model)  # noqa


def load_model_cm(filename):
    assert filename.endswith(".obj")
    v = []
    with open(filename) as f:
        for line in f.readline():
            if not line.startswith("v "):
                continue
            v.append([float(x) for x in line[2:].split(" ")])
    return {"pts": np.asarray(v)}


def run_validation(params):

    models = {}
    pnp_solvers = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera = np.zeros((3,3))
    matrix_camera[0,0] = params["camera_settings"]['fx']
    matrix_camera[1,1] = params["camera_settings"]['fy']
    matrix_camera[0,2] = params["camera_settings"]['cx']
    matrix_camera[1,2] = params["camera_settings"]['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4,1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        model_6d = load_model_cm(get_mesh_file(model))

        models[model] =\
            ModelData(
                model, 
                join(weights_dir, params['weights'][model])
            )
        models[model].load_net_model()

        draw_colors[model] = \
            tuple(params["draw_colors"][model])
        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )

        data_size = 100
        dataset = MultipleVertexJson(
            root="/home/eacousineau/Downloads/dope/fat/mixed/kitchen_0",
            objectsofinterest=model,
            keep_orientation=True,
            noise=0,
            sigma=3,
            data_size=data_size,
            transform=transforms.Compose([]),
            test=True,
            )

        def is_zero(x):
            return x.shape[0] == 1 and (x == 0).all()

        # All translations are in centimeters.
        for index in range(data_size):# tqdm(range(data_size)):
            print(index)
            target = dataset[index]
            # Yawr
            if is_zero(target["translations"]) and is_zero(target["rot_quaternions"]):
                print("no ground truth...")
                continue

            img = target["img"]
            # Detect object
            results = ObjectDetector.detect_object_in_image(
                        models[model].net, 
                        pnp_solvers[model],
                        img,
                        config_detect
                        )

            # https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
            q_xyzw_gt_list, t_cm_gt_list = target["rot_quaternions"], target["translations"]
            R_gt_list = [Matrix33.from_quaternion(q_xyzw) for q_xyzw in q_xyzw_gt_list]

            # Get stuff
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                t_cm_est = result["location"]
                q_xyzw_est = result["quaternion"]
                R_est = Matrix33.from_quaternion(q_xyzw_est)
                print(i_r, t_cm_est, R_est)


if __name__ == "__main__":
    sys.stdout = sys.stderr
    config_name = "config_validate.yaml"
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    print("Loading DOPE parameters from '{}'...".format(yaml_path))
    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)
    print('    Parameters loaded.')

    run_validation(params)
    # import sys, trace
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.runfunc(run_validation, params)
