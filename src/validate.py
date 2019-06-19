#!/usr/bin/env python

"""
YWARARAWR
"""

from __future__ import print_function, division
from os.path import dirname, realpath, join
import random
import sys
import cPickle as pickle

import numpy as np
import pandas as pd
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
        for line in f.readlines():
            if not line.startswith("v "):
                continue
            v.append([float(x) for x in line[2:].split(" ")])
    return {"pts": np.asarray(v)}


class Pose(object):
    # Intended to show pose of model in camera frame.
    def __init__(self, R, t_cm):
        self.R = R
        self.t_cm = t_cm


class Comparison(object):
    def __init__(self, pose_est_list, pose_gt_list):
        self.pose_est_list = pose_est_list
        self.pose_gt_list = pose_gt_list
        self.num_est = len(pose_est_list)
        self.num_gt = len(pose_gt_list)
        self.est_to_gt_indices = None

    def greedy_match(self, pose_error_func):
        # Greedy match.
        self.est_to_gt_indices = np.array([-1] * self.num_est, dtype=int)
        matched_gt = np.zeros(self.num_gt, dtype=bool)
        for i_est, pose_est in enumerate(self.pose_est_list):
            if np.all(matched_gt):
                break
            dist_to_gt = np.array([
                pose_error_func(pose_est, pose_gt)
                for pose_gt in self.pose_gt_list])
            dist_to_gt[matched_gt] = np.inf
            i_gt = np.argmin(dist_to_gt)
            self.est_to_gt_indices[i_gt] = i_gt
            matched_gt[i_gt] = True

    dtype = np.dtype([
        ('num_est', float),
        ('num_gt', float),
        ('num_tp', float),
        ('num_fp', float),
        ('num_fn', float),
        ('precision', float),
        ('recall', float),
        ('f1_score', float),
    ])

    def compute_metrics(self, pose_error_func, error_threshold):
        assert self.est_to_gt_indices is not None
        est_accurate = np.zeros(self.num_est, dtype=bool)
        for i_est, pose_est in enumerate(self.pose_est_list):
            i_gt = self.est_to_gt_indices[i_est]
            if i_gt == -1:
                continue
            pose_gt = self.pose_gt_list[i_gt]
            pose_error = pose_error_func(pose_est, pose_gt)
            if pose_error < error_threshold:
                est_accurate[i_est] = True
        num_tp = np.sum(est_accurate)
        num_fp = self.num_est - num_tp
        num_fn = np.sum(self.est_to_gt_indices == -1)
        # TODO(eric): Er... How do I compute true negatives???
        # For now, just gonna do F1 score...
        # Convention:
        # - No detections, precision is 100%.
        # - No labels, recall is 0%.
        if self.num_est == 0:
            precision = 1.
        else:
            precision = num_tp / self.num_est
        if self.num_gt == 0:
            recall = 0.
        else:
            recall = num_tp / self.num_gt
        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        return np.rec.array((
            self.num_est,
            self.num_gt,
            num_tp,
            num_fp,
            num_fn,
            precision,
            recall,
            f1_score,
        ), dtype=self.dtype)


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
        model_cm = load_model_cm(get_mesh_file(model))

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

        data_size = 1000
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

        def pose_error(X_est, X_gt):
            return add_metric(X_est.R, X_est.t_cm, X_gt.R, X_gt.t_cm, model_cm)

        comp_list = []

        # All translations are in centimeters.
        indices = range(data_size)
        for index in tqdm(indices):
            print(index)
            target = dataset[index]

            img = target["img"]
            # Detect object
            results = ObjectDetector.detect_object_in_image(
                        models[model].net, 
                        pnp_solvers[model],
                        img,
                        config_detect
                        )

            # https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
            pose_gt_list = []
            # Dunno why (0, n) tensors weren't used, but meh.
            if is_zero(target["translations"]) and is_zero(target["rot_quaternions"]):
                pass
            else:
                t_cm_gt_list = target["translations"].numpy()
                R_gt_list = [
                    np.array(Matrix33.from_quaternion(q_xyzw))
                    for q_xyzw in target["rot_quaternions"].numpy()]
                for R, t_cm in zip(R_gt_list, t_cm_gt_list):
                    pose_gt_list.append(Pose(R, t_cm))

            est_list = [est for est in results if est["location"] is not None]
            pose_est_list = []
            for est in est_list:
                t_cm_est = np.asarray(est["location"])
                q_xyzw_est = np.asarray(est["quaternion"])
                R_est = np.array(Matrix33.from_quaternion(q_xyzw_est))
                pose_est_list.append(Pose(R=R_est, t_cm=t_cm_est))

            comp = Comparison(pose_est_list, pose_gt_list)
            comp.greedy_match(pose_error)
            comp_list.append(comp)
            # Just for kicks:
            # TODO(eric): Figure out why this is so bad?
            print(comp.compute_accuracy(pose_error, 30))

    save_file = join(g_path2package, "comp_list.pkl")
    with open(save_file, "w") as f:
        pickle.dump(comp_list, f)


if __name__ == "__main__":
    sys.stdout = sys.stderr

    # set the manual seed.
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
