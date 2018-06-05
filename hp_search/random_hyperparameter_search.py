#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
from subprocess import call


parser = argparse.ArgumentParser(description='LifeLong VAE MNIST HP Search')
parser.add_argument('--num-trials', type=int, default=50,
                    help="number of different models to run for the HP search (default: 50)")
parser.add_argument('--num-titans', type=int, default=6,
                    help="number of TitanXP's (default: 6)")
parser.add_argument('--num-pascals', type=int, default=15,
                    help="number of P100's (default: 15)")
args = parser.parse_args()


def get_rand_hyperparameters():
    return {
        'batch-size': 32,           # TODO: randomize to test these
        'reparam-type': 'mixture',  # TODO: randomize to test these
        'layer-type': 'conv',       # TODO: randomize to test these
        'epochs': 500,                               # FIXED, but uses ES
        'calculate-fid-with': 'inceptionv3',         # FIXED
        'task': 'fashion',                            # FIXED
        'visdom-url': 'http://neuralnetworkart.com', # FIXED
        'visdom-port': 8104,                         # FIXED
        'shuffle-minibatches': np.random.choice([1, 0]),
        'discrete-size': np.random.choice([1, 3, 5, 10]),
        'continuous-size': np.random.choice([6, 8, 10, 20, 30, 40]),
        'optimizer': np.random.choice(['adam', 'rmsprop', 'adamnorm']),
        'continuous-mut-info': np.random.choice([1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 10.0]),
        'discrete-mut-info': np.random.choice([1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 10.0]),
        'use-pixel-cnn-decoder': np.random.choice([1, 0]),
        'monte-carlo-infogain': np.random.choice([1, 0]),
        'consistency-gamma': np.random.choice([0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0]),
        'kl-reg': np.random.choice([1.0, 1.1, 1.2, 1.3, 2.0, 3.0]),
        'likelihood-gamma': np.random.choice([0.0, 0.0, 0.0, 1.0, 0.5, 0.2, 1.2, 1.5, 2.0]),
        'generative-scale-var': np.random.choice([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1]),
        'mut-clamp-strategy': np.random.choice(['norm']), #TODO: randomize and test against clamp
        'mut-clamp-value': np.random.choice([1, 2, 5, 10, 30, 50, 100])
    }

def format_job_str(job_map, run_str):
    return """#!/bin/bash -l

#SBATCH --job-name={}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition={}
#SBATCH --time={}
#SBATCH --gres=gpu:{}:1
#SBATCH --mem=20000
#SBATCH --constraint="COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
srun {}""".format(
        job_map['job-name'],
        job_map['partition'],
        job_map['time'],
        job_map['gpu'],
        run_str
)

def unroll_hp_and_value(hpmap):
    base_str = ""
    no_value_keys = ["shuffle-minibatches", "use-pixel-cnn-decoder", "monte-carlo-infogain"]
    for k, v in hpmap.items():
        if k in no_value_keys and v == 0:
            continue
        elif k in no_value_keys:
            base_str += " --{}".format(k)
            continue

        if k == "mut-clamp-value" and hpmap['mut-clamp-strategy'] != 'clamp':
            continue

        base_str += " --{}={}".format(k, v)

    return base_str

def format_task_str(hp):
    hpmap_str = unroll_hp_and_value(hp)
    return """/home/ramapur0/.venv3/bin/python ../main.py --model-dir=.nonfidmodels
    --early-stop {} --uid={}""".format(
        hpmap_str,
        "{}".format(hp['task']) + "_hp_search{}_"
    ).replace("\n", " ").replace("\r", "").replace("   ", " ").replace("  ", " ").strip()

def get_job_map(idx, gpu_type):
    return {
        'partition': 'shared-gpu',
        'time': '12:00:00',
        'gpu': gpu_type,
        'job-name': "hp_search{}".format(idx)
    }

def run(args):
    # grab some random HP's and filter dupes
    hps = [get_rand_hyperparameters() for _ in range(args.num_trials)]

    # create multiple task strings
    task_strs = [format_task_str(hp) for hp in hps]
    print("#tasks = ", len(task_strs),  " | #set(task_strs) = ", len(set(task_strs)))
    task_strs = set(task_strs) # remove dupes
    task_strs = [ts.format(i) for i, ts in enumerate(task_strs)]

    # create GPU array and tile to the number of jobs
    gpu_arr = []
    for i in range(args.num_titans + args.num_pascals):
        if i < args.num_titans:
            gpu_arr.append('titan')
        else:
            gpu_arr.append('pascal')

    num_tiles = int(np.ceil(float(args.num_trials) / len(gpu_arr)))
    gpu_arr = [gpu_arr for _ in range(num_tiles)]
    gpu_arr = [item for sublist in gpu_arr for item in sublist]
    gpu_arr = gpu_arr[0:len(task_strs)]

    # create the job maps
    job_maps = [get_job_map(i, gpu_type) for i, gpu_type in enumerate(gpu_arr)]

    # sanity check
    assert len(task_strs) == len(job_maps) == len(gpu_arr), "#tasks = {} | #jobs = {} | #gpu_arr = {}".format(
        len(task_strs), len(job_maps), len(gpu_arr)
    )

    # finally get all the required job strings
    job_strs = [format_job_str(jm, ts) for jm, ts in zip(job_maps, task_strs)]

    # spawn the jobs!
    for i, js in enumerate(set(job_strs)):
        print(js + "\n")
        job_name = "hp_search_{}.sh".format(i)
        with open(job_name, 'w') as f:
            f.write(js)

        call(["sbatch", "./{}".format(job_name)])


if __name__ == "__main__":
    run(args)
