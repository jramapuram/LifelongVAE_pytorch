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
        'batch-size': 100,          # TODO: randomize to test these
        'reparam-type': 'mixture',  # TODO: randomize to test these
        'epochs': 500,
        'layer-type': 'conv',       # TODO: randomize to test these
        'task': 'mnist',
        'visdom-url': 'http://neuralnetworkart.com',
        'visdom-port': 8099,
        'discrete-size': np.random.choice([1, 3, 5, 10]),
        'continuous-size': np.random.choice([10, 20, 30, 40]),
        'optimizer': np.random.choice(['adam', 'rmsprop', 'adamnorm']),
        'mut-reg': np.random.choice([1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0]),
        'use-pixel-cnn-decoder': np.random.choice([1, 0]),
        'monte-carlo-infogain': np.random.choice([1, 0]),
        'continuous-mut-info': np.random.choice([1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.0, 1.0]),
        'consistency-gamma': np.random.choice([0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0]),
        'mut-clamp-strategy': np.random.choice(['norm', 'clamp', 'none'])
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

def format_task_str(hp):
    return """/home/ramapur0/.venv3/bin/python ../main.py --batch-size={} --model-dir=.nonfidmodels
    --reparam-type={} --discrete-size={} --continuous-size={} --epochs={} --layer-type={}
    --ngpu=1 --optimizer={} --mut-reg={} --task mnist --uid={}
    --calculate-fid-with=inceptionv3 --visdom-url={} {}
    {} --continuous-mut-info={} --consistency-gamma={}
    --mut-clamp-strategy={} --visdom-port={} --early-stop""".format(
        hp['batch-size'],
        hp['reparam-type'],
        hp['discrete-size'],
        hp['continuous-size'],
        hp['epochs'],
        hp['layer-type'],
        hp['optimizer'],
        hp['mut-reg'],
        "mnist_hp_search{}",
        hp['visdom-url'],
        "--use-pixel-cnn-decoder" if hp['use-pixel-cnn-decoder'] else "",
        "--monte-carlo-infogain" if hp['monte-carlo-infogain'] else "",
        hp['continuous-mut-info'],
        hp['consistency-gamma'],
        hp['mut-clamp-strategy'],
        hp['visdom-port'],
    ).replace("\n", " ").replace("\r", "").replace("   ", " ").replace("  ", " ").strip()

def get_job_map(idx, gpu_type):
    return {
        'partition': 'shared-gpu',
        'time': '12:00:00',
        'gpu': gpu_type,
        'job-name': "mnist_hp_search{}".format(idx)
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
