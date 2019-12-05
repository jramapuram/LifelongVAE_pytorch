#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np
from subprocess import call
from os.path import expanduser

parser = argparse.ArgumentParser(description='LifelongVAE HP Search')
parser.add_argument('--num-trials', type=int, default=50,
                    help="number of different models to run for the HP search (default: 50)")
parser.add_argument('--num-titans', type=int, default=15,
                    help="number of TitanXP's (default: 15)")
parser.add_argument('--num-pascals', type=int, default=15,
                    help="number of P100's (default: 15)")
parser.add_argument('--singularity-img', type=str, default=None,
                    help="if provided uses a singularity image (default: None)")
parser.add_argument('--early-stop', action='store_true', default=False,
                    help='enable early stopping (default: False)')
args = parser.parse_args()



def get_rand_hyperparameters():
    return {
        'seed': 1234,
        'reparam-type': 'mixture',  # TODO: randomize to test these
        'epochs': 1000,                      # FIXED, but uses ES
        'data-dir': '/home/ramapur0/datasets/binarized_mnist',
        # 'calculate-fid-with': 'none',        # FIXED
        'task': 'binarized_mnist',                     # FIXED
        'visdom-url': 'http://neuralnetworkart.com', # FIXED
        'visdom-port': 8098,                         # FIXED
        'lr': np.random.choice([1e-5, 1e-4, 2e-4, 3e-4, 1e-3]),
        'filter-depth': np.random.choice([8, 16, 32, 32, 32, 64, 128]),
        'batch-size': np.random.choice([32, 64, 128, 256]),
        'activation' :np.random.choice(['elu', 'selu', 'relu']),
        # 'encoder-layer-type': np.random.choice(['conv', 'coordconv', 'dense', 'resnet']),
        # 'decoder-layer-type': np.random.choice(['conv', 'coordconv', 'resnet', 'dense']),
        'encoder-layer-type': np.random.choice(['dense']),
        'decoder-layer-type': np.random.choice(['dense']),
        'shuffle-minibatches': np.random.choice([1, 0]),
        'conv-normalization': np.random.choice(['batchnorm', 'groupnorm', 'weightnorm', 'none']),
        'dense-normalization': np.random.choice(['batchnorm', 'weightnorm', 'none']),
        'disable-gated': np.random.choice([1, 0]),
        'discrete-size': np.random.choice([1]),
        'continuous-size': np.random.choice([8, 10, 20, 30, 40, 64, 96, 128]),
        'optimizer': np.random.choice(['adam', 'rmsprop']),
        # 'continuous-mut-info': np.random.choice(list(np.linspace(0, 10)) + [0,0,0,0,0]),
        # 'discrete-mut-info': np.random.choice(list(np.linspace(0, 10)) + [0,0,0,0,0]),
        'continuous-mut-info': np.random.choice([1e-4, 1e-3, 2e-3, 1e-2, 0.1, 0.3, 0.9]),
        'discrete-mut-info': np.random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        'monte-carlo-infogain': np.random.choice([1, 0]),
        # 'consistency-gamma': np.random.choice(np.linspace(0.1, 100, num=1000)),
        'consistency-gamma': np.random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3]),
        'kl-beta': np.random.choice([0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.5, 1.6]),
        # 'kl-beta': np.random.choice(list(np.linspace(0, 10)) + [1, 1, 1, 1]),
        #'likelihood-gamma': np.random.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.2, 1.2, 1.5, 2.0]),
        'likelihood-gamma': 0,
        'generative-scale-var': np.random.choice([0.3, 0.6, 1.0]),
        'mut-clamp-strategy': np.random.choice(['clamp']),
        # 'mut-clamp-strategy': np.random.choice(['norm', 'clamp', 'none']), #TODO: randomize and test against clamp
        #'mut-clamp-value': np.random.choice([1, 2, 5, 10, 30, 50, 100])
        'mut-clamp-value': np.random.choice([100])
    }

def format_job_str(job_map, run_str):
    singularity_str = "" if args.singularity_img is None \
        else "module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12 CUDA/10.0.130"
    return """#!/bin/bash -l

#SBATCH --job-name={}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition={}
#SBATCH --time={}
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --constraint="COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
echo $CUDA_VISIBLE_DEVICES
{}
srun --unbuffered {}""".format(
    job_map['job-name'],
    job_map['partition'],
    job_map['time'],
    singularity_str,
    # job_map['gpu'],
    run_str
)

def unroll_hp_and_value(hpmap):
    base_str = ""
    no_value_keys = ["disable-gated", "use-pixel-cnn-decoder", "monte-carlo-infogain",
                     "use-noisy-rnn-state", "use-prior-kl", "add-img-noise", "shuffle-minibatches"]
    for k, v in hpmap.items():
        if k in no_value_keys and v == 0:
            continue
        elif k in no_value_keys:
            base_str += " --{}".format(k)
            continue

        if k == "mut-clamp-value" and hpmap['mut-clamp-strategy'] != 'clamp':
            continue

        if k == "normalization" and hpmap['layer-type'] == 'dense':
            # only use BN or None for dense layers [GN doesnt make sense]
            base_str += " --normalization={}".format(np.random.choice(['batchnorm', 'none']))
            continue

        base_str += " --{}={}".format(k, v)

    return base_str

def format_task_str(hp):
    hpmap_str = unroll_hp_and_value(hp) # --model-dir=.nonfidmodels
    python_native = os.path.join(expanduser("~"), '.venv3/bin/python')
    # python_bin = "singularity exec -B /home/ramapur0/opt:/opt --nv {} python".format(
    python_bin = "singularity exec --nv {} python".format(
        args.singularity_img) if args.singularity_img is not None else python_native
    early_str = "--early-stop" if args.early_stop else ""
    return """{} ../main.py {} {} --uid={}""".format(
        python_bin,
        early_str,
        hpmap_str,
        "llHPv0{}_0"
    ).replace("\n", " ").replace("\r", "").replace("   ", " ").replace("  ", " ").strip()

def get_job_map(idx, gpu_type):
    return {
        #'partition': '\"shared-gpu,kalousis-gpu-EL7,cui-gpu,kruse-gpu\"',
        'partition': 'shared-gpu-EL7',
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
