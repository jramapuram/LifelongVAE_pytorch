#!/bin/bash -l

#SBATCH --job-name=fashionmnist
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --constraint="COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
#SBATCH --mem-per-cpu=6000

#module load CUDA/8.0.61 GCC/4.9.3-2.25 OpenMPI/1.10.2 Python/2.7.11

# see here for more samples:
# /opt/cudasample/NVIDIA_CUDA-8.0_Samples/bin/x86_64/linux/release/

# if you need to know the allocated CUDA device, you can obtain it here:
echo $CUDA_VISIBLE_DEVICES


#CUDA_VISIBLE_DEVICES=3 srun python main.py --batch-size=300 --reparam-type="mixture" --discrete-size=10 --continuous-size=40 --epochs=200 --layer-type="conv" --ngpu=1 --optimizer=adam --mut-reg=0.0 --disable-regularizers --task rotated_mnist --uid=rotatedvanilla2 --calculate-fid --visdom-url="http://login1.cluster" --visdom-port=8097

# srun --partition=shared-gpu --gres=gpu:pascal:1 --pty bash


srun python main.py --batch-size=300 --reparam-type="mixture" --discrete-size=100 --continuous-size=40 --epochs=200 --layer-type="conv" --ngpu=1 --optimizer=adam --mut-reg=0.0 --disable-regularizers --task fashion --uid=fashionvanilla --calculate-fid --visdom-url="http://login1.cluster" --visdom-port=8098
