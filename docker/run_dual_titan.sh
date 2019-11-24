#!/bin/bash

# example: ./docker/run_clutter.sh "python main.py --continuous-size=32 --synthetic-upsample-size=1000 --window-size=32 --downsample-scale=1 --uid=vs2_clutter_test --batch-size=128 --conv-normalization=groupnorm --dense-normalization=batchnorm --encoder-layer-type=conv --decoder-layer-type=conv --activation=selu --disable-gated --max-time-steps=3 --max-image-percentage=0.3 --visdom-port=8098 --visdom-url=http://neuralnetworkart.com --reparam-type=beta --task=clutter --data-dir=/datasets/cluttered_mnist"

# first grab the root directory
ROOT_DIR=$(git rev-parse --show-toplevel)
echo "using root ${ROOT_DIR}"

# use the following command
CMD=$1
echo "executing $CMD "

# execute it in docker
nvidia-docker run --ipc=host -v /storage/jramapuram/datasets:/datasets2 -v $HOME/datasets:/datasets -v ${ROOT_DIR}:/workspace -e NVIDIA_VISIBLE_DEVICES=0,1 -it jramapuram/pytorch:1.1.0-cuda10.0 $CMD
