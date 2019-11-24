#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull some s3 dataset, be sure to set the correct S3 allowance for your nodes.
# aws s3 sync s3://bucket/dataset ~/datasets/mydataset

#cd ~/LifelongVAE_pytorch && sh ./docker/run.sh "python main.py --early-stop --reparam-type=mixture --epochs=200 --data-dir=/datasets/mnist --task=mnist+permuted+permuted+permuted+permuted+permuted+permuted+permuted+permuted+permuted+permuted --standard-batch-training --visdom-url=http://neuralnetworkart.com --visdom-port=8098 --lr=0.0001 --filter-depth=128 --batch-size=128 --activation=selu --encoder-layer-type=conv --decoder-layer-type=conv --conv-normalization=batchnorm --dense-normalization=none --discrete-size=1 --continuous-size=10 --optimizer=adam --latent-size=512 --continuous-mut-info=0 --discrete-mut-info=0.3 --consistency-gamma=0.5 --kl-beta=1.0 --likelihood-gamma=1.0 --disable-gated --shuffle-minibatches --generative-scale-var=1.0 --mut-clamp-strategy=clamp --mut-clamp-value=100 --uid=llPerm139_0"

# MNIST (non binary)
cd ~/LifelongVAE_pytorch && sh ./docker/run.sh "python main.py --early-stop --reparam-type=mixture --epochs=200 --data-dir=/datasets/mnist --task=mnist --visdom-url=http://neuralnetworkart.com --visdom-port=8098 --lr=0.0001 --filter-depth=128 --batch-size=128 --activation=selu --encoder-layer-type=conv --decoder-layer-type=conv --conv-normalization=batchnorm --dense-normalization=none --discrete-size=1 --continuous-size=10 --optimizer=adam --latent-size=512 --continuous-mut-info=0 --discrete-mut-info=0.3 --consistency-gamma=0.5 --kl-beta=1.0 --likelihood-gamma=1.0 --disable-gated --shuffle-minibatches --generative-scale-var=1.0 --mut-clamp-strategy=clamp --mut-clamp-value=100 --uid=llAMv139v5_0"

aws s3 sync ~/LifelongVAE_pytorch s3://jramapuram-logs/$(curl http://169.254.169.254/latest/meta-data/instance-id)/LifelongVAE_pytorch
