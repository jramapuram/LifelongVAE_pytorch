#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull some s3 dataset, be sure to set the correct S3 allowance for your nodes.
# aws s3 sync s3://bucket/dataset ~/datasets/mydataset

# Execute the binarized MNIST experiment WITHOUT prior kl & a dense net: BEST BELOW
#cd ~/kanerva_plus_plus && sh ./docker/run.sh "python main.py --activation=elu --batch-size=10 --continuous-size=256 --conv-normalization=groupnorm --data-dir=/datasets/binarized_mnist --decoder-layer-type=conv --dense-normalization=weightnorm --encoder-layer-type=conv --epochs=2000 --read-key-kl-beta=1.0 --write-key-kl-beta=1.0 --kl-beta=1.0 --prior-kl-beta=0 --mem-kl-beta=0 --latent-size=256 --lr=3e-4 --max-image-percentage=1.0 --memory-init=normal --memory-read-steps=20 --memory-write-steps=5 --memory-size=128 --nll-type=bernoulli --optimizer=rmsprop --seed=1234 --task=binarized_mnist --visdom-port=8099 --visdom-url=http://neuralnetworkart.com --window-size=32 --reparam-type=isotropic_gaussian --clamp-prior-variance --disable-gated --disable-3d-memory --clip=5.0 --uid=kppMv02_1v5DRead2"


cd ~/kanerva_plus_plus && sh ./docker/run.sh "python main.py --activation=elu --batch-size=5 --episode-length=20 --continuous-size=128 --conv-normalization=groupnorm --data-dir=/datasets/omniglot --decoder-layer-type=conv --dense-normalization=weightnorm --encoder-layer-type=conv --epochs=2000 --read-key-kl-beta=1.0 --write-key-kl-beta=1.0 --kl-beta=1.0 --prior-kl-beta=1.0 --mem-kl-beta=0 --latent-size=256 --lr=3e-4 --max-image-percentage=0.3 --memory-init=normal --memory-read-steps=8 --memory-write-steps=3 --memory-size=128 --nll-type=bernoulli --optimizer=rmsprop --seed=1234 --task=binarized_omniglot --visdom-port=8099 --visdom-url=http://neuralnetworkart.com --window-size=32 --reparam-type=isotropic_gaussian --clamp-prior-variance --disable-gated --disable-3d-memory --clip=10.0 --image-size-override=28 --uid=kppOv00_5"
