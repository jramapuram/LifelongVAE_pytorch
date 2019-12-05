#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull some s3 dataset, be sure to set the correct S3 allowance for your nodes.
aws s3 cp s3://jramapuram-datasets/celeba.tar.gz ~/datasets/celeba.tar.gz \
    && cd ~/datasets && tar xvf celeba.tar.gz && rm celeba.tar.gz

# Execute a simple VAE in the cloud for MNIST --use-prior-kl
#cd ~/kanerva_plus_plus && sh ./docker/run.sh "python main.py --activation=elu --batch-size=10 --continuous-size=256 --conv-normalization=groupnorm --data-dir=/datasets/celeba --decoder-layer-type=conv --dense-normalization=none --encoder-layer-type=conv --epochs=2000 --read-key-kl-beta=1.0 --write-key-kl-beta=1.0 --kl-beta=1.0 --latent-size=256 --lr=3e-4 --max-image-percentage=0.2 --max-time-steps=1 --mem-kl-beta=1.0 --memory-init=normal --memory-size=128 --nll-type=gaussian --optimizer=adam --seed=1234 --task=celeba --visdom-port=8099 --image-size-override=32 --visdom-url=http://neuralnetworkart.com --window-size=32 --reparam-type=isotropic_gaussian --clamp-prior-variance --disable-gated --disable-3d-memory --uid=kppCA0_7"


# 0 mem-kl
cd ~/kanerva_plus_plus && sh ./docker/run.sh "python main.py --activation=elu --batch-size=10 --continuous-size=256 --conv-normalization=groupnorm --data-dir=/datasets/celeba --decoder-layer-type=conv --dense-normalization=none --encoder-layer-type=conv --epochs=2000 --read-key-kl-beta=1.0 --write-key-kl-beta=1.0 --kl-beta=1.0 --latent-size=256 --lr=3e-4 --max-image-percentage=0.1 --max-time-steps=1 --mem-kl-beta=0.0 --memory-init=normal --memory-size=128 --nll-type=gaussian --optimizer=adam --seed=1234 --task=celeba --visdom-port=8099 --image-size-override=32 --visdom-url=http://neuralnetworkart.com --window-size=32 --reparam-type=isotropic_gaussian --clip=5 --disable-gated --clamp-prior-variance --uid=kppCA2_8SP"
