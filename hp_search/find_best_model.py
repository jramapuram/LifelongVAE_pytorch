import re
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

NUM_DISTRIBUTIONS = 10

# read all files
filenames = os.listdir('./experiments')
fid_files = [f for f in filenames if 'fid' in f]
elbo_files = [f for f in filenames if 'elbo' in f]

# grab the exp numbers
filenums = [re.match('mnist_hp_search(.*)__fid.csv', f, re.M|re.I).groups()[0] for f in filenames if 'fid' in f]
largest_experiment = np.max([int(f) for f in filenums]) + 1

# extract fids and elbos
fids = [('exp_{}'.format(i), pd.read_csv('experiments/mnist_hp_search{}__fid.csv'.format(i), header=None).values) for i in filenums]
elbos = [('exp_{}'.format(i), pd.read_csv('experiments/mnist_hp_search{}__test_elbo.csv'.format(i), header=None).values) for i in filenums]
consistencies = [('exp_{}'.format(i), pd.read_csv('experiments/mnist_hp_search{}__consistency.csv'.format(i), header=None).values) for i in filenums]

# build empty histograms
fid_hist = np.zeros(largest_experiment)
elbo_hist = np.zeros(largest_experiment)
consistency_hist = np.zeros(largest_experiment)

# remove the incomplete experiments
fids = [f for f in fids if f[1].shape == (NUM_DISTRIBUTIONS, 1)]
elbos = [f for f in elbos if f[1].shape == (NUM_DISTRIBUTIONS, 1)]
consistencies = [f for f in consistencies if f[1].shape == (NUM_DISTRIBUTIONS, 1)]
print("{} experiments completed successfully".format(len(fids)))

# find top 5 fids and elbos
for i in range(NUM_DISTRIBUTIONS):
    min_fids = sorted(fids, key=lambda t: t[1][i])[0:20]
    min_elbos = sorted(elbos, key=lambda t: t[1][i])[0:20]
    max_consistencies = sorted(consistencies, key=lambda t: t[1][i], reverse=True)[0:20]
    print("[Experiement {}] best 5 fids: ".format(i), [t[0] for t in min_fids[0:5]])
    print("[Experiement {}] best 5 elbos: ".format(i) , [t[0] for t in min_elbos[0:5]])
    print("[Experiement {}] best 5 consistencies: ".format(i), [t[0] for t in max_consistencies[0:5]])
    for mf in min_fids:
        exp_num = int(mf[0].split("exp_")[1])
        fid_hist[exp_num] += 1

    for mf in min_elbos:
        exp_num = int(mf[0].split("exp_")[1])
        elbo_hist[exp_num] += 1

    for mf in max_consistencies:
        exp_num = int(mf[0].split("exp_")[1])
        consistency_hist[exp_num] += 1

# plot histograms
def _plot_hist(hist, largest_experiment, name):
    plt.figure()
    print(hist.shape)
    plt.hist(hist)#, bins=largest_experiment, normed=1)
    plt.xlabel('experiment number')
    plt.ylabel(name)
    plt.savefig(name+'.png', bbox_inches='tight')

# _plot_hist(np.expand_dims(fid_hist, 0), largest_experiment, 'fid')
# _plot_hist(elbo_hist, largest_experiment, 'elbo')
# _plot_hist(consistency_hist, largest_experiment, 'consistency')
print("overall best FID : ", np.argmax(fid_hist))
print("overall best ELBO : ", np.argmax(elbo_hist))
print("overall best Conistency : ", np.argmax(consistency_hist))
