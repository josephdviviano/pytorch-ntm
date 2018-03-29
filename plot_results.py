#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glob import glob
import json
import os
import sys

import torch
from IPython.display import Image as IPythonImage
from PIL import Image, ImageDraw, ImageFont
import io
from tasks.copytask import dataloader
from train import evaluate

from tasks.copytask import CopyTaskModelTraining, CopyTaskParams, CopyTaskBaselineModelTraining, CopyTaskBaselineParams

BATCH_NUM = 40000

def plot_history(name):

    fname = 'outputs/{}.json'.format(name)
    history = json.loads(open(fname, "rt").read())
    training = np.array([history['cost'], history['loss'], history['seq_lengths']])

    dv = 1000
    x = np.arange(dv / 1000, (BATCH_NUM / 1000) + (dv / 1000), dv / 1000)
    training = training.reshape(1, 3, -1, dv).mean(axis=3)[0][:][:]


    plt.plot(x, training[0], linewidth=2, label='Cost')
    plt.grid()
    plt.yticks(np.arange(0, training[0][0]+5, 5))
    plt.ylabel('Cost per sequence (bits)')
    plt.xlabel('Sequence (thousands)')
    plt.title('Training Convergence for {}'.format(name), fontsize=16)

    ax = plt.axes([.57, .55, .25, .25], facecolor=(0.97, 0.97, 0.97))
    plt.title("BCELoss")
    plt.plot(x, training[1], 'r-', label='BCE Loss')
    plt.yticks(np.arange(0, training[1][0]+0.2, 0.2))
    plt.grid()

    plt.savefig('outputs/{}_convergence.jpg'.format(name))
    plt.savefig('outputs/{}_convergence.svg'.format(name))
    plt.close()


def plot_seqlen(name):
    fname = 'outputs/{}.json'.format(name)
    history = json.loads(open(fname, "rt").read())

    loss = history['loss']
    cost = history['cost']
    seq_lengths = history['seq_lengths']

    unique_sls = set(seq_lengths)
    all_metric = list(zip(range(1, BATCH_NUM+1), seq_lengths, loss, cost))

    fig = plt.figure(figsize=(12, 5))
    plt.ylabel('Cost per sequence (bits)')
    plt.xlabel('Iteration (thousands)')
    plt.title('Training Convergence (Per Sequence Length) for {}'.format(name),
        fontsize=16)

    for sl in unique_sls:
        sl_metrics = [i for i in all_metric if i[1] == sl]

        x = [i[0] for i in sl_metrics]
        y = [i[3] for i in sl_metrics]

        num_pts = len(x) // 50
        total_pts = num_pts * 50

        x_mean = [i.mean()/1000 for i in np.split(np.array(x)[:total_pts], num_pts)]
        y_mean = [i.mean() for i in np.split(np.array(y)[:total_pts], num_pts)]

        plt.plot(x_mean, y_mean, label='Seq-{}'.format(sl))

    plt.yticks(np.arange(0, 80, 5))
    plt.legend(loc=0)

    plt.savefig('outputs/{}_convergence-perlen.jpg'.format(name))
    plt.savefig('outputs/{}_convergence-perlen.svg'.format(name))
    plt.close()

def evaluate(name):
    """ """
    N = 20
    if name == 'lstm':
        params = CopyTaskBaselineParams(sequence_max_len=N)
        model = CopyTaskBaselineModelTraining(params=params)
    elif name == 'mlp-ntm':
        params = CopyTaskParams(memory_m=N, sequence_max_len=N, controller_type='MLP')
        model = CopyTaskModelTraining()
    elif name == 'lstm-ntm':
        params = CopyTaskParams(memory_m=N, sequence_max_len=N, controller_type='lstm')
        model = CopyTaskModelTraining()

    model.net.load_state_dict(torch.load("./outputs/{}.model".format(name)))
    import IPython; IPython.embed()


plot_history('mlp-ntm')
#plot_history('lstm-ntm')
#plot_history('lstm')

plot_seqlen('mlp-ntm')
#plot_seqlen('lstm-ntm')
#plot_seqlen('lstm')

#evaluate('mlp-ntm')
#evaluate('lstm-ntm')
#evaluate('lstm')

