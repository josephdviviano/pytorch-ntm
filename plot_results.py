#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glob import glob
import json
import os
import sys

fname = 'outputs/mlp-ntm.json'

history = json.loads(open(fname, "rt").read())
training = np.array([history['cost'], history['loss'], history['seq_lengths']])

plt.plot(training[0], linewidth=2, label='Cost')
plt.grid()
plt.yticks(np.arange(0, training[0][0]+5, 5))
plt.ylabel('Cost per sequence (bits)')
plt.xlabel('Sequence')
plt.title('Training Convergence', fontsize=16)

ax = plt.axes([.57, .55, .25, .25], facecolor=(0.97, 0.97, 0.97))
plt.title("BCELoss")
plt.plot(x, training[1], 'r-', label='BCE Loss')
plt.yticks(np.arange(0, training[1][0]+0.2, 0.2))
plt.grid()

plt.show()

import IPython; IPython.embed()

