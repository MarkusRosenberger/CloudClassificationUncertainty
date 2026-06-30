#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:13:17 2026

@author: mrosenberger
"""

import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt

import classification_analysis as ana


# load ground truth, observations of volunteers and MV 
# both SYNOP scheme and collapsed scheme
# both all-sky and atlas data
with h5py.File('/srvfs/home/mrosenberger/Data/PublishedData/HumanCloudObs/Experiment3.nc', mode='r') as file:
    allsky = file['All-sky'][:]
    allsky_16c = file['All-sky_16c'][:]
    atlas = file['Atlas'][:]
    atlas_16c = file['Atlas_16c'][:]

# tick labels for heatmaps
xlabels = ['gt', 'Obs 1', 'Obs 2', 'Obs 3', 'Obs 4', 'Obs 5']
ylabels = ['Obs 1', 'Obs 2', 'Obs 3', 'Obs 4', 'Obs 5', 'MV']

# percentile for statistical significance
perc = 99

# some parameters for a nice plot
wspace = 0.1
loc = [0.33, .03, 0.3, 0.02]

# sub-titles and sub-labels of heatmaps
titles = ['All-sky 30 classes', 'All-sky 16 classes', 'Atlas 30 classes', 'Atlas 16 classes']
labels = ['a)', 'b)', 'c)', 'd)']

# initiate figure
fig, ax = plt.subplots(2, 2, figsize = (10, 10), gridspec_kw = {'wspace': wspace, 'hspace': 0.2}) # hspace 0.1 für Paper Version; gridspec_kw = {'wspace': 0.1, 'hspace': 0.1} ohne cbar
axs = ax.flatten()
cax = fig.add_axes(loc)

# u.l. allsky 30c
# u.r. allsky 16c
# l.l. atlas 30c
# l.r. atlas 16c
for i, data in enumerate([allsky, allsky_16c, atlas, atlas_16c]):
    # calculate values and get annotations
    weighted_avg, annots = ana.get_pairwise_weightedMCC_withbt(data, perc)

    # add colorbar for one of the plots
    if i == 1:
        sns.heatmap(weighted_avg, mask = np.triu(np.ones_like(weighted_avg, dtype=bool)), annot = annots, fmt = '', cmap = 'coolwarm',
                    vmin = -1, vmax = 1, xticklabels=xlabels, yticklabels=False, linewidths = .5, square = True, ax = ax.flatten()[i], 
                    cbar_ax = cax, cbar_kws = {'orientation': 'horizontal', 'label': 'Weighted MCC'})

    # no colorbar for the others
    else:
        sns.heatmap(weighted_avg, mask = np.triu(np.ones_like(weighted_avg, dtype=bool)), annot = annots, fmt = '', cmap = 'coolwarm',
                vmin = -1, vmax = 1, xticklabels=xlabels, yticklabels=False, linewidths = .5, square = True, ax = axs[i], cbar = False)

    # some more plot details
    axs[i].set_yticks(np.arange(1.5, 7.5), ylabels)
    axs[i].text(2.8, 2, titles[i], ha = 'left', va = 'center', fontweight= 'semibold')
    axs[i].text(-0.6, .6, labels[i], fontsize = 13, fontweight = 'semibold')
