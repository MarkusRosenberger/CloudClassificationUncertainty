#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:14:47 2025

@author: mrosenberger
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# for the Sankey Diagramm
import plotly.graph_objects as go


def onehot_encoder(CL, CM, CH):
    '''

    One-hot encoding of cloud observation arrays. 

    Parameters
    ----------
    CL : 1d-array, shape == (n_instances,)
        Cloud classes observed in the low level, always observed
    CM : 1d-array, shape == (n_instances,)
        Cloud classes observed in the low level, NaN if not observed
    CH : 1d-array, shape == (n_instances,)
        Cloud classes observed in the low level, NaN if not observed

    Returns
    -------
    2d-array, shape = (n_instances, 30)
        One-hot encoded version of observations in every layer. For each instance 30 values 
        representing 30 WMO SYNOP cloud classes. Value is 1 if this class was observed, 0 otherwise.
    '''
    # split observations into 3 altitude levels
    obs = np.zeros(shape = (len(CL), 3))
    obs[:,0] = CL # low
    obs[:,1] = CM # middle
    obs[:,2] = CH # high
    
    # low between 0 and 9
    # middle between 10 and 19
    # high between 20 and 29
    obs += np.array([0,10,20])

    ground_truth = np.zeros(shape = (len(CL), 30))

    for index, i in enumerate(obs):
        for c in i:
            if np.isfinite(c): 
                ground_truth[index, int(c)] += 1 # if no NaN, i.e. if observation in altitude level
            else: 
                pass # otherwise

    return ground_truth

# get data from csv file
df = pd.read_csv('Experiment2_simultaneous_WHWTUVIE.csv', index_col = 0)

'''
Fig. 5
'''

S1 = 'WHW'
S2 = 'TU'

# extract only data from WHW and TU
df_2stations = df[[S1 + '_CL', S1 + '_CM', S1 + '_CH', S2 + '_CL', S2 + '_CM', S2 + '_CH']]

# drop timesteps, where no low clouds were reported, i.e. CL = NaN
# this correpsonds to either no report at all or observation not possible
df_2stations = df_2stations[(df_2stations[S1 + '_CL'] < 10) & (df_2stations[S2 + '_CL'] < 10)] 

# one-hot encoding of single levels, i.e. 1 if class reported, 0 otherwise
# can also work with NaNs (does not consider them as class)
# result is array with length 30 for each timestep and station
# each entry is one SYNOP class

S1_onehot = onehot_encoder(df_2stations[S1 + '_CL'].values, df_2stations[S1 + '_CM'].values, df_2stations[S1 + '_CH'].values)
S2_onehot = onehot_encoder(df_2stations[S2 + '_CL'].values, df_2stations[S2 + '_CM'].values, df_2stations[S2 + '_CH'].values)

# how many classes have been reported at each timestep
S1_n_reported_classes = S1_onehot.sum(axis = 1)
S2_n_reported_classes = S2_onehot.sum(axis = 1)

# sum of both arrays can have on of three values for each category:
# 0 if no station observed a class,
# 1 if one of the stations did,
# 2 if both stations did
# looking only for observation at both stations
n_same_class_reported = ((S1_onehot + S2_onehot) == 2).sum(axis = 1)

'''
Check agreement and put it into 2 dictionaries:

    -) All possible combinations of numbers of reported classes at each station
        * key is xy: x is number of reported classes at station 1, y same at station 2
        * value is how often each combination occurs

    -) how many of the observed categories are identical at both stations
        * key is xyz: x & y as above, z is number of identical reported categories for combination xy
        * value is how often this occurs
'''

# compare number of reported classes
comparison_n_reported_classes = dict()
# compare reported classes themselves
comparison_n_identical_classes = dict()

# iterating over all possible combinations of number of reported classes
# i.e. [(1,1), (2,1), (1,2), (2,2), ..., (2,3), (3,3)]
for (n1, n2) in list(itertools.product([1,2,3], repeat = 2)):
    # how often a combination occurs and insert into dict
    comparison_n_reported_classes[(n1, n2)] = (S2_n_reported_classes[S1_n_reported_classes == n1] == n2).sum()

    # instances at which a combination occurs
    # and how many identical classes have been reported at each of them 
    temp_list = n_same_class_reported[(S1_n_reported_classes == n1) & (S2_n_reported_classes == n2)]
    
    # filter number of agreeing observations and put into dict
    for i in range(min(n1, n2) + 1):
        comparison_n_identical_classes[(n1, n2, i)] = (temp_list == i).sum()

# Plot

label_list = ['1', '2', '3', 
              '1', '2', '3',
              '1', '2', '3',
              '1', '2', '3', 
              '0 matching', '1 matching', '2 matching', '3 matching']

# each line needs a source, a target and a value
# all in correct respective order
source = [0, 0, 0, # 1 reported class at S1
          1, 1, 1, # 2 reported classes at S1
          2, 2, 2, # 3 reported classes at S1
                    
        # until here: agreement in number of observed classes
          
          3, 4, 5, # 1 reported class at S1, 1-3 reported classes at S2
          3, 4, 5, # 1 reported class at S1, 1-3 reported classes at S2

          6, 7, 8, # 2 reported classes at S1, 1-3 reported classes at S2
          6, 7, 8, # 2 reported classes at S1, 1-3 reported classes at S2
          7, 8, # 2 reported classes at S1, 1-3 reported classes at S2

          9, 10, 11, # 3 reported classes at S1, 1-3 reported classes at S2
          9, 10, 11, # 3 reported classes at S1, 1-3 reported classes at S2
          10, 11, # 3 reported classes at S1, 1-3 reported classes at S2
          11, # 3 reported classes at S1, 1-3 reported classes at S2
          ]

target = [3, 4, 5, # 1 reported class at S1, 1-3 reported classes at S2
          6, 7, 8, # 2 reported classes at S1, 1-3 reported classes at S2
          9, 10, 11, # 3 reported classes at S1, 1-3 reported classes at S2
          
          # until here: agreement in number of observed classes

          12, 12, 12, # 0 matching
          13, 13, 13, # 1 matching

          12, 12, 12, # 0 matching
          13, 13, 13, # 1 matching
          14, 14, # 2 matching

          12, 12, 12, # 0 matching
          13, 13, 13, # 1 matching
          14, 14, # 2 matching
          15, # 3 matching
          ]

values = [comparison_n_reported_classes[1,1], comparison_n_reported_classes[1,2], comparison_n_reported_classes[1,3], # 1 reported class at S1, 1-3 reported classes at S2
         comparison_n_reported_classes[2,1], comparison_n_reported_classes[2,2], comparison_n_reported_classes[2,3], # 2 reported classes at S1, 1-3 reported classes at S2
         comparison_n_reported_classes[3,1], comparison_n_reported_classes[3,2], comparison_n_reported_classes[3,3], # 3 reported classes at S1, 1-3 reported classes at S2

            # until here: agreement in number of observed classes

         comparison_n_identical_classes[1,1,0], comparison_n_identical_classes[1,2,0], comparison_n_identical_classes[1,3,0], # 1 reported class at S1, 1-3 reported classes at S2, 0 matching
         comparison_n_identical_classes[1,1,1], comparison_n_identical_classes[1,2,1], comparison_n_identical_classes[1,3,1], # 1 reported class at S1, 1-3 reported classes at S2, 1 matching

         comparison_n_identical_classes[2,1,0], comparison_n_identical_classes[2,2,0], comparison_n_identical_classes[2,3,0], # 2 reported classes at S1, 1-3 reported classes at S2, 0 matching
         comparison_n_identical_classes[2,1,1], comparison_n_identical_classes[2,2,1], comparison_n_identical_classes[2,3,1], # 2 reported classes at S1, 1-3 reported classes at S2, 1 matching
         comparison_n_identical_classes[2,2,2], comparison_n_identical_classes[2,3,2], # 2 reported classes at S1, 1-3 reported classes at S2, 2 matching

         comparison_n_identical_classes[3,1,0], comparison_n_identical_classes[3,2,0], comparison_n_identical_classes[3,3,0], # 3 reported classes at S1, 1-3 reported classes at S2, 0 matching
         comparison_n_identical_classes[3,1,1], comparison_n_identical_classes[3,2,1], comparison_n_identical_classes[3,3,1], # 3 reported classes at S1, 1-3 reported classes at S2, 1 matching
         comparison_n_identical_classes[3,2,2], comparison_n_identical_classes[3,3,2], # W 3, S 2-3 gemeldet, 2 matching
         comparison_n_identical_classes[3,3,3] # 3 reported classes at S1, 1-3 reported classes at S2, 3 matching
        ]

# Plotly colors 
color_1obs = '#636EFA'
color_2obs = '#FF97FF'
color_3obs = '#FFA15A'

color_0correct = 'rgba(0,0,0,.15)'
color_1correct = 'rgba(0,0,0,.40)'
color_2correct = 'rgba(0,0,0,.65)'
color_perfect = "#22B81D"

# locations of boxes are fine-tuned for the comparison of WHW and TU as shown in paper
fig = go.Figure(data=[go.Sankey(
    arrangement = 'snap',
    node = {'x': [0.001]*3 + [0.25]*9 + [0.99]*4,
            'y': [0.27, 0.62, .99, # S1 boxes
                  .24, .25, .26, .3, .31, .4, .45, .51, .72, # S2 boxes
                  .001, .08, .58, .99], # matching boxes
            'pad': 15,
            'line': dict(color = 'black', width = 0.8),
            'label':  label_list, 
            'color': [color_1obs] + [color_2obs] + [color_3obs] + [color_1obs]*3 + [color_2obs]*3 + [color_3obs]*3 + [color_0correct] + [color_1correct] + [color_2correct] + [color_perfect] # Graustufen für n correct, schwärzer für größeres n
            },
    link = {'source': source, 'target': target, 'value': values, 
            'color': [color_1obs]*3 + [color_2obs]*3 + [color_3obs]*3 + [color_0correct]*3 + [color_perfect] + [color_1correct]*2 + [color_0correct]*3 + [color_1correct]*3 + [color_perfect] + [color_2correct] + [color_0correct]*3 + [color_1correct]*3 + [color_2correct]*2 + [color_perfect]  # Graustufen für n correct, schwärzer für größeres n
            }
    )])

fig.update_layout(
    title=dict(text = S1 + '                       ' + S2, x = 0.052, y = .8, xanchor = 'left'),
    font=dict(size = 16, color = 'black'),
)

fig.write_image('SankeyDiagram.png', width=1000, height=550)
fig.show()

'''
Fig. 6
'''
# start with initial dataframe and only keep times where each station 
# reported at least a low cloud, i.e. an observation was in principle possible
# 87,558 time steps
df = df[(df['WHW_CL'] < 10) & (df['TU_CL'] < 10) & (df['VIE_CL'] < 10)] 

# one-hot encoding of single altitude levels as above
WHW_obs = onehot_encoder(df['WHW_CL'].values, df['WHW_CM'].values, df['WHW_CH'].values)
TU_obs = onehot_encoder(df['TU_CL'].values, df['TU_CM'].values, df['TU_CH'].values)
VIE_obs = onehot_encoder(df['VIE_CL'].values, df['VIE_CM'].values, df['VIE_CH'].values)

# number of observations of each class
WHW_obs_per_class = np.sum(WHW_obs, axis = 0)
TU_obs_per_class = np.sum(TU_obs, axis = 0)
VIE_obs_per_class = np.sum(VIE_obs, axis = 0)

# Plot

# ensure that different stations have different opacities
alpha_vec = [.6, .8, 1]
# correct position and with of the bars
xax = np.arange(10)
bar_width = .25
# fontsize for plot
fs = 11

# which classes correspond to left and right y-axis
# order of low, middle, high level is flipped compared to plot order
mask_left = [[0, 1, 2, 5, 6, 8], # low
             [0, 3, 4, 7], # middle
             [0, 1, 2]] # high

mask_right = [[3, 4, 7, 9], # low
            [1, 2, 5, 6, 8, 9], # middle
            [3, 4, 5, 6, 7, 8, 9]] # high

# Figure and second axes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 12), gridspec_kw = dict(hspace = 0.1), sharex = True)
ax1_twin = ax1.twinx()
ax2_twin = ax2.twinx()
ax3_twin = ax3.twinx()

# iterate over stations
for oi, o in enumerate([TU_obs_per_class, WHW_obs_per_class, VIE_obs_per_class]):
    # position on x-axis
    xax_plot = xax - bar_width + oi*bar_width

    # separate plots for left and right axis
    # separate altitude levels
    # high level
    ax1.bar(xax_plot[mask_left[2]], o[20:][mask_left[2]], color = 'tab:red', alpha = alpha_vec[oi], width = bar_width)
    collection = ax1_twin.bar(xax_plot[mask_right[2]], o[20:][mask_right[2]], color = 'tab:blue', alpha = alpha_vec[oi], width = bar_width)
    # middle level
    ax2.bar(xax_plot[mask_left[1]], o[10:20][mask_left[1]], color = 'tab:red', alpha = alpha_vec[oi], width = bar_width)
    collection = ax2_twin.bar(xax_plot[mask_right[1]], o[10:20][mask_right[1]], color = 'tab:blue', alpha = alpha_vec[oi], width = bar_width)
    # low level
    ax3.bar(xax_plot[mask_left[0]], o[:10][mask_left[0]], color = 'tab:red', alpha = alpha_vec[oi], width = bar_width)
    collection = ax3_twin.bar(xax_plot[mask_right[0]], o[:10][mask_right[0]], color = 'tab:blue', alpha = alpha_vec[oi], width = bar_width)

# some plot and axis details - same routine for each subplot
for a, a_twin, label, level, limit in zip([ax1, ax2, ax3], [ax1_twin, ax2_twin, ax3_twin], ['a)', 'b)', 'c)'], ['High level', 'Middle level', 'Low level'], [22500, 30000, 40000]):
    # label, altitude level, y-axis limit (hard-coded), y-axis ticks as ints, horizontal grid lines
    a.text(-0.11, 1.02, label, ha = 'center', va = 'center', transform = a.transAxes, fontsize = fs + 1, fontweight = 'bold')
    a.text(x = .1, y = .92, s = level, ha = 'center', va = 'center', fontsize = fs + 1, fontweight = 'bold', transform = a.transAxes)
    a.set_ylim(0, limit)
    a.set_yticks(a.get_yticks(), a.get_yticks().astype(int), fontsize = fs)
    a.grid(alpha = .4, axis = 'y')

    # color of tick labels, label of left axes, limit and ticks of right axes
    a.tick_params(axis='y', colors='tab:red')
    a_twin.tick_params(axis='y', colors='tab:blue')

    a.set_ylabel('Number of observations', fontsize = fs)

    a_twin.set_ylim(0, a.get_ylim()[1]/10)
    a_twin.set_yticks(a_twin.get_yticks(), a_twin.get_yticks().astype(int), fontsize = fs)
    

ax3.set_xticks(xax, xax, fontsize = fs)
ax3.set_xlabel('Cloud class', fontsize = fs)

#######################################################
# Small dummy plot to show order of stations in plots
ax_dummy = ax1.inset_axes([.5, .65, .15, .3])
for i, h, a in zip([-2, 0, 2], [.7, 1, .8], alpha_vec):

    ax_dummy.bar(i, h, fc = 'k', alpha = a, width = 2)

ax_dummy.set_xticks([-2,0,2], ['TU', 'WHW', 'VIE'], fontsize = fs - 3)
ax_dummy.get_yaxis().set_visible(False)
ax_dummy.set_xlim(-4, 4)
#########################################################

plt.savefig('BarPlot.png', bbox_inches = 'tight', dpi = 600)
plt.show()
