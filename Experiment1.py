#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:13:37 2025

@author: mrosenberger
"""

import numpy as np
import datetime
import pandas as pd

# get raw data
df = pd.read_csv('Experiment1_non-simultaneous.csv')
# change time format
df.time = pd.to_datetime(df.time.values, dayfirst = 'True')

'''
Only use time steps with enough daylight to observe clouds properly
Never use images before 04UTC and after 20UTC.

Between 15.05. until 04.08. (incl.) 19UTC can be used.
Between 06.04. until 08.09. (incl.) 18UTC can be used.
Between 23.02. until 03.10. (incl.) 17UTC can be used.
Between 17.01. until 03.11. (incl.) 16UTC can be used.

Between 05.03. until 17.10. (incl.) 05UTC can be used.
'''

# drop before 04UTC and after 20UTC
df_day = df.loc[(df['time'].dt.hour > 4) & (df['time'].dt.hour < 20)]

# drop everything else, year-wise
for y in range(2012, 2025):
    df_day = df_day.drop(df_day.loc[(df_day['time'].dt.year == y) & (~df_day['time'].isin(pd.date_range(datetime.datetime(y, 3, 5, 0), datetime.datetime(y, 10, 17, 23), freq = 'h'))) & (df_day['time'].dt.hour <= 5)].index)
    df_day = df_day.drop(df_day.loc[(df_day['time'].dt.year == y) & (~df_day['time'].isin(pd.date_range(datetime.datetime(y, 1, 17, 0), datetime.datetime(y, 11, 3, 23), freq = 'h'))) & (df_day['time'].dt.hour >= 16)].index)
    df_day = df_day.drop(df_day.loc[(df_day['time'].dt.year == y) & (~df_day['time'].isin(pd.date_range(datetime.datetime(y, 2, 23, 0), datetime.datetime(y, 10, 3, 23), freq = 'h'))) & (df_day['time'].dt.hour >= 17)].index)
    df_day = df_day.drop(df_day.loc[(df_day['time'].dt.year == y) & (~df_day['time'].isin(pd.date_range(datetime.datetime(y, 4, 6, 0), datetime.datetime(y, 9, 8, 23), freq = 'h'))) & (df_day['time'].dt.hour >= 18)].index)
    df_day = df_day.drop(df_day.loc[(df_day['time'].dt.year == y) & (~df_day['time'].isin(pd.date_range(datetime.datetime(y, 5, 15, 0), datetime.datetime(y, 8, 4, 23), freq = 'h'))) & (df_day['time'].dt.hour >= 19)].index)

df = df_day.copy()

# name of column with observer data
col = 'Observer (anonymous)'
# count observations per each observer
observer_counts = df[col].value_counts()
# names of observers
observers = observer_counts.keys()

# first 7 observers conducted ca. 4300 to 9800 instances; 
# in total almost exactly 90% of instances
# the eigth conducted only 1276 instances
n_observers_plot = 7

# only use first 7 observers in 
observers_to_plot = observer_counts.keys()[:n_observers_plot]
# sort observers in ascending order
ascending_observer_number = np.argsort(observers_to_plot)

# define seasons and according months
# for Fig. 3a
seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}

# iterate through seasons
for si, season in enumerate(seasons):
    # get months of a season
    m1, m2, m3 = seasons[season]
    # get data of months
    df_season = df.loc[(df['time'].dt.month == m1) | (df['time'].dt.month == m2) | (df['time'].dt.month == m3) ]
    # count reports of each observer per season
    observer_counts_season = df_season[col].value_counts()

    # normalize for each observer
    season_values = observer_counts_season[observers_to_plot]/observer_counts[observers_to_plot]

# define time intervals
# for Fig. 3b
tods = {r'$05 - 08$': [5, 6, 7, 8], r'$09 - 12$': [9, 10, 11, 12], r'$13 - 16$': [13, 14, 15, 16], r'$17 - 19$': [17, 18, 19]}

for ti, tod in enumerate(tods):

    # get data of time intervals
    df_season = df.loc[df['time'].dt.hour.isin(tods[tod])]

    # count reports of each observer per time interval
    observer_counts_season = df_season[col].value_counts()# .to_frame()
    # normalize for each observer
    time_values = observer_counts_season[observers_to_plot]/observer_counts[observers_to_plot]

'''
ratio of observations of each class conducted by every observer
'''
# empty arrays to fill
# for Fig. 4a
all_values_rel = np.zeros(shape = (n_observers_plot+1, 30))
all_values = np.zeros(shape = (n_observers_plot+1, 30))

for i in range(n_observers_plot):
    # get observer
    o = df.loc[df[col] == observers[i]]

    # rewrite observations from 3x10 to 30 categories
    obs = np.array([o['CL'].values, o['CM'].values + 10, o['CH'].values + 20]).T
    # count obs per class
    values, _ = np.histogram(obs, bins = np.arange(-.5, 30.5))
    # write in array
    all_values_rel[i] = values/len(o) # relative per pbserver (Fig. 2a)
    all_values[i] = values # absolute

# all together
all_obs = np.array([df['CL'].values, df['CM'].values + 10, df['CH'].values + 20]).T
values, _ = np.histogram(all_obs, bins = np.arange(-.5, 30.5))
all_values_rel[-1] = values/len(df)
all_values[-1] = values

class_wise = all_values[:-1]/all_values[-1] # for Fig. 4b
