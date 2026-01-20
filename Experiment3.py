#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:13:17 2026

@author: mrosenberger
"""

import numpy as np
import h5py


def get_accuracy(ground_truth, observation, bins = [-.5, .5, 1.5, 2.5, 3.5]):
    '''

    Calculates agreement between 2 one-hot encoded arrays.
    Each array has to have exactly three observed classes per instance

    Parameters
    ----------
    ground_truth : 1d-array, shape == (n_instances, n_classes)
        Truth vector of cloud classes
    observation : 1d-array, shape == (n_instances, n_classes)
        Observation vector of cloud classes
    bins : array-like, optional
        Bins for the histogram function. At this point only the default makes sense. The default is [-.5, .5, 1.5, 2.5, 3.5]

    Returns
    -------
    float
        Accuracy averaged over all instances
    '''

    # check if both vectors have exactly three observations per instances
    assert np.all(ground_truth.sum(axis = 1) == 3)
    assert np.all(observation.sum(axis = 1) == 3)

    # get length of vectors
    n_observations = len(ground_truth)

    # sum of both vectors results in value 2 if they agree, 0 or 1 otherwise
    # Only looking for agreement
    counts = ((ground_truth + observation) == 2).sum(axis = 1)
    # get cardinalities of number of identically reported classes
    counts = np.histogram(counts, bins = bins)[0]

    # result is weigthed with thirds and averaged
    return (counts*[0, 1/3, 2/3, 1]).sum()/n_observations


# load ground truth, observations of volunteers and MV 
# both SYNOP scheme and collapsed scheme
# both all-sky and atlas data
with h5py.File('Experiment3.nc', mode='r') as file:
    allsky = file['All-sky'][:]
    allsky_16c = file['All-sky_16c'][:]
    atlas = file['Atlas'][:]
    atlas_16c = file['Atlas_16c'][:]

# empty lists to fill
accuracy_vs_gt = []
accuracy_vs_Obs2 = []
accuracy_vs_16c = []

# bins as default
bins = [-.5, .5, 1.5, 2.5, 3.5]

# iterate over image sources and size of classification schemes
for data, data_16c in zip([allsky, atlas], [allsky_16c, atlas_16c]):
    
    # extract values from loaded arrays
    # truth
    gt = data[0]
    Obs2 = data[2]
    Obs2_16c = data_16c[2]

    # volunteers
    observations = data[1:6]
    observations_16c = data_16c[1:6]

    # MV
    MV = data[6]
    MV_16c = data_16c[6]

    # iterate over volunteers
    for iobs, (obs, obs_16c) in enumerate(zip(observations, observations_16c)):

        # Agreement with ground truth
        accuracy_vs_gt.append(get_accuracy(gt, obs, bins))

        # only if not Observer 2 is investigated
        if iobs != 1:
            
            # Agreement with Observer 2
            accuracy_vs_Obs2.append(get_accuracy(Obs2, obs, bins))

            # Agreement with Observer 2 using smaller scheme
            accuracy_vs_16c.append(get_accuracy(Obs2_16c, obs_16c, bins))

        # Observer 2 not compared with themselves, i.e. NaNs for accuracy
        else:
            accuracy_vs_Obs2.append(np.nan)
            accuracy_vs_16c.append(np.nan)

        # just counting ...
        iobs += 1

    # Agreement MV with ground truth
    accuracy_vs_gt.append(get_accuracy(gt, MV, bins))

    # Agreement MV with Observer 2  
    accuracy_vs_Obs2.append(get_accuracy(Obs2, MV, bins))

    # Agreement MV with Observer 2 using smaller scheme
    accuracy_vs_16c.append(get_accuracy(Obs2_16c, MV_16c, bins))

# Split resulting accuracies by data source
accuracy_vs_gt = np.stack((accuracy_vs_gt[:6], accuracy_vs_gt[6:]))
accuracy_vs_Obs2 = np.stack((accuracy_vs_Obs2[:6], accuracy_vs_Obs2[6:]))
accuracy_vs_16c = np.stack((accuracy_vs_16c[:6], accuracy_vs_16c[6:]))
