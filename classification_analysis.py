#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mrosenberger
"""

import numpy as np
import tensorflow as tf

#########################################################################################################################################################

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

#########################################################################################################################################################

def get_pairwise_weightedMCC_withbt(dataset, perc):
    '''

    Calculates pairwise agreement between 2 one-hot encoded arrays.
    Each array has to have exactly three observed classes per instance

    Parameters
    ----------
    dataset : 3d-array, shape == (n_observers, n_instances, n_classes)
        One-hot encoded observations - gt, five observers, MV 
    perc : float
        Percentile for which statistical significance shall be calculated
    
    Returns
    -------
    7x7 array
        Pairwise MCC between all observers averaged over all classes and instances
    
    7x7 array
        Annotations corresponding to pair-wise agreement values for heatmap plot
    '''

    # how many different classes
    n_classes = np.shape(dataset)[-1]
    # this will be the output
    weighted_avg = []
    # bootstrapped sample
    bt_sample = []
    # sum MLCMs of all comparisons between gt and any observer 
    # to get "real" distribution for bootstrap sampling
    MLCM_summed = np.zeros(shape = (n_classes+1, n_classes+1))

    # weights are always given by ground truth distribution for consistency
    wgts = np.sum(dataset[0], axis = 0)[:30]

    # iterate over dataset twice
    for i1, obs1 in enumerate(dataset):
        for i2, obs2 in enumerate(dataset):

            MCC_values = []
            # calculate multi-label confusion matrix (Heydarian et al., 2022)
            MLCM = calculate_MLCM(obs1, obs2, from_logits=False)
            # "real" distribution for bootstrapping only for comparisons
            # between ground truth and single observers
            if i1 == 0 and 0 < i2 < 6:
                MLCM_summed += MLCM

            # calculate MCC classwise
            for i in range(n_classes):

                output = calculate_MCC(MLCM, index = i, do_bootstrap = False)
                MCC_values.append(output['MCC'])

            # weighted average of classwise MCC
            weighted_avg.append(np.ma.average(np.ma.array(MCC_values, mask=np.isnan(MCC_values)), weights=wgts))

    # bootstrapping sample from summed MLCM
    for i in range(n_classes):

        output = calculate_MCC(MLCM_summed, index = i, do_bootstrap = True)
        bt_sample.append(output['bt_MCC'])

    # calculate classwise weighted average for each run
    bt_sample = np.array(bt_sample).transpose()
    bt_avg = [np.ma.average(np.ma.array(i, mask=np.isnan(i)), weights=wgts) for i in bt_sample]
    # get border value for statistical significance based on given percentile
    border = np.percentile(bt_avg, perc)

    # add asterisk to annotation of each box 
    # where observed value is below border value
    sig_annot = []

    for v in weighted_avg:
        if v > border:
            sig_annot.append('{:.2f}'.format(v))
        else:
            sig_annot.append('{:.2f}*'.format(v))

    # reshape observed values and corresponding annotations for correct placement in matrix
    return np.array(weighted_avg).reshape(7,7), np.array(sig_annot).reshape(7,7)

#########################################################################################################################################################

def calculate_MLCM(ground_truth, predicted_classes, thresh = 0.5, from_logits = False):
    '''

    Calculation of the Multi-Label Classification Matrix as defined by Heydarian et al. (2022)

    Parameters
    ----------
    ground_truth : 2d-array, shape == (n_instances, n_classes)
        Matrix holding the ground-truth observations one-hot encoded.
    predicted_classes : 2d-array, shape == (n_instances, n_classes)
        Matrix holding the model predictions, either probabilities or binary.
    thresh : float or array-like, optional
        Threshold probability for a prediction to be considered as true prediction.
        Can be a single value valid for each class or an array-like object with length "n_classes"
        if each class has a specific value. The default is 0.5.
    return_plot : string, optional
        Defines which plot is to be returned. Either "None" or "absolute" or "precision" or "recall" or "all". The default is 'all'.
    types : array-like, optional
        Array-like object containing the class-names to be used as tick parameters in plot. The default is None.
    from_logits : Boolean, optional
        Stating if the argument "predicted_classes" consists of logits. If "True" sigmoid activation is applied.
        If "False" nothing happens. The default is False.
    **kwargs : keyword(s)
        Additional keyword input for the plotting routine, e.g. cmap = 'Reds'.

    Returns
    -------
    2d-array
        The calculated values of the Multi-Label Confusion Matrix.
    Figure
        If stated by "return_plot" a plot of the MLCM of the corresponding measure is returned.
    '''
    
    # check shapes of ground truth and predictions
    assert np.shape(ground_truth) == np.shape(predicted_classes)
    
    # one probability threshold for all classes
    if type(thresh) is float:
        thresh = [thresh]*np.shape(ground_truth)[-1]

    # one probability threshold per class
    elif type(thresh) is list:
        thresh = np.array(thresh)
    
    # convert logits to probabilities if necessary
    # sigmoid activation for multi-label prediction
    if from_logits == True:
        predicted_classes = tf.keras.activations.sigmoid(predicted_classes)
    
    num_classes = np.shape(ground_truth)[1] # number of classes
    MLCM = np.zeros(shape = (num_classes+1, num_classes+1)) # empty MLCM matrix
    
    # filling the MLCM according to Heydarian et al. (2022)
    for y_true, y_pred in zip(ground_truth, predicted_classes):
        
        T = np.where(y_true == 1)[0]
        P = np.where(y_pred >= thresh)[0]
        
        T1 = []
        T2 = []
        P2 = []
        
        for p in P:
            if p in T:
                T1.append(p)
            else:
                P2.append(p)
        
        
        for t in T:
            if t not in T1:
                T2.append(t)
                
        # Step 1 (same for all categories)
        for t in T1:
            MLCM[t, t] += 1
        
        if len(P) == 0 and len(T) == 0:
            MLCM[num_classes, num_classes] += 1
        
        
        if P2 == []:
            # Step 2 of category 1
            for t in T2:
                MLCM[t, num_classes] += 1
                
        elif T2 == [] and P2 != []:
            # Step 2 of category 2
            for p in P2:
                if len(T) == 0:
                    MLCM[num_classes, p] += 1 
                else:
                    for t in T: 
                        MLCM[t, p] += 1 
                        
        elif T2 != [] and P2 != []:
            # Step 2 of category 3
            for p in P2:
                for t in T2: 
                    MLCM[t, p] += 1 
    
    return MLCM

#########################################################################################################################################################

def Bootstrap_CM(TP, FP, FN, TN, n_draws = 10000):
    '''
    Draws a given number of randomly arranged permutations of a confidence matrix (CM) 
    according to Fowlkes et al. (1983) drawn from a hypergeometric distribution.

    Parameters
    ----------
    TP : int
        Number of True Positives in confidence matrix.
    FP : int
        Number of False Positives in confidence matrix.
    FN : int
        Number of False Negatives in confidence matrix.
    TN : int
        Number of True Negatives in confidence matrix.
    n_draws : int, optional
        Number of random draws, that should be made. The default is 10 000.
        
    Returns
    -------
    TP_bootstrap, FP_bootstrap, FN_bootstrap, TN_bootstrap: Directory
        The directory contains the arrays holding "n_draws" values of each of these four statistics.

    '''

    # calculate row- and column-wise sums of CM
    pos_obs = TP+FN # number of observations
    pos_pred = TP+FP # number of predictions
    neg_obs = FP+TN # number of non-observations
    neg_pred = FN+TN # number of non-predictions

    random_TP = []
    random_FP = []
    random_FN = []
    random_TN = []

    '''
    According to Fowlkes et al. (1983), one has to use a hypergeometric distribution. 
    np.random.hypergeometric needs 3 input variables:
        -) ngood: number of good selections
        -) nbad: number of bad selections
        -) nsample: number of items sampled

    for a given statistic:
        -) ngood is the sum of the row in which the statistic is located
        -) nbad = N - ngood, where N = TP + FP + FN + TN
        -) nsample is the sum of the column in which the statistic is located

    IMPORTANT: rows and columns are interchangeable in this context!
    '''

    for n in range(n_draws):
        # decide randomly which statistic is drawn first, otherwise biases could appear
        # and not all theoretically possible values are possibly drawn
        first_n = np.random.randint(0,4) # 0 -> TP, 1 -> FP, 2 -> FN, 3 -> TN
        
        # TP:
        if first_n == 0:
            # random draw with above described parameters
            draw_TP = np.random.hypergeometric(ngood = pos_obs, nbad = neg_obs, nsample = pos_pred)

            # calculate the other statistics given unchanged row- and column-wise sums
            draw_FN = pos_obs - draw_TP
            draw_FP = pos_pred - draw_TP
            draw_TN = neg_obs - draw_FP
        
        # FP:
        elif first_n == 1:
            draw_FP = np.random.hypergeometric(ngood = neg_obs, nbad = pos_obs, nsample = pos_pred)

            draw_TP = pos_pred - draw_FP
            draw_TN = neg_obs - draw_FP
            draw_FN = pos_obs - draw_TP
        
        # FN:
        elif first_n == 2:
            draw_FN = np.random.hypergeometric(ngood = pos_obs, nbad = neg_obs, nsample = neg_pred)

            draw_TP = pos_obs - draw_FN
            draw_TN = neg_pred - draw_FN
            draw_FP = neg_obs - draw_TN
        
        # TN:
        elif first_n == 3:
            draw_TN = np.random.hypergeometric(ngood = neg_obs, nbad = pos_obs, nsample = neg_pred)

            draw_FP = neg_obs - draw_TN
            draw_FN = neg_pred - draw_TN
            draw_TP = pos_obs - draw_FN

        random_TP.append(draw_TP)
        random_FP.append(draw_FP)
        random_FN.append(draw_FN)
        random_TN.append(draw_TN)

    random_TP = np.array(random_TP)
    random_FP = np.array(random_FP)
    random_FN = np.array(random_FN)
    random_TN = np.array(random_TN)

    # some final checks
    assert np.min((random_TP, random_FP, random_FN, random_TN)) >=0    
    assert np.all(random_TP + random_FN == pos_obs)
    assert np.all(random_TP + random_FP == pos_pred)
    assert np.all(random_FP + random_TN == neg_obs)
    assert np.all(random_FN + random_TN == neg_pred)

    return {'random_TP': random_TP, 'random_FP' : random_FP,
            'random_FN' : random_FN, 'random_TN' : random_TN}


#########################################################################################################################################################

def calculate_MCC(MLCM, index, do_bootstrap = True, **kwargs):
    '''
    Parameters
    ----------
    MLCM : (n+1) x (n+1) matrix
        (n+1) x (n+1) matrix with n being the number of classes. 
        TP, FP, FN, TN are calculated from the MLCM
    index : int
        0 <= index <= n-1 to decide for which class measures are calculated.
    do_bootstrap : boolean, optional
        States if bootstrap sampling of TP, FP, FN, TN shall be made. The default is True.
    **kwargs : int
        Only possible keyword is "n_draws" to state how large the bootstrap sample has to be. Default is 10 000.

    Returns
    -------
    dict
        Dictionary holding all the measures in alphabetical order.
        If "do_bootstrap" == True after each measure all respective bootstrapped values are given 
        with key "bt_" + *name of measure*. 

    '''
    # read TP, FP, FN, TN from MLCM
    TP = MLCM[index, index]
    FN = np.sum(MLCM[index])-TP
    FP = np.sum(MLCM[:,index])-TP

    main_diagonal = np.eye(len(MLCM))*MLCM
    TN = np.sum(main_diagonal)-TP

    # create output directory with dummy values
    output_dict = {}

    # calculate MCC
    num = TP*TN - FP*FN
    denom = np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))

    # Chicco et al. (2020) showed that 
    #   -) MCC -> 0 if a row- or column-wise sum of the confidence matrix equals 0
    #   -) MCC -> 1 if only one entry of the matrix is > 0, i.e. the whole sample is located in a single box
    # only the first case is considered here, since the second one does not occur anyway

    if denom == 0:
        output_dict['MCC'] = ( 0 )
    else:
        output_dict['MCC'] = ( num/denom )

    if do_bootstrap == True:
        # create random rearrangements of the collapsed MLCM
        bt = Bootstrap_CM(TP, FP, FN, TN, **kwargs)
                
        bt_TP = bt['random_TP']
        bt_FP = bt['random_FP']
        bt_FN = bt['random_FN']
        bt_TN = bt['random_TN']
        
        bt_num = bt_TP*bt_TN - bt_FP*bt_FN
        bt_denom = np.sqrt((bt_TP+bt_FP) * (bt_TP+bt_FN) * (bt_TN+bt_FP) * (bt_TN+bt_FN))

        output_dict['bt_MCC'] = ( bt_num/bt_denom )    

    return output_dict

#########################################################################################################################################################