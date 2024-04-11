"""
-sls-
Functions for working with exceedances.

Functions
---------
- `thresh_cross_del` -- delete all but the maximum value in groups of consecutive threshold crossings.
- `time_above_threshold` -- calculate the time in hours for each group of consecutive threshold crossings.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import numpy as np
import itertools
from . import tools

def thresh_cross_del(data, threshold):
    """
    Description
    ----------
    Delete all but the maximum value in groups of consecutive threshold crossings.

    Parameters
    ----------
    data : array
        Array of data.
    threshold : float
        Threshold value. If multiple values are given, columns in data should correspond 
        to each threshold and the crossing deletion will occur for each column.

    Returns
    -------
    data : array
        Array of data with all but the maximum value in each group of consecutive threshold crossings set to NaN.
    """

    threshold = tools.conv2list(threshold)
    data = tools.conv2np(data, force_2d=True)

    for i in range(0, tools.ncols(data)):
        indices = np.where(data[:, i] > threshold[0])[0]
        # groups = [list(g) for k, g in itertools.groupby(data, key=lambda x: x > threshold) if k] # gets the data
        # Group consecutive indices
        groups = [list(g) for _, g in itertools.groupby(indices, key=lambda x, c=itertools.count(): x - next(c))]

        # Iterate through each group
        for group_indices in groups:
            if len(group_indices) > 1:
                # Find the index of the maximum value in the group
                max_index = group_indices[np.argmax(data[group_indices])]
                # Set all values except the maximum to NaN
                naninds = [x for x in group_indices if x != max_index]
                data[naninds, i] = np.nan

    return data

def time_above_threshold(data, time='time', threshold=None, include_thresh=False, colnames=False):
    """
    Description
    ----------
    Calculate the time in hours for each group of consecutive threshold crossings.
    Threshold exceedances with only 1 consecutive value will have a time of 0 hours.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the data with a column for timestamps, or a data array.
    time : str or array
        The name of the column containing the times in a dataframe, or an array of times.
    threshold : float, int, array or list
        The threshold value/s for exceedances. If multiple columns in data, multiple thresholds should be given.
    include_thresh : bool, optional
        Boolean flag to include values equal to the threshold.
        If True, considers values greater than or equal to the threshold.
        If False, considers values strictly greater than the threshold.
    colnames : list, optional
        List of column names for the data. If not given, dataframe column names will be used,
        otherwise names will be 'Var 1' etc.
        
    Returns
    -------
    time_above_thresh : dict or list
        Dictionary containing the time in hours for each group of consecutive threshold crossings,
        as well as the times corresponding to the first and last values in each group,
        or the only time for exceedances of length 1. If only one column, a list is returned.
    """
    
    if len(threshold) == 1 and not threshold:
        raise ValueError("Threshold/s must be given.")
    threshold = tools.conv2list(threshold)
    
    if isinstance(time, str):
        data, time, colinds = tools.pd2np(data, time)
    else:
        colinds = False

    if not colinds and not colnames:
        colnames = ['Var ' + str(i) for i in range(1, tools.ncols(data) + 1)]
    if colinds and not colnames:
        colnames = [key for key in colinds.keys()]

    time_above_thresh = {}
    for j in range(0, tools.ncols(data)):
        if include_thresh:
            indices = np.where(data[:, j] >= threshold[j])[0]
        else:
            indices = np.where(data[:, j] > threshold[j])[0]
        groups = [list(g) for _, g in itertools.groupby(indices, key=lambda x, c=itertools.count(): x - next(c))]
        # Calculate time in hours for each group
        t = [None] * len(groups)
        tat = np.empty((len(groups),1))
        for i in range(0, len(groups)):
            group_times = time[groups[i]]
            # Calculate time difference between first and last timestamps within each group
            t[i] = group_times[0] if len(group_times) == 1 else [group_times[0], group_times[-1]]
            tat[i] = (group_times[-1] - group_times[0]) / np.timedelta64(1, 's')

        time_above_thresh[colnames[j]] = [tat, t]
        
    if len(time_above_thresh) == 1:
        time_above_thresh = time_above_thresh[colnames[j]]

    return time_above_thresh

