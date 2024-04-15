"""
-sls-
Various functions for multiple use or for the decluttering of main functions.

Functions
---------
- `sortdftime` -- deals with time column in dataframes.
- `fnames` -- get files and folder names in a directory.
- `groupbin` -- group values into specified bins.
- `check_input_df` -- check and change column headings for an inputted df.
- `conv2np` -- convert input data to a NumPy array if it's not already.
- `conv2list` -- convert input data to a list if it's not already.
- `dfcolinds` -- gets the column indexes from a dataframe.
- `pd2np` -- converts a pandas dataframe to a numpy array.
- `np2pd` -- converts numpy array inputs to a pandas dataframe.
- `ncols` -- get the number of columns.
- `nrows` -- get the number of rows.

Convert input data to a NumPy array if it's not already.
Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import os
import re
import pandas as pd
import numpy as np

def sortdftime(t):
    """
    Description
    ----------
    Converts the 'time' column of a df to datetime, then checks for NaT's, unsorted times, 
    and duplicate times, returning the df after corrections.
    """
    
    t['time'] = pd.to_datetime(t['time'], errors='coerce')
    t = t.dropna(subset=['time'])

    if not t['time'].is_monotonic_increasing: # if any times not in order
        t.sort_values(by=['time'], inplace=True)
        t.reset_index(drop=True, inplace=True)

    if t['time'].duplicated().any(): # if any duplicate times
        t = t.drop_duplicates()
        if t['time'].duplicated().any():
            # Identify duplicate rows based on the 'time' column
            dups = t.duplicated(subset=['time'], keep=False)
            # Function to find row with least NaNs in a group of duplicates
            def rownans(group):
                return group.loc[group.isna().sum(axis=1).idxmin()]
            # Apply the function to groups of duplicate rows
            #t = t[dups].groupby('time', as_index=False, include_groups=False).apply(rownans)
            t = t[dups].groupby('time', as_index=False).apply(rownans)

    return t

def fnames(directory, starts_with=None, ends_with=None, pattern=None, case_sensitive=True, folders='n'):
    """
    Description
    ----------
    Gets all the file names, and folder names if specified, in a given directory, with options for matching the 
    start and/or end, and/or a specific pattern, as well as case sensitivity.
    """

    items = os.listdir(directory)
    if case_sensitive:
        starts_with_func = str.startswith
        ends_with_func = str.endswith
        regex_flags = 0  # Case-sensitive
    else:
        starts_with_func = lambda s, prefix: s.lower().startswith(prefix.lower())
        ends_with_func = lambda s, suffix: s.lower().endswith(suffix.lower())
        regex_flags = re.IGNORECASE  # Case-insensitive
    
    if starts_with:
        items = [item for item in items if starts_with_func(item, starts_with)]
    if ends_with:
        items = [item for item in items if ends_with_func(item, ends_with)]
    if pattern:
        regex = re.compile(pattern, flags=regex_flags)
        items = [item for item in items if regex.search(item)]
    
    # Separate files and folders
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    
    if folders == 'y':
        files = [files, folders]

    return files

def groupbin(vals, bins, percentiles='n', right=False):
    """
    Description
    ----------
    Bin data into groups specified by bins, with an option to make bins percentiles, 
    e.g., 25, 50, and 75 would become the 25th, 75th and 75th perecentiles of vals.
    Returns an array the same size of vals containing the groups, specified by numbers e.g., [1 1 2 3 1 2]
    Thank you numpy for digitize! This function was much longer in Matlab....

    Additional Information
    ----------
    Remember!
    
    right   order of bins   returned index i satisfies

    False   increasing      bins[i-1] <= x < bins[i]

    True    increasing      bins[i-1] < x <= bins[i]

    False   decreasing      bins[i-1] > x >= bins[i]

    True    decreasing      bins[i-1] >= x > bins[i]
    """

    if percentiles == 'y':
        bins = np.percentile(vals, bins)

    groupbin_var = np.digitize(vals, bins, right=right)
    # groupbin_var = np.searchsorted(bins, vals, side='right' if right else 'left') # think same?

    return groupbin_var

def checkinputdf(data, func):
    """
    Description
    ----------
    Checks and changes inputted dataframes if not formatted properly.
    """

    if func == 'remove_msl':
        data.rename(columns={col: 'time' if pd.api.types.is_datetime64_any_dtype(data[col]) and col != 'time' else col for col in data.columns}, inplace=True)
        data.rename(columns={col: 'water levels' if pd.api.types.is_numeric_dtype(data[col]) and col != 'water levels' else col for col in data.columns}, inplace=True)

    return data

def conv2np(data, nans='ignore', force_2d=False):
    """
    Convert input array to a NumPy array if it's not already.
    """
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        if ncols(data) == 1 and force_2d:
            data = data.reshape(-1, 1)
    if nans == 'remove' and np.isnan(data).any():
        data = data[~np.isnan(data)]
        
    return data

def conv2list(data, flatten=True):
    """
    Convert input data to a list if it's not already.
    """
   
    if isinstance(data, (np.ndarray, pd.Series)):
        data = data.tolist()
    elif isinstance(data, (int, float)):
        data = [data]
    elif isinstance(data, pd.DataFrame):
        if flatten:
            data = data.values.flatten().tolist()
        else:
            data = data.values.tolist()

    return data

def dfcolinds(colnames):
    """
    Description
    ----------
    Gets the column names and positions from a dataframe.
    """
   
    colinds = dict(zip(colnames, list(range(0,len(colnames)))))
    # data[0, colinds['Name of the column']] = 9

    return colinds

def pd2np(data, timecol, colinds=True, nans='ignore'):
    """
    Description
    ----------
    Converts a pandas dataframe to a numpy array.
    """

    time = data[timecol].to_numpy()
    data = data.drop(columns=timecol)
    if colinds:
        colinds = dfcolinds(data.columns)
    else:
        colinds = None
    data = data.to_numpy()
    if nans == 'remove' and np.isnan(data).any():
        data = data[~np.isnan(data)]

    return data, time, colinds

def np2pd(data, time, colinds=True):
    """
    Description
    ----------
    Converts numpy array inputs to a pandas dataframe.
    """
    
    data = pd.DataFrame(data)
    data['time'] = time
    time = 'time'
    if colinds:
        colinds = dfcolinds(data.columns[data.columns != time])

    return data, time, colinds

def ncols(data):
    """
    Description
    ----------
    Get the number of columns.
    """
    
    try:
        return data.shape[1]
    except IndexError:
        return 1

def nrows(data):
    """
    Description
    ----------
    Get the number of rows.
    """
    
    try:
        return data.shape[0]
    except IndexError:
        return 1

def closest_vals(A, B, indexes=False):
    """
    Description
    ----------
    Find the closest values (or indexes) in B to those in A.

    Credit: Divakar (https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python/45350318#45350318)
    """
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx == L] = L - 1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx - 1]) < np.abs(A - sorted_B[sorted_idx])) )
    if indexes:
        out = sidx_B[sorted_idx-mask]
    else:
        out = B[sidx_B[sorted_idx-mask]]
    
    return out