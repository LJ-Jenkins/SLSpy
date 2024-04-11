"""
-sls-
Functions for NNRCMP wave and water level data.

Functions
---------
- `load_wave` -- loads NNRCMP wave data from all files in a directory.
- `load_wl` -- loads NNRCMP water level data from all files in a directory.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import os
import pandas as pd
from . import tools

def load_wave(sitecode, directory, flags=None, dispfile='n'):
    """
    Description
    ----------
    Loads NNRCMP qc wave data from files that contain the given sitecode in a directory.

    Parameters
    ----------
    sitecode : str
        NNRCMP three letter sitecode e.g., "PBy" (case insensitive).
    directory : str
        Directory of the files.
    flags : list, containing int
        Flags for which data should be removed, if 'df' is given then defaults are used (see below).
    dispfile : str ('y' or 'n')
        Choose to show the file being loaded at the command line. 

    Returns
    -------
    t : 
        Dataframe of time, water level (cd), water level (od), and flags.

    Additional Information
    -------
    Parameters

    Significant wave height: Hs (m)
    Spectrally-derived zero-crossing wave period: Tz (s)
    Peak wave period: Tp (s)
    Wave direction: Dir (°)
    (Directional) Spread: Spd (°)
    Maximum observed wave height: Hmax (m)
    Sea surface temperature: SST (C°)

    CCO Wave QC Data Quality Control Flags (as of January 2022)

    -Note flag 7 (bouy off location) is set to remove all data parameters (except lat/lon) if specified,
    not none like the table below suggests. Defaults are: 1, 2, 3, 4, 8, 9. 
 
    Flag	Suspect Parameters          Not Suspect Parameters      Description

    0	                                All data pass   
    	    
    1	    Hs, Hmax Tz, Tp,            SST                         Either Hs or Tz fail,
            Direction, Spread		                                so all data fail, except SST

    2	    Tp, Direction, Spread	    Hs, Hmax, Tz, SST	        Tp fail + derivatives

    3	    Direction, Spread	        Tp, Hs, Hmax, Tz, SST	    Direction fail + derivatives

    4	    Spread	                    Direction, Tp, Hs, Hmax,    Spread fail
                                        Tz, SST	

    7		                            Hs, Hmax, Tz, Tp,           Buoy off location
                                        Direction, Spread, SST

    8	    SST	                        Hs, Hmax, Tz, Tp,           SST fail
                                        Direction, Spread

    9	    Hs, Hmax, Tz, Tp,                                       Missing data
            Direction, Spread, SST	
    """
    
    files = tools.fnames(directory, ends_with='.txt', pattern=sitecode, case_sensitive=False)

    t_list = [0] * len(files) # List to store DataFrames

    for i, file in enumerate(files):
        if dispfile == 'y':
            print(file)
        filein = os.path.join(directory, file)
        T = pd.read_csv(filein, skiprows=1, header=None, delimiter='\t')
        t_list[i] = T

    t = pd.concat(t_list, ignore_index=True)
    t.columns = ['time', 'latitude', 'longitude', 'flags', 'Hs (m)', 'Hmax (m)', 'Tp (s)',
                'Tz (s)', 'Dir (degrees)', 'Spd (degrees)', 'SST (degrees C)']
        
    t = tools.sortdftime(t)

    if flags is not None:
        # Define flags and their corresponding columns to set as NaN
        flag_columns_mapping = {
            1: ['Hs (m)', 'Hmax (m)', 'Tp (s)', 'Tz (s)', 'Dir (degrees)', 'Spd (degrees)'],
            2: ['Tp (s)', 'Dir (degrees)', 'Spd (degrees)'],
            3: ['Dir (degrees)', 'Spd (degrees)'],
            4: ['Spd (degrees)'],
            7: ['Hs (m)', 'Hmax (m)', 'Tp (s)', 'Tz (s)', 'Dir (degrees)', 'Spd (degrees)', 'SST (degrees C)'],
            8: ['SST (degrees C)'],
            9: ['Hs (m)', 'Tp (s)', 'Tz (s)', 'Dir (degrees)', 'Spd (degrees)', 'SST (degrees C)']
        }
        if flags == 'df':
            flags = [1, 2, 3, 4, 8, 9]
        for flag in flags:
            if flag in flag_columns_mapping:
                cols = flag_columns_mapping[flag]
                t.loc[t['flags'] == flag, cols] = pd.NA

    return t

def load_wl(sitecode, directory, flags=None, dispfile='n'):
    """
    Description
    ----------
    Loads NNRCMP qc water level data from files that contain the given sitecode in a directory.

    Parameters
    ----------
    sitecode : str
        NNRCMP three letter sitecode e.g., "PBy" (case insensitive).
    directory : str
        Directory of the files.
    flags : list, containing int
        Flags for which data should be removed, if 'df' is given then defaults are used (see below).
    dispfile : str, 'y' or 'n'
        Choose to show the file being loaded at the command line. 

    Returns
    -------
    t : 
        Dataframe of time, water level (cd), water level (od), and flags.

    Additional Information
    -------
    Data quality is marked by one column of flags.
    Defaults are: 3, 4, 7, 8, 9

    Flag = 1: Correct value
    Flag = 2: Interpolated value
    Flag = 3: Doubtful value
    Flag = 4: Isolated spike or wrong value
    Flag = 5: Correct but extreme value
    Flag = 6: Reference change detected
    Flag = 7: Constant value
    Flag = 8: Out of range
    Flag = 9: Missing data
    """

    #files = [f for f in os.listdir(directory) if f.endswith('.txt') and sitecode in f]
    files = tools.fnames(directory, ends_with='.txt', pattern=sitecode, case_sensitive=False)

    t_list = [0] * len(files) # List to store DataFrames

    for i, file in enumerate(files):
        if dispfile == 'y':
            print(file)
        filein = os.path.join(directory, file)
        T = pd.read_csv(filein, skiprows=1, header=None, delimiter='\t')
        t_list[i] = T

    t = pd.concat(t_list, ignore_index=True)
    t.columns = ['time', 'wl (OD) (m)', 'wl (CD) (m)', 'residual (m)', 'flags']
    
    t = tools.sortdftime(t)

    if flags is not None:
        if flags == 'df':
            flags = [3, 4, 7, 8, 9]
        cols = ['wl (od) (m)', 'wl (cd) (m)', 'residual (m)']
        t.loc[t['flags'].isin(flags), cols] = pd.NA # remove flagged data
        for col in cols: # remove any remaining 99 or -99 data
            t.loc[(t[col] > 90) | (t[col] < -90), col] = pd.NA

    return t










