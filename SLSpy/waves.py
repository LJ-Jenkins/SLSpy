"""
-sls-
Various functions for use on wave data.

Functions
---------
- `stockdon` -- Calculate wave runup magnitude using Stockdon et al. (2006).

Convert input data to a NumPy array if it's not already.
Luke Jenkins May 2024
L.Jenkins@soton.ac.uk
"""

import numpy as np

def stockdon(
    slope: np.ndarray,
    Hs: np.ndarray,
    Tp: np.ndarray,
    IrbFull: bool = False,
    SimplifiedReflective: bool = False,
    tanslope: bool = False
):
    """
    Calculate wave runup magnitude using Stockdon et al. (2006).
    
    Parameters
    ----------
    slope : np.ndarray
        Foreshore beach slope
    Hs : np.ndarray
        Offshore (deep water) significant wave height
    Tp : np.ndarray
        Peak wave period
    IrbFull : bool, optional
        Use the full Iribarren number calculation (default is False)
    SimplifiedReflective : bool, optional
        Use the simplified reflective beach runup (default is False)
    tanslope : bool, optional
        Use tan(slope) instead of slope (default is False)
    
    Returns
    -------
        setup component of runup, swash component of runup, total runup
    """
    
    if tanslope:
        slope = np.tan(slope)

    T2 = Tp ** 2
    L0 = (9.81 * T2) / (2 * np.pi)
    
    if IrbFull:
        iribarren = slope / np.sqrt(Hs / L0)
    else:
        iribarren = slope / np.sqrt(Hs / L0)
    
    setup = np.full_like(Tp, np.nan)
    swash = np.full_like(Tp, np.nan)
    runup = np.full_like(Tp, np.nan)

    d = iribarren < 0.3  # dissipative conditions
    g = iribarren >= 0.3  # general application

    setup[d] = 0.016 * np.sqrt(Hs[d] * L0[d])
    swash[d] = 0.046 * np.sqrt(Hs[d] * L0[d])
    runup[d] = 0.043 * np.sqrt(Hs[d] * L0[d])

    setup[g] = 0.35 * slope[g] * np.sqrt(Hs[g] * L0[g])
    swash[g] = np.sqrt(Hs[g] * L0[g] * (0.563 * slope[g]**2 + 0.004)) / 2
    runup[g] = 0.043 * np.sqrt(Hs[g] * L0[g])

    if SimplifiedReflective:
        r = iribarren > 1.25
        runup[r] = 0.73 * slope[r] * np.sqrt(Hs[r] * L0[r])

    return setup, swash, runup



