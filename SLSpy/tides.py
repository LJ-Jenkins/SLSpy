"""
-sls-
Functions to do with tides.

Functions
---------
- `calc_tide` -- calculates tide using UTide from sea level data.
- `tidal_phase` -- gets the tidal phase relative to high/low water.
- `high_low_water` -- gets the high and/or low water from tidal levels.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import pandas as pd
import numpy as np
import itertools
from utide import solve, reconstruct
from scipy.interpolate import interp1d
from multiprocessing import Pool
from warnings import warn
from . import tools
from . import exc
from . import stats


from datetime import datetime, timedelta


from scipy.signal import find_peaks
from scipy.ndimage import label, find_objects


# __all__ = [
#     'calc_tide', 'tidal_phase', 'high_low_water'
#     ]


def calc_tide(
    time,
    water_levels,
    latitude,
    period="yearly",
    newtime="n",
    newtime_freq="15min",
    cov_perc=50,
    cov_npoints=7000,
    ncnstits=None,
    msl_kwargs={
        "trend": "linear",
        "perc_cover": 75,
        "degf": 1,
        "out_type": "array",
    },
    **solve_kwargs
):
    """
    Description
    ----------
    Calculates the tide using UTide from sea level data.

    Parameters
    ----------
    time : array-like
        Times.
    water_levels : ndarray
        Still water levels.
    latitude : ndarray
        Latitude of target location.
    period : str, optional
        Period of harmonic analysis. Options are 'yearly' for a year-by-year anaylsis or
        'whole' for the whole timeseries at once.
    newtime : str, optional
        Whether to create a new time array. Options are 'y' or 'n'.
    newtime_freq : str, optional
        Frequency of the new time array (pandas freq inputs see pd.date_range).
    cov_perc : float, optional
        Data coverage percentage to ignore, e.g., if 50 then only years with more than
        50% coverage will be considered for the year by year harmonic analysis.
    cov_npoints : int, optional
        Number of points, e.g., if 7000 then only years with more than 7000 points will be
        considered for the year by year harmonic analysis.
    ncnstits : int, optional
        Number of constituents to use in the tidal reconstruction.
    msl_kwargs : dict, optional
        Keyword arguments for the sls remove_msl function.
        Defaults: {'trend': 'linear', 'perc_cover': 75, 'degf': 1, 'out_type': 'array'}
        Defaults are set *in the function definition*. These defaults are the same as the
        defaults for the remove_msl function and therefore by just changing one the others
        will remain as their defaults.
    solve_kwargs : any keyword argument for the utide 'solve' function.
        Keyword arguments for the utide solve function.
        Defaults: {'cnstit': 'auto', 'conf_int': 'linear', 'method': 'ols', 'trend': False, 'white': True}
        Defaults are set *within the function body* and therefore to overwrite these defaults
        each will need to be given in turn. If only one is given, the others will remain
        as their defaults.

    Returns
    -------
    tide : ndarray
        Tidal levels.
    t_time : ndarray
        Times of the tide.
    coef : dict
        Coefficients.
    """

    # default utide solve settings
    # defaults are set here and any user given solve_kwargs that are the same will overwite these defaults
    def_solve_settings = {
        "constit": "auto",
        "conf_int": "linear",
        "method": "ols",
        "trend": False,
        "white": True,
    }
    solve_kwargs = {**def_solve_settings, **solve_kwargs}

    water_levels = tools.conv2np(water_levels)

    # remove msl trends
    water_levels = stats.remove_msl(water_levels, time=time, **msl_kwargs)

    if newtime == "y":
        yrs = np.arange(time[0].year, time[len(time) - 1].year + 1)
        t_time = pd.date_range(
            start=str(yrs[0]), end=str(yrs[-1]), freq=newtime_freq
        )
    else:
        yrs = time.dt.year.unique()
        t_time = time

    if isinstance(t_time, pd.Series):
        t_time_array = t_time.dt.year
    elif isinstance(t_time, pd.DatetimeIndex):
        t_time_array = t_time.year

    if period == "yearly":
        print("Tidal analysis year by year")

        # Get the year by year coverage checks (% cover and n points)
        # and drop years that do not meet the given coverage
        yr_array = time.dt.year
        tide = [0] * len(yrs)
        cov_checks = np.zeros((len(yrs), 3))

        for j, yr in enumerate(yrs):
            yr_mask = yr_array == yr
            yr_water_levels = water_levels[yr_mask]
            cov_checks[j, 0] = yr
            cov_checks[j, 1] = np.sum(~np.isnan(yr_water_levels))
            if cov_checks[j, 1] > 0:
                cov_checks[j, 2] = (
                    cov_checks[j, 1] / len(yr_water_levels) * 100
                )

        if cov_npoints:
            cov_checks = cov_checks[cov_checks[:, 1] > cov_npoints]
        if cov_perc:
            cov_checks = cov_checks[cov_checks[:, 2] > cov_perc]

        yrs_meeting_checks = cov_checks[:, 0]

        for j, yr in enumerate(yrs):
            succesful_solve = False
            coef_result = np.nan

            while not succesful_solve and np.isnan(coef_result):
                nearest_yr = yrs_meeting_checks[
                    np.abs(yrs_meeting_checks - yr).argmin()
                ]
                yr_mask = yr_array == nearest_yr
                yr_water_levels = water_levels[yr_mask]
                yr_time = time[yr_mask]
                try:
                    coef = solve(
                        yr_time, yr_water_levels, lat=latitude, **solve_kwargs
                    )
                    succesful_solve = True
                    coef_result = coef.mean

                except:
                    warn(
                        "solve FAILED: ignoring "
                        + str(nearest_yr)
                        + " and attempting the next nearest year with coverages met.\nConsider amending coverage inputs"
                    )
                    yrs_meeting_checks = yrs_meeting_checks[
                        yrs_meeting_checks != nearest_yr
                    ]

            if ncnstits:
                tide_bunch = reconstruct(
                    t_time[t_time_array == yr],
                    coef,
                    constit=coef["name"][:ncnstits],
                )

            else:
                tide_bunch = reconstruct(t_time[t_time_array == yr], coef)

            tide[j] = tide_bunch.h

        tide = np.hstack(tide)

    elif period == "whole":
        print("Tidal analysis for whole period")

        coef = solve(time, water_levels, lat=latitude, **solve_kwargs)
        tide_bunch = reconstruct(t_time, coef)
        tide = tide_bunch.h

    if period == "yearly" and ncnstits is not None:
        coef = solve(time, water_levels, lat=latitude, **solve_kwargs)

    return tide, t_time, coef


def high_low_water(data, time="time", crossingvalue=0, high=True):
    """
    Description
    ----------
    Calculates the high and low waters from given water level or tide data by getting the maxima/minima
    of consecutive groups of values above/below a crossingvalue. Function is designed to be used with a
    more sinusodial tidal curve, typically generated from the 3 main tidal constituents.

    Parameters
    ----------
    data : pandas DataFrame or array
        DataFrame containing the data with a column for timestamps or an array.
    time : str, array
        Column name for the times (str) or an array of times if data is also an array.
    crossingvalue : float, optional
        Value to cross to get the high/low water levels. default is 0.
    high : bool, optional
        Boolean flag to calculate the high water levels.
        If False, calculates the low water levels.

    Returns
    -------
    hl_water : pandas DataFrame
        DataFrame containing the time and high/low water levels.
    """

    if isinstance(time, str):
        data, time = tools.pd2np(data, time)
    else:
        data = tools.conv2np(data, force_2d=True)

    colname = "high "
    if not high:
        data = data * -1
        colname = "low "

    peaks = exc.thresh_cross_del(data, crossingvalue)
    peaks[peaks < crossingvalue] = np.nan

    if not high:
        peaks = peaks * -1

    hl_water = pd.DataFrame({"time": time, colname + "waters": peaks[:, 0]})
    hl_water = hl_water.dropna()

    return hl_water


def tidal_phase(tide_time, hwlw_time, unit="h"):
    """
    Description
    ----------
    Gets the tidal phase relative to high/low water.

    Parameters
    ----------
    tide_time : array
        Times of the tide.
    hwlw_time : array
        Times of the high/low water.
    unit : str, optional
        Unit of the phase. Options are 'h' for hours, 'm' for minutes, 's' for seconds.
        Default is 'h'.
        np.timedelta64 options:
        Code    Meaning     Time span (relative) Time span (absolute)
        Y       year        +/- 9.2e18 years     [9.2e18 BC, 9.2e18 AD]
        M       month       +/- 7.6e17 years     [7.6e17 BC, 7.6e17 AD]
        W       week        +/- 1.7e17 years     [1.7e17 BC, 1.7e17 AD]
        D       day         +/- 2.5e16 years     [2.5e16 BC, 2.5e16 AD]
        h       hour        +/- 1.0e15 years     [1.0e15 BC, 1.0e15 AD]
        m       minute      +/- 1.7e13 years     [1.7e13 BC, 1.7e13 AD]
        s       second      +/- 2.9e11 years     [2.9e11 BC, 2.9e11 AD]
        ms      millisecond +/- 2.9e8 years      [2.9e8 BC, 2.9e8 AD]
        us / μs microsecond +/- 2.9e5 years      [290301 BC, 294241 AD]
        ns      nanosecond  +/- 292 years        [1678 AD, 2262 AD]
        ps      picosecond  +/- 106 days         [1969 AD, 1970 AD]
        fs      femtosecond +/- 2.6 hours        [1969 AD, 1970 AD]
        as      attosecond  +/- 9.2 seconds      [1969 AD, 1970 AD]

    Returns
    -------
    phase : array
        Phase relative to high/low water.
    """

    tide_time = tools.conv2np(tide_time, times=True)
    hwlw_time = tools.conv2np(hwlw_time, times=True)
    # Get closest indexes
    closest_indexes = tools.closest_vals(tide_time, hwlw_time, indexes=True)
    # Get time differences
    phase = tide_time - hwlw_time[closest_indexes]
    # Convert to specified unit
    phase = phase / np.timedelta64(1, unit)

    return phase


def skew_surge(water_levels, tide, sine_tide, time, window=6):

    water_levels = tools.conv2np(water_levels)
    tide = tools.conv2np(tide)
    sine_tide = tools.conv2np(sine_tide)
    time = tools.conv2np(time, times=True)

    high_water = high_low_water(
        sine_tide, time=time, crossingvalue=0, high=True
    )

    # check from here for type clashes etc

    # get the hw times
    hw_time = high_water["time"].values
    # get the time to hw in hours
    phase = tidal_phase(time, hw_time, unit="h")
    # index where the phase is within the window length / 2
    phase_index = np.where(np.abs(phase) < window / 2)[0]
    # group consecutive indices
    groups = [
        list(g)
        for _, g in itertools.groupby(
            phase_index, key=lambda x, c=itertools.count(): x - next(c)
        )
    ]
    # preallocate
    l = len(groups)
    v = [
        "skew",
        "time_2_tidal_hw",
        "tidal_hw",
        "tidal_hw_time",
        "wl_hw",
        "wl_hw_time",
        "nan_perc_in_window",
        "nhours_2_nearest_nan",
    ]
    s = {name: np.full(l, np.nan) for name in v}
    # iterate through each group
    for n, group_indices in groups:
        try:
            # tidal hw
            i = np.nanargmax(tide[group_indices])
            s["tidal_hw"][n] = tide[group_indices[i]]
            s["tidal_hw_time"][n] = time[group_indices[i]]
            # wl hw
            j = np.nanargmax(water_levels[group_indices])
            s["wl_hw"][n] = water_levels[group_indices[j]]
            s["wl_hw_time"][n] = time[group_indices[j]]
            # skew
            s["skew"][n] = s["wl_hw"][n] - s["tidal_hw"][n]
            s["time_2_tidal_hw"][n] = (
                s["wl_hw_time"][n] - s["tidal_hw_time"][n]
            ) / np.timedelta64(1, "h")
            # nan flags
            ni = np.where(np.isnan(water_levels[group_indices]))[0]
            s["nan_perc_in_window"][n] = (len(ni) / len(group_indices)) * 100
            s["nhours_2_nearest_nan"][n] = np.nanmin(
                (s["wl_hw_time"][group_indices[ni]] - s["wl_hw_time"][n])
                / np.timedelta64(1, "h")
            )
        except:
            [ar.__setitem__(n, np.nan) for ar in s]

    colnames = [
        "skew surge",
        "time to tidal high water from actual high water",
        "tidal high water",
        "time of tidal high water",
        "sea level high water",
        "time of sea level high water",
        "percentage of sea level nans in window",
        "hours to nearest nan from sea level high water",
    ]
    skew_surge = pd.DataFrame(s, columns=colnames)
    # get rid of the groups with no tidal or wl max (e.g., all nan data)
    skew_surge = skew_surge.dropna(how="all")

    return skew_surge

    # tide_time_numeric = tide_time.values.astype('float')
    # hwlw_time_numeric = hwlw_time.values.astype('float')

    # def phase_single(j):
    #     interp_func = interp1d(hwlw_time_numeric, hwlw_time.index, kind='nearest', fill_value='extrapolate')
    #     v = interp_func(tide_time_numeric[j])
    #     return (tide_time[j] - hwlw_time[v]) / np.timedelta64(1, unit)

    # tide_phase = np.full(len(tide_time), np.nan)

    # if parallel:
    #     with Pool() as pool:
    #         tide_phase = pool.map(phase_single, range(len(tide_time)))
    # else:
    #     for j in range(len(tide_time)):
    #         tide_phase[j] = phase_single(j)

    # return tide_phase


# def align_time_series(time1, data1, time2, data2):
#     # Convert time arrays to np.datetime64 if not already
#     if not isinstance(time1, np.ndarray):
#         time1 = np.array(time1, dtype='datetime64[m]')
#     if not isinstance(time2, np.ndarray):
#         time2 = np.array(time2, dtype='datetime64[m]')

#     # Create a combined time series spanning the entire range of both time arrays
#     start_time = min(time1.min(), time2.min())
#     end_time = max(time1.max(), time2.max())
#     combined_time = np.arange(start_time, end_time + np.timedelta64(1, 'm'), dtype='datetime64[m]')

#     # Initialize NaN-filled arrays for each time series
#     aligned_data1 = np.full(len(combined_time), np.nan)
#     aligned_data2 = np.full(len(combined_time), np.nan)

#     # Find indices for inserting data into the combined time series
#     indices1 = np.searchsorted(combined_time, time1)
#     indices2 = np.searchsorted(combined_time, time2)

#     # Fill the corresponding values from each original time series into the combined time series
#     aligned_data1[indices1] = data1
#     aligned_data2[indices2] = data2

#     return combined_time, aligned_data1, aligned_data2
