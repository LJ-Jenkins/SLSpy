"""
-sls-
Functions for statistical analysis and manipulation of data.

Functions
---------
- `remove_msl` -- remove mean sea level trends from sea level data.
- `bm_return_levels` -- calculates return levels for specified return periods from block maxima data.
- `pot_return_levels` -- calculates return levels for specified return periods from peaks over threshold data.
- `rl_plot` -- plot results from return level calculations.
- `bands_chisq` -- chi squared test for data split into bands (exp) and a sample (obs).
- `pot_decluster` -- decluster exceedances above given thresholds by given time windows.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import genextreme, gumbel_r, genpareto, chisquare, probplot
from . import tools
from . import exc

def remove_msl(data, time=None, trend='linear', perc_cover=75, degf=1, out_type='array'):
    """
    Description
    ----------
    Removes mean sea level trends from sea level data using linear or quadratic trend.
    Linear method used by mutliple studies e.g., https://doi.org/10.1007/s11069-022-05617-z

    Parameters
    ----------
    data : dataframe or array-like
        Either a dataframe of 'time' and 'water levels' or an array of the water level values.
    time : array-like
        Times of the water levels, only for use if data input is also an array.
    trend : str, 'linear' or 'quadratic'
        Specify trend.
    perc_cover : int
        What percentage of data for which to discount that year in the calculation.
    degf : int
        Degrees of freedom.
    out_type : str, 'df' or (default) 'array'
        Specify for either dataframe or array return.

    Returns
    -------
    data : dataframe or array
        Either a dataframe of 'time', 'water levels', and 'water levels detrended' if out_type='df',
        or an array of the detrended water levels if out_type='array'.
    """

    if time is None or isinstance(time, str):
        data = tools.checkinputdf(data, func='remove_msl')
    else:
        data = pd.DataFrame({'time': time,'water levels': data})

    # Extract year from the 'time' column and add it as a new column 'year'
    data['year'] = data['time'].dt.year

    # Group by year, calculate percentage of non-NaN observations, and mean value
    d = data.groupby('year')['water levels'].agg(
        perc_cover=lambda x: (x.notnull().sum() / len(x)) * 100,
        mean_value=lambda x: x.mean(skipna=True)
    )
    all_yrs = range(d.index.min(),d.index.max() + 1) # get all years, in case some missing
    d = d.reindex(all_yrs)

    # Make mean values nan if percentage cover was < perc_cover input
    d.loc[d['perc_cover'] < perc_cover, 'mean_value'] = pd.NA 

    def lqtrend(d, trend, degf):
        def dmatrix(x, trend):
            return ((np.column_stack((x**2, x, np.ones_like(x))), 3, 2)
                    if trend == 'quadratic'
                    else (np.column_stack((x, np.ones_like(x))),2, 1))
        dn = d.dropna(subset='mean_value') # drop nan rows
        x = dn.index.to_numpy()
        y = dn['mean_value'].to_numpy()
        # Create the design matrix X
        X, j, v = dmatrix(x, trend)
        # Compute coefficients using OLS
        m = np.linalg.lstsq(X, y, rcond=None)[0]
        # Compute responses at each data point and residuals
        yfit = np.dot(X, m)
        r = y - yfit
        # Calculate RMSE and degrees of freedom
        n = len(x)
        nu = n - j
        rmse = np.sqrt(np.sum(r**2)) / np.sqrt(nu)
        # Compute the standard errors
        Xi = np.linalg.inv(np.dot(X.T, X))
        se = rmse * np.sqrt(np.diag(np.abs(Xi))) # account for negatives with abs()
        # Residual degrees of freedom
        nu2 = np.ceil((n - j) * degf)
        # Root mean square error
        rmse2 = np.sqrt(np.sum(r**2)) / np.sqrt(nu2)
        # Standard error with autocorrelation
        se2 = rmse2 * np.sqrt(np.diag(np.abs(Xi))) # account for negatives with abs()
        # Extract results
        T = np.array([m[0]*v, se[0]*v, se2[0]*v])
        if trend == 'linear':
            X2 = np.column_stack((d.index.values, np.ones_like(d.index.values)))
            yfit = np.dot(X2, m)
        
        return yfit, T

    yf = lqtrend(d,trend,degf)[0]
    b = yf - yf[-1]
    yr_dns = np.array([pd.Timestamp(year, 7, 1).toordinal() + 366 for year in d.index.values])
    time_dns = np.array([pd.Timestamp(t).toordinal() + 366 for t in time])
    msl_func = interp1d(yr_dns, b, kind='linear', fill_value='extrapolate')
    msl = msl_func(time_dns)
    data['water levels detrended'] = data['water levels'] - msl
    data.drop(columns='year', inplace=True)
    if out_type == 'array':
        data = np.array(data['water levels detrended'])

    return data

def bm_return_levels(data, return_periods, dist, plot='n'):
    """
    Description
    ----------
    Calculates return levels for specified return periods by fitting a generalised extreme value
    distribution or a Gumbel distribution to block maxima.

    Parameters
    ----------
    data : array-like
        Data for the calculation (block maxima, e.g., annual detrended water level maxima).
    return_periods : array-like
        Return periods to be used e.g., [1, 5, 10].
    dist : str, 'gev' or 'gum'
        Specify a GEV or GUM distribution fit.
    plot : str, 'y' or 'n'
        Plot a 4 tile plot of a) Probability, b) Quantiles, c) Return periods, and d) Density using
        Weibull plotting positions.

    Returns
    -------
    return_levels : dataframe
        Return periods and their corresponding return levels.
    """
    
    data = tools.conv2np(data, nans='remove')
    return_periods = tools.conv2np(return_periods)

    # Fit the distribution
    # ppf: percent point function (inverse cumulative distribution function) of the GEV distribution.
    # isf: inverse survival function for the GEV distribution.
    # ppf deals with the direct cumulative distribution function (CDF), providing 
    # values based on the specified probability, whereas isf deals with the survival 
    # function, providing values based on the complement of the CDF.
    # isf matches best with Matlab's gev and ev functions.
    fit_functions = {'gev': [genextreme.fit, genextreme.isf],
                    'gum': [gumbel_r.fit, gumbel_r.isf]}

    params = fit_functions[dist][0](data)
    f = 1 - np.exp(-1 / return_periods)
    return_levels = fit_functions[dist][1](f, *params)

    if plot == 'y':
        rl_plot(data, return_periods, return_levels, dist, params)

    return pd.DataFrame({'Return periods': return_periods, 'Return levels': return_levels})

def pot_return_levels(data, return_periods, threshold, avgpy, plot='n'):
    """
    Description
    ----------
    Calculates return levels for specified return periods by fitting a generalised pareto
    distribution to peaks over threshold data. Set up for average number of exceedances per year.

    Parameters
    ----------
    data : array-like
        Data for the calculation (block maxima, e.g., annual detrended water level maxima).
    return_periods : array-like
        Return periods to be used e.g., [1, 5, 10].
    threshold : int
        Threshold value.
    avgpy : int
        Average number of exceedances per year.
    plot : str, 'y' or 'n'
        Plot a 4 tile plot of a) Probability, b) Quantiles, c) Return periods, and d) Density using
        Weibull plotting positions.

    Returns
    -------
    return_levels : dataframe
        Return periods and their corresponding return levels.
    """

    data = tools.conv2np(data, nans='remove')
    return_periods = tools.conv2np(return_periods)

    params = genpareto.fit(data, floc=threshold)
    f = 1 - (1 / (return_periods * avgpy))
    # ppf matches best with Matlab's gp functions.
    return_levels = genpareto.ppf(f, *params)

    if plot == 'y':
        rl_plot(data, return_periods, return_levels, 'gp', params, avgpy)

    return pd.DataFrame({'Return periods': return_periods, 'Return levels': return_levels})

# add threshold to params for gpd
def rl_plot(data, return_periods, return_levels, dist, params, avgpy=None):
    # Weibull plotting position, Coles (2001) page 43
    fit_functions = {'gev': [genextreme.isf, genextreme.cdf],
                'gum': [gumbel_r.isf, gumbel_r.cdf],
                'gp': [genpareto.ppf, genpareto.cdf]}
    # Empirical model
    P_em = np.arange(1,len(data)) / (len(data) + 1)
    # Calculate inverse EV/GEV/GP for empirical model
    zp_em = fit_functions[dist][0](P_em, *params)
    PPy = np.sort(data)
    P_data = fit_functions[dist][1](PPy, *params)

    # Store plotting positions
    if dist == 'gp':
        PPx = 1 / ((1 - P_em) * avgpy)
    else:
        PPx  = -1 / np.log(P_em)
    
def bands_chisq(population, sample, nbands, input_type='percentage'):
    """
    Description
    ----------
    Chi squared goodness of fit test for population partitioned into nbands and a sample relative to those bands.

    Parameters
    ----------
    population : array-like
        Population of data.
    sample : array-like
        Sample of data.
    nbands : int
        Number of bands.
    input_type : str, 'percentage' or 'proportion'
        Specify what the input to the chi squared function should be.

    Returns
    -------
    x2 : float
        Chi squared result.
    pval : float
        P value result.
    """

    # Remove nans
    population = tools.conv2np(population, nans='remove')
    sample = tools.conv2np(sample, nans='remove')
    # Calculate percentiles to split the data into bands
    percentiles = np.linspace(0, 100, nbands + 1)
    band_boundaries = np.percentile(population, percentiles)
    # Split the data into bands based on the calculated boundaries
    pop_bands = [population[(population >= band_boundaries[i]) & (population <= band_boundaries[i + 1])] for i in range(nbands)]
    # Calculate the counts of pop and sample values in each population band
    pop = np.array([len(band) for band in pop_bands], dtype=float) / len(population)
    samp = np.array([np.sum((sample >= band.min()) & (sample < band.max())) for band in pop_bands], dtype=float) / len(sample)
    if input_type == 'percentage':
        pop *= 100
        samp *= 100
    # Chisq
    x2, pval = chisquare(f_obs=samp, f_exp=pop)

    return x2, pval

def pot_decluster(data, time, threshold, window, window_units='hours', include_thresh=False, time_between=True, thresh_cross_del=False, colnames=False):
    """
    Description
    ----------
    Decluster exceedances above given thresholds by given time window.

    Parameters
    ----------
    data : pandas DataFrame
        Either a dataframe containing the data with a column for timestamps,
        or an array of just the data (no times).
    time : array of times or str, optional
        Either the column name of the times within a dataframe (when dataframe inputted),
        or an array of times (when array of data inputted).
    threshold : float, int, array or list
        The threshold value/s for exceedances. If multiple columns in data array/dataframe, 
        then multiple thresholds should be given.
    window : float, int
        The time window used to identify clusters around exceedances.
    window_units : str, optional
        The units for the time window ('hours', 'minutes', 'days').
    include_thresh : bool, optional
        Boolean flag to include values equal to the threshold.
        If True, considers values greater than or equal to the threshold.
        If False, considers values strictly greater than the threshold.
    time_between : bool, optional
        Boolean flag to calculate the time between exceedances (in hours).
    thresh_cross_del : bool, optional
        Boolean flag to delete all but the maximum value in groups of consecutive threshold crossings.
    colnames : string, optional
        Column names to either override the automatic naming of columns from an input dataframe or to 
        give names to array inputs. If an array is given and colnames are not, each column will be 'Var 1', 'Var 2', etc.
        
    Returns
    -------
    pot : dict or pandas DataFrame
        A dictionary where keys are column names and values are DataFrames containing declustered 
        exceedances and their times. If only one column is considered, returns a single DataFrame.
    """
    
    if isinstance(time, str):
        data, time, colinds = tools.pd2np(data, time)
    else:
        colinds = False
    
    if not colinds and not colnames:
        colnames = ['Var ' + str(i) for i in range(1, tools.ncols(data) + 1)]
    if colinds and not colnames:
        colnames = [key for key in colinds.keys()]

    threshold = tools.conv2list(threshold)

    if window_units == 'days':
        window *= 24
    elif window_units == 'minutes':
        window /= 60
    
    pot = {}
    for index in range(0, tools.ncols(data)):
        if thresh_cross_del:
            data[:, index] = exc.thresh_cross_del(data[:, index], threshold[index])
        if include_thresh:
            roverd = data[data[:, index] >= threshold[index], index]
            rovert = time[data[:, index] >= threshold[index]]
        else:
            roverd = data[data[:, index] > threshold[index], index]
            rovert = time[data[:, index] > threshold[index]]
        i = np.argsort(roverd) # descending order indexes
        i = np.flip(i) # flip them to ascending order
        roverd = roverd[i]
        rovert = rovert[i]
        for i in range(len(roverd)):
            j = (rovert >= rovert[i]- pd.Timedelta(hours=window/2)) & \
                (rovert <= rovert[i] + pd.Timedelta(hours=window/2))
            j[i] = False
            roverd = roverd[~j]
            rovert = rovert[~j]
            if i == len(roverd) - 1:
                break
        i = np.argsort(rovert) # sort by time 
        roverd = roverd[i]
        rovert = rovert[i]
        ro = pd.DataFrame({'time': rovert, colnames[index]: roverd})
        if time_between:
            ro['time between'] = ro['time'].diff().dt.total_seconds() / 3600
        pot[colnames[index]] = ro
    
    if len(pot) == 1:
        pot = pot[colnames[index]]
        
    return pot
