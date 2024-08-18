"""
-sls-
Functions for descriptive statistics for data blocks.

Functions
---------
- `period_minmax` -- calculate the min, max, or both within a specified time period.
- `n_exc_per_period` -- count the number of exceedances, or average number per given period.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import pandas as pd
from . import tools


def period_minmax(data, time="time", period="Y", min=False, max=True):
    """
    Description
    ----------
    Calculate the min, max, or both within a specified time period.

    Parameters
    ----------
    data : dataframe, array
        Dataframe with a column of times (not an index) or an array.
    time : str, array
        Column name for the times (str) or an array of times if data is also an array.
    period : str
        Time period string (see additional information, default 'Y').
    min : boolean
        Specify whether to calculate the minimum.
    max : boolean
        Specify whether to calculate the maximum.

    Returns
    -------
    data : dataframe
        Dataframe of the minima and/or maxima.

    Additional Information
    -------
    Pandas timeseries time period options. See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-period-aliases
    for full list.

    Alias   Description
    B       business day frequency
    D       calendar day frequency
    W       weekly frequency
    M       monthly frequency
    Q       quarterly frequency
    Y       yearly frequency
    h       hourly frequency
    min     minutely frequency
    s       secondly frequency
    ms      milliseconds
    us      microseconds
    ns      nanoseconds
    W-SUN   weekly frequency (Sundays). Same as 'W'
    W-MON   weekly frequency (Mondays) (TUE, WED, THU, FRI, SAT also available)
    (B)Q(E)(S)-DEC quarterly frequency, year ends in December. Same as 'QE'
    (B)Q(E)(S)-JAN quarterly frequency, year ends in January (FEB, MAR, APR, etc.)
    (B)Y(E)(S)-DEC annual frequency, anchored end of December. Same as 'YE'
    (B)Y(E)(S)-JAN annual frequency, anchored end of January (FEB, MAR, APR, etc.)
    """

    if not (min or max):
        raise ValueError("At least one of 'min' or 'max' must be True.")

    if isinstance(time, str):
        non_time_columns = data.columns[data.columns != time]
    else:
        data, time, colinds = tools.np2pd(data, time)
        non_time_columns = [str(key) for key in colinds.keys()]

    pmm = pd.DataFrame(
        {"Time period": data[time].dt.to_period(period).unique()}
    )

    for col in non_time_columns:
        if max:
            max_values = data.groupby(data[time].dt.to_period(period))[
                col
            ].max()
            time_of_max = data.groupby(data[time].dt.to_period(period))[
                col
            ].idxmax()
            pmm[col + " max"] = max_values.values
            pmm[col + " time of max"] = data.loc[time_of_max, time].values
        if min:
            min_values = data.groupby(data[time].dt.to_period(period))[
                col
            ].min()
            time_of_min = data.groupby(data[time].dt.to_period(period))[
                col
            ].idxmin()
            pmm[col + " min"] = min_values.values
            pmm[col + " time of min"] = data.loc[time_of_min, time].values

    return pmm


def n_exc_per_period(
    data,
    time="time",
    threshold=None,
    period="Y",
    include_thresh=False,
    avg=False,
):
    """
    Description
    ----------
    Calculate the number, or average number, of exceedances per time period based on a given threshold.

    Parameters
    ----------
    data : pandas DataFrame or array
        DataFrame containing the data with a column for timestamps or an array.
    time : str, array
        Column name for the times (str) or an array of times if data is also an array.
    threshold : float, int, array or list
        The threshold value/s for exceedances, if multiple columns in data then
        multiple thresholds should be givem.
    period : str
        Frequency for time aggregation ('Y' for years, 'M' for months, 'D' for days, see info in above func).
    include_thresh : bool, optional
        Boolean flag to include values equal to the threshold.
        If True, considers values greater than or equal to the threshold.
        If False, considers values strictly greater than the threshold.
    avg : bool, optional
        Boolean flag to calculate the average number of exceedances per time period.
        If given, averages will be given instead of counts.

    Returns
    -------
    n_exc : float or pandas Series
        The number, or average number of exceedances per time period.
    """

    if len(threshold) == 1 and not threshold:
        raise ValueError("Threshold/s must be given.")
    threshold = tools.conv2list(threshold)

    if isinstance(time, str):
        non_time = data.columns[data.columns != time]
    else:
        data, time, colinds = tools.np2pd(data, time)
        non_time = [str(key) for key in colinds.keys()]

    # Convert timestamps to the specified frequency
    data["period"] = data[time].dt.to_period(period)
    all_periods = data["period"].unique()

    # Filter exceedances based on the threshold condition
    # Group by period
    # Count the number of exceedances
    exc_list = [None] * len(non_time)
    if include_thresh:
        for i, col in enumerate(non_time):
            exc_list[i] = (
                data.loc[data[col] >= threshold[i], [col, "period"]]
                .groupby("period")
                .size()
                .reset_index(name=col + " exc count")
            )
    else:
        for i, col in enumerate(non_time):
            exc_list[i] = (
                data.loc[data[col] > threshold[i], [col, "period"]]
                .groupby("period")
                .size()
                .reset_index(name=col + " exc count")
            )

    # Get all the periods
    n_exc = pd.DataFrame({"period": all_periods})
    for df in exc_list:
        n_exc = pd.merge(n_exc, df, on="period", how="left")

    if avg:
        # Calculate the average number of exceedances per time period
        n_exc = n_exc.iloc[:, 1:].mean()
        n_exc = n_exc.rename(lambda x: x + " avg")

    return n_exc
