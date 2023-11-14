import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import adfuller
import scipy.stats
from typing import Union


def convert_to_series(timeseries: Union[pd.Series, pd.DataFrame], col_id: str = None):
    """
    :param timeseries: the timeseries to be interpolated
    :param col_id: the column id string, should be provided if timeseries is a pd.DataFrame
    :return: timeseries saved as a pandas series
    """
    if isinstance(timeseries, pd.Series):
        return timeseries
    if isinstance(timeseries, pd.DataFrame):
        return timeseries[col_id]


def resample_timeseries(ts):
    """ Reads the most frequent time step and resamples the dataframe.
    :param ts: timeseries to be resampled.
    :return: resampled timeseries
    """
    typical_freq = pd.to_timedelta(np.diff(ts.index)).value_counts().index[0]
    return ts.resample(typical_freq).fillna(method=None)


def group_nans(timeseries: pd.Series, step: timedelta):
    """
    :param timeseries:  the timeseries to be interpolated
    :param step: index step of the data
    :nan_li: list of nan indexes
    :gaps_li: list of gaps to be returned from this function
    :curr_li: list of nan indexes within multi-step gaps, not used in case of 1-step gaps.
    :return: a list of lists with gaps. Each list contains all indexes within the gap.
    """
    nan_li = timeseries.loc[pd.isna(timeseries)].index.to_list()
    if len(nan_li) == 0:
        return []
    elif len(nan_li) == 1:
        return [nan_li]
    gaps_li, curr_li = list(), list()
    for num, nan_idx in enumerate(nan_li):
        if num == 0 and nan_li[1] - step != nan_idx:  # first nan, 1-step gap
            gaps_li.append([nan_idx])
        elif nan_idx == nan_li[-1] and nan_li[num - 1] + step != nan_idx:  # last nan, 1-step gap
            gaps_li.append([nan_idx])
        elif nan_idx != nan_li[-1] and nan_li[num + 1] - step == nan_idx:  # nan with continuation (multi-step gap)
            curr_li.append(nan_idx)
            curr_li.append(nan_li[num + 1])  # append next observation to the curr_li
        elif nan_idx != nan_li[-1] and nan_li[num - 1] + step != nan_idx != nan_li[num + 1] - step:  # nan inside, 1-step gap
            gaps_li.append([nan_idx])
        else:  # last nan in multi-step gap
            curr_li = sorted(list(set(curr_li)))  # delete overlapping indexes originating from adding the next observation to curr_li
            gaps_li.append(curr_li)
            curr_li = []
    return gaps_li


def fill_data(df: pd.Series, data_to_append: pd.Series):
    """
    :param df: series to which data should be appended
    :param data_to_append: series with data to append
    :return: dataframe with appended data
    """
    appended = pd.concat([df, data_to_append])
    return appended[~appended.index.duplicated()]


def update_timeseries(df: pd.Series, data_to_append: pd.Series):
    """
    :param df: series to which data should be appended
    :param data_to_append: series with data to append
    """
    for index, row in data_to_append.items():
        df.at[index] = row


def difference(data_to_diff: pd.Series, step: timedelta):
    """
    :param data_to_diff: time series to be differenced
    :param step: index step of the data
    :return: differenced time series
    """
    diff = data_to_diff[1:].copy()
    index_range = data_to_diff.index[1:]
    for idx in index_range:
        val = data_to_diff.loc[idx] - data_to_diff.loc[idx - step]
        diff.at[idx] = val
    return diff


def f_test(timeseries: pd.Series):
    """
    This function performs the ftset, which aims to assess data stationarity by comparing variances of the first and
    second half of the time series.

    :param timeseries: time series on which the ftest will be applied
    :return: p value of the ftest.
    """
    timeseries = timeseries.to_list()
    split = round(len(timeseries) / 2)
    x1, x2 = timeseries[0:split], timeseries[split:]
    if len(x1) > len(x2):
        x1 = timeseries[1:split]
    x, y = np.array(x1), np.array(x2)
    var1, var2 = sorted([np.var(x1, ddof=1), np.var(x2, ddof=1)])
    f = var2 / var1
    nun = x.size - 1
    dun = y.size - 1
    p_value = 1 - scipy.stats.f.cdf(f, nun, dun)
    return p_value


def get_stationary_data(timeseries: pd.Series, step: timedelta, sig_adf: float, sig_ft: float, num_of_diffs: int):
    """
    This function aims to obtain stationary data by differencing it.
    The stationarity is tested with the use of the Augmented Dickey Fuller (ADF) test and Ftest.
    The differencing is performed in a while loop until stationarity is obtained,
    or the number of differencings reaches the threshold defined by the num_of_diffs parameter.

    :param timeseries: time series, the stationarity of which should be obtained.
    :param step: index step of the data
    :param sig_adf: the significance level for the ADF test.
    :param sig_ft: the significance level for the Ftest.
    :param num_of_diffs: Number of differencings allowed in the while loop.
    :return: a list of at least two pd.Series: the undifferenced data and the 1st order differenced data.
             In the case of multiple differencing, successive time series are added to the list.
    """
    diff, diffs_list = difference(timeseries, step), [timeseries]
    diffs_list.append(diff)
    p_val_adf = adfuller(diff)[1]
    p_val_ftest = f_test(diff)
    while p_val_adf > sig_adf or p_val_ftest < sig_ft:
        diff = difference(diff, step)
        diffs_list.append(diff)
        if len(diffs_list) > num_of_diffs+1:
            return None
        p_val_adf = adfuller(diff)[1]
        p_val_ftest = f_test(diff)
    return diffs_list


def get_trend_and_breakpoints(timeseries: pd.Series, nan_group: list, step: timedelta):
    """
    :param timeseries: the time series on the basis of which the trend is to be calculated.
    :param nan_group: the gap to be interpolated.
    :param step: index step of the data.
    :return: the time series trend (linearly interpolated), the length of the gap, and the indexes and values
             before and after the gap.
    """
    first_time, last_time = nan_group[0] - step, nan_group[-1] + step
    first_val, last_val = timeseries[first_time], timeseries[last_time]
    gap_length = len(nan_group)
    delta_to_next_meas = (last_val - first_val) / (gap_length+1)
    delta = ((last_val - delta_to_next_meas) - first_val) / gap_length
    timeseries_trend = pd.Series([first_val + a * delta for a in range(1, gap_length + 1)])
    return timeseries_trend, gap_length, first_time, last_time, first_val, last_val


def create_model(timeseries: list, maxlags: int, criteria: str = 'aic'):
    """
    :param timeseries: the time series on which the model is to be built up.
    :param maxlags: The maximum number of autoregressive lags included in the model.
    :param criteria: The information criterion to use in the selection, options: {‘aic’, ‘hqic’, ‘bic’}.
    :return: the fitted autoregressive model, which can be used to predict observations.
    """
    lgs = ar_select_order(timeseries, maxlag=maxlags, ic=criteria)
    return AutoReg(timeseries, lags=lgs.ar_lags).fit()


def undiff(first, diff, step):
    """
    :param first: first value from which to start undifferencing.
    :param diff: time series to be undifferenced.
    :param step: index step of the data.
    :return: undifferenced time series.
    """
    new = first.copy()
    for i, val in diff.items():
        if i != first.index:
            new.at[i] = val + new.loc[i - step]
    return new


def get_undifferenced_data(differenced_series: list, pr: pd.Series, step: timedelta):
    """
    Performs the undifferencing of a time series. The order of differences is inferred from the length of the first
    parameter, which is a list with: the original data (first elem.) and time series with increasing differencing order.
    :param differenced_series: list including the original data, and the differenced series.
    :param pr: prediction to be undifferenced.
    :param step: index step of the data.
    :return: the undifferenced prediction.
    """
    differenced_series = differenced_series[::-1]
    und1 = undiff(differenced_series[1].tail(1), pr, step)
    for x in range(2, len(differenced_series)):
        undx = undiff(differenced_series[x].tail(1), und1, step)
        und1 = undx
    return und1.iloc[1:]


def adjust_to_next_obs(timeseries: pd.Series, nan_group: list, undf_preds: pd.Series, step: timedelta):
    """
    :param timeseries: the time series with measurements.
    :param nan_group: the gap (list of indexes with nan value) to be interpolated.
    :param undf_preds: the undifferenced AR prediction.
    :param step: index step of the data.
    :return: LinAR interpolation: a pd.Series with the AR prediction adjusted to the linear trend.
    """
    timeseries_trend, gap_length, first_time, last_time, first, last = get_trend_and_breakpoints(timeseries, nan_group, step)
    delta_pred = (undf_preds[-1] - first) / gap_length
    timeseries_trend_pred = pd.Series([first + a*delta_pred for a in range(1, gap_length+1)])
    subtracted_trends = timeseries_trend - timeseries_trend_pred

    subtracted_trends.index = pd.date_range(nan_group[0], nan_group[-1], freq=step)
    timeseries_trend.index = pd.date_range(nan_group[0], nan_group[-1], freq=step)
    timeseries_trend_pred.index = pd.date_range(nan_group[0], nan_group[-1], freq=step)

    corrected_preds = undf_preds + subtracted_trends
    return corrected_preds


def interpolate_linear(timeseries: pd.Series, nan_group: list, step: timedelta):
    """
    :param timeseries: time series to be interpolated.
    :param nan_group: the gap (list of indexes with nan value) to be interpolated.
    :param step: index step of the data.
    """
    ts_trend = get_trend_and_breakpoints(timeseries, nan_group, step)[0]
    ts_trend.index = pd.date_range(nan_group[0], nan_group[-1], freq=step)
    update_timeseries(timeseries, ts_trend)


def interpolate_linar(timeseries, col_id: str, learn_len: int, max_lags: int, max_linear: int, max_linar: int, sig_adf: float, sig_ft: float, num_of_diffs: int):
    """
    :param timeseries: the time series (dataframe) to be interpolated.
    :param col_id: the name (string) of the column, which contains the time series to be interpolated.
    :param learn_len: the length of the autoregression train data length.
    :param max_lags: The maximum number of autoregressive lags included in the model.
    :param max_linear: The maximum gap size, which should be linearly interpolated.
    :param max_linar: The maximum gap size, which should be interpolated with the LinAR method.
    :param sig_adf: the significance level for the ADF test.
    :param sig_ft: the significance level for the F test.
    :param num_of_diffs: Number of differencings allowed in the while loop.
    :return: The interpolated time series.
    """
    tseries = convert_to_series(timeseries, col_id)
    tseries = resample_timeseries(tseries)
    stp = tseries.index.freq
    nan_groups = group_nans(tseries, stp)
    itpd = tseries.copy()

    for ng in nan_groups:
        if ng[0] == itpd.index[0] or ng[-1] == itpd.index[-1]:
            continue
        if len(ng) > max_linear:
            continue
        elif len(ng) > max_linar:
            interpolate_linear(itpd, ng, stp)
            continue
        data_cut = itpd[ng[0] - learn_len * stp:ng[0]-stp]
        if len(data_cut) != learn_len or data_cut.isnull().values.any():
            interpolate_linear(itpd, ng, stp)
            continue
        data_cut_diffs = get_stationary_data(data_cut, stp, sig_adf, sig_ft, num_of_diffs)
        if not data_cut_diffs:
            interpolate_linear(itpd, ng, stp)
            continue

        ar_model = create_model(data_cut_diffs[-1], max_lags)
        pred = ar_model.predict(start=len(data_cut_diffs[-1]), end=len(data_cut_diffs[-1]) + len(ng) - 1)
        data_cut_undf = get_undifferenced_data(data_cut_diffs, pred, stp)
        final_prog = adjust_to_next_obs(itpd, ng, data_cut_undf, stp)
        update_timeseries(itpd, final_prog)
    return itpd
