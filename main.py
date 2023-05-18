from methods import *


def linar_interpolate(timeseries, col_id: str, learn_len: int, max_lags: int, max_linear: int, max_linar: int, sig_adf: float, sig_ft: float, num_of_diffs: int):
    """
    :param timeseries: the time series (dataframe) to be interpolated.
    :param col_id: the name (string) of the column, which contains the time series to be interpolated.
    :param learn_len: the length of the autoregression train data length.
    :param max_lags: The maximum number of autoregressive lags included in the model.
    :param max_linear: The maximum gap size, which should be linearly interpolated.
    :param max_linar: The maximum gap size, which should be interpolated with the LinAR method.
    :param sig_adf: the significance level for the ADF test.
    :param sig_ft: the significance level for the Ftest.
    :param num_of_diffs: Number of differencings allowed in the while loop.
    :return: The interpolated time series.
    """
    tseries = convert_to_series(timeseries, col_id)
    tseries.index = add_freq(tseries.index)
    stp = tseries.index.freq
    nan_groups = group_nans(tseries, stp)
    itpd = tseries.copy()
    linars, linears = 0, 0

    for ng in nan_groups:
        if len(ng) > max_linear:
            continue
        elif len(ng) > max_linar:
            interpolate_linear(itpd, ng, stp)
            linears += 1
            continue
        data_cut = itpd[ng[0] - learn_len * stp:ng[0]-stp]
        if len(data_cut) != learn_len or data_cut.isnull().values.any():
            interpolate_linear(itpd, ng, stp)
            linears += 1
            continue
        data_cut_diffs = get_stationary_data(data_cut, stp, sig_adf, sig_ft, num_of_diffs)
        if not data_cut_diffs:
            interpolate_linear(itpd, ng, stp)
            linears += 1
            continue

        ar_model = create_model(data_cut_diffs[-1], max_lags)
        pred = ar_model.predict(start=len(data_cut_diffs[-1]), end=len(data_cut_diffs[-1]) + len(ng) - 1)
        data_cut_undf = get_undifferenced_data(data_cut_diffs, pred, stp)
        final_prog = adjust_to_next_obs(itpd, ng, data_cut_undf, stp)
        update_timeseries(itpd, final_prog)
        linars += 1
        print(f'LinAR interpolation for gap {ng[0]}: {ng[-1]}')
    print(linars, linears)
    return itpd
