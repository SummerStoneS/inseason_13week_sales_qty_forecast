import numpy as np

def curve_function_traffic(x, s0, t0, lambda1, lambda2, gamma, cr):
    """
        x: its the train data
            **traffic: average traffic of tmall whole site(this should be style level)
            **q0:maximum weekly selling rate if sold at base price P0 throughout the season (use buy plan instead)
            **sales_wk: number of week the style has been sold since its launch week
            **rm_days: number of retail moment days happend in that week
            **md: sold price / msrp
        params:
            **s0: effect of retail moment (but this should not equal to different retail moment I think)
            **t0: week that sales reach its peak
            **lambda1: trend-up period
            **lambda2: trend-down period
            **gamma: price sensitivity
            **cr: conversion rate of traffic
    """

    traffic, q0, sales_wk, rm_days, md = x

    RetailMoment = (1 + s0 * rm_days/7)
    MarketTrend = np.where(((sales_wk < t0) & (sales_wk >= (t0 - lambda1))), np.power((1 - np.power((t0 - sales_wk)/lambda1, 3)), 3),
                np.where(((sales_wk >= t0) & (sales_wk < (t0 + lambda2))), np.power((1 - np.power((sales_wk - t0)/lambda2, 3)),3), 0))
    PriceSensitivity = np.power(md, gamma)

    return (1 + np.log(traffic+1) * cr) * q0 * MarketTrend * RetailMoment * PriceSensitivity


def get_init_and_bounds_traffic(train_data):

    MAX_SOLD_WEEK = train_data["sales_wk"].max()
    ini_t0 = MAX_SOLD_WEEK / 2
    ini_lambda1, ini_lambda2 = ini_t0 , MAX_SOLD_WEEK / 2
    ini_s0 = 1.0
    ini_gamma = -1.0
    ini_cr = 0.0005
    init_params = [ini_s0, ini_t0, ini_lambda1, ini_lambda2, ini_gamma, ini_cr]
    bounds = ([0, 0, 0, 0, -np.inf, 0], [10, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 0, 0.1])
    return init_params, bounds


def curve_function_linear_md(x, s0, t0, lambda1, lambda2, gamma, cr):
    """
        x: its the train data
            **traffic: average traffic of tmall whole site(this should be style level)
            **q0:maximum weekly selling rate if sold at base price P0 throughout the season (use buy plan instead)
            **sales_wk: number of week the style has been sold since its launch week
            **rm_days: number of retail moment days happend in that week
            **md: sold price / msrp
        params:
            **s0: effect of retail moment (but this should not equal to different retail moment I think)
            **t0: week that sales reach its peak
            **lambda1: trend-up period
            **lambda2: trend-down period
            **gamma: price sensitivity
            **cr: conversion rate of traffic
            **st: sell through
    """

    traffic, q0, sales_wk, rm_days, md = x

    RetailMoment = (1 + s0 * rm_days/7)
    MarketTrend = np.where(((sales_wk < t0) & (sales_wk >= (t0 - lambda1))), np.power((1 - np.power((t0 - sales_wk)/lambda1, 3)), 3),
                np.where(((sales_wk >= t0) & (sales_wk < (t0 + lambda2))), np.power((1 - np.power((sales_wk - t0)/lambda2, 3)),3), 0))
    PriceSensitivity = (1 + gamma * (1-md))

    return (1 + np.log(traffic+1) * cr) * q0 * MarketTrend * RetailMoment * PriceSensitivity


def get_init_and_bounds_linear_md(train_data):

    MAX_SOLD_WEEK = train_data["sales_wk"].max()
    ini_t0 = MAX_SOLD_WEEK / 2
    ini_lambda1, ini_lambda2 = ini_t0 , MAX_SOLD_WEEK / 2
    ini_s0 = 1.0
    ini_gamma = 0.02
    ini_cr = 0.0005
    init_params = [ini_s0, ini_t0, ini_lambda1, ini_lambda2, ini_gamma, ini_cr]
    bounds = ([0, 0, 0, 0, 0, 0], [10, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 500, 0.1])
    return init_params, bounds

# use exp traffic and linear markdown
def curve_function_lin_md_exp_traffic(x, s0, t0, lambda1, lambda2, gamma, cr):
    """
        x: its the train data
            **traffic: average traffic of tmall whole site(this should be style level)
            **q0:maximum weekly selling rate if sold at base price P0 throughout the season (use buy plan instead)
            **sales_wk: number of week the style has been sold since its launch week
            **rm_days: number of retail moment days happend in that week
            **md: sold price / msrp
        params:
            **s0: effect of retail moment (but this should not equal to different retail moment I think)
            **t0: week that sales reach its peak
            **lambda1: trend-up period
            **lambda2: trend-down period
            **gamma: price sensitivity
            **cr: conversion rate of traffic
    """

    traffic, traffic0, q0, sales_wk, rm_days, md = x

    RetailMoment = (1 + s0 * rm_days/7)
    MarketTrend = np.where(((sales_wk < t0) & (sales_wk >= (t0 - lambda1))), np.power((1 - np.power((t0 - sales_wk)/lambda1, 3)), 3),
                np.where(((sales_wk >= t0) & (sales_wk < (t0 + lambda2))), np.power((1 - np.power((sales_wk - t0)/lambda2, 3)),3), 0))
    PriceSensitivity = (1 + gamma * (1-md))

    return np.exp(cr * (traffic / traffic0 - 1 )) * q0 * MarketTrend * RetailMoment * PriceSensitivity


def get_init_and_bounds_lin_md_exp_traffic(train_data):

    MAX_SOLD_WEEK = train_data["sales_wk"].max()
    ini_t0 = MAX_SOLD_WEEK / 2
    ini_lambda1, ini_lambda2 = ini_t0 , MAX_SOLD_WEEK / 2
    ini_s0 = 1.0
    ini_gamma = 0.02
    ini_cr = 0.0005
    init_params = [ini_s0, ini_t0, ini_lambda1, ini_lambda2, ini_gamma, ini_cr]
    bounds = ([0, 0, 0, 0, 0, 0], [10, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 500, 10])
    return init_params, bounds


# use exp traffic and exp markdown
def curve_function_exp_md_exp_traffic(x, s0, t0, lambda1, lambda2, gamma, cr):
    """
        x: its the train data
            **traffic: average traffic of tmall whole site(this should be style level)
            **q0:maximum weekly selling rate if sold at base price P0 throughout the season (use buy plan instead)
            **sales_wk: number of week the style has been sold since its launch week
            **rm_days: number of retail moment days happend in that week
            **md: sold price / msrp
        params:
            **s0: effect of retail moment (but this should not equal to different retail moment I think)
            **t0: week that sales reach its peak
            **lambda1: trend-up period
            **lambda2: trend-down period
            **gamma: price sensitivity
            **cr: conversion rate of traffic
    """

    traffic, traffic0, q0, sales_wk, rm_days, md = x

    RetailMoment = (1 + s0 * rm_days/7)
    MarketTrend = np.where(((sales_wk < t0) & (sales_wk >= (t0 - lambda1))), np.power((1 - np.power((t0 - sales_wk)/lambda1, 3)), 3),
                np.where(((sales_wk >= t0) & (sales_wk < (t0 + lambda2))), np.power((1 - np.power((sales_wk - t0)/lambda2, 3)),3), 0))
    PriceSensitivity = np.exp(gamma * (1-md))

    return np.exp(cr * (traffic / traffic0 - 1 )) * q0 * MarketTrend * RetailMoment * PriceSensitivity


def get_init_and_bounds_exp_md_exp_traffic(train_data):

    MAX_SOLD_WEEK = train_data["sales_wk"].max()
    ini_t0 = MAX_SOLD_WEEK / 2
    ini_lambda1, ini_lambda2 = ini_t0 , MAX_SOLD_WEEK / 2
    ini_s0 = 1.0
    ini_gamma = 0.02
    ini_cr = 0.0005
    init_params = [ini_s0, ini_t0, ini_lambda1, ini_lambda2, ini_gamma, ini_cr]
    bounds = ([0, 0, 0, 0, 0, 0], [20, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 50, 10])
    # bounds = ([0, 0, 0, 0, 0, 0], [20, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 50, 10])
    return init_params, bounds


def curve_function_lin_traffic_exp_md(x, s0, t0, lambda1, lambda2, gamma, cr):
    """
        x: its the train data
            **traffic: average traffic of tmall whole site(this should be style level)
            **q0:maximum weekly selling rate if sold at base price P0 throughout the season (use buy plan instead)
            **sales_wk: number of week the style has been sold since its launch week
            **rm_days: number of retail moment days happend in that week
            **md: sold price / msrp
        params:
            **s0: effect of retail moment (but this should not equal to different retail moment I think)
            **t0: week that sales reach its peak
            **lambda1: trend-up period
            **lambda2: trend-down period
            **gamma: price sensitivity
            **cr: conversion rate of traffic
    """

    traffic, traffic0, q0, sales_wk, rm_days, md = x

    RetailMoment = (1 + s0 * rm_days/7)
    MarketTrend = np.where(((sales_wk < t0) & (sales_wk >= (t0 - lambda1))), np.power((1 - np.power((t0 - sales_wk)/lambda1, 3)), 3),
                np.where(((sales_wk >= t0) & (sales_wk < (t0 + lambda2))), np.power((1 - np.power((sales_wk - t0)/lambda2, 3)),3), 0))
    PriceSensitivity =np.exp(gamma * (1-md))

    return cr * (traffic / traffic0) * q0 * MarketTrend * RetailMoment * PriceSensitivity


def get_init_and_bounds_lin_traffic_exp_md(train_data):

    MAX_SOLD_WEEK = train_data["sales_wk"].max()
    ini_t0 = MAX_SOLD_WEEK / 2
    ini_lambda1, ini_lambda2 = ini_t0 , MAX_SOLD_WEEK / 2
    ini_s0 = 1.0
    ini_gamma = -1.0
    ini_cr = 0.0005
    init_params = [ini_s0, ini_t0, ini_lambda1, ini_lambda2, ini_gamma, ini_cr]
    bounds = ([0, 0, 0, 0, -np.inf, 0], [10, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 50, 500])
    return init_params, bounds

