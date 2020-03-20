#@Time    :3/16/2020
#@Author  : Ruofei
import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.graph_objs as go
import plotly.offline as py
from config import *
import json
py.init_notebook_mode()


def plot_daily_overall_qty(tmall_inseason_style_sold):
    tmall_inseason_style_sold["year"] = tmall_inseason_style_sold["sales_date"].dt.year
    data2018 = tmall_inseason_style_sold.query("year==2018").groupby(["sales_date"])['gross_sales_qty'].sum()
    data2019 = tmall_inseason_style_sold.query("year==2019").groupby(["sales_date"])['gross_sales_qty'].sum()
    trace0 = go.Scatter(x=data2018.index,y=data2018.values,mode='lines+markers',name='lines+markers')
    trace1 = go.Scatter(x=data2019.index,y=data2019.values,mode='lines+markers',name='lines+markers')
    data = [trace0, trace1]
    py.plot(data)


def get_prophet_holiday_df(rm_days):
   rm = rm_days[rm_days["Event"].notnull()]
   rm["Event"] = rm["Event"].astype(str)
   holidays_df = pd.DataFrame()
   for rm_type in rm["Event"].unique():
       rm_i = rm.query("Event == @rm_type")
       df = pd.DataFrame({
           'holiday': rm_i.Event.iloc[0],
           'ds': rm_i.Date,
           'lower_window': rm_i.lower.iloc[0],
           'upper_window': rm_i.upper.iloc[0],
           'prior_scale': rm_i.prior_scale.iloc[0]
       })
       holidays_df = pd.concat([holidays_df, df])
   return holidays_df

class Prophet_Model:
   def __init__(self, ts_data, target_year, target_season, holidays_df=None):
       self.target_sn_begin = pd.to_datetime(f'{target_year}-{season_periods_dict[target_season][0]}')
       self.target_sn_end = pd.to_datetime(f'{target_year}-{season_periods_dict[target_season][1]}')
       self.train_period_end = self.target_sn_begin - datetime.timedelta(days=5)
       self.train_period_latest_begins = self.target_sn_begin - datetime.timedelta(days=365)
       self.train_ds = ts_data[ts_data["ds"] < self.train_period_end]
       self.test_ds = ts_data[(ts_data["ds"] >= self.train_period_end) & (ts_data["ds"] < self.target_sn_end)]
       self.holidays_df = holidays_df

   def build_prophet(self, **kwargs):
       # 0.3, 5, 8, 3, 2
       if self.holidays_df is None:
           m = Prophet(changepoint_prior_scale=kwargs["changepoint_prior_scale"])
       else:
           m = Prophet(holidays=self.holidays_df, changepoint_prior_scale=kwargs["changepoint_prior_scale"])
       m.add_seasonality(name="yearly", period=365, fourier_order=kwargs["yearly_order"])
       m.add_seasonality(name='quarterly', period=91.5, fourier_order=kwargs["quarterly_order"])
       m.add_seasonality(name='monthly', period=30.5, fourier_order=kwargs["monthly_order"])
       m.add_seasonality(name='weekly', period=7, fourier_order=kwargs["weekly_order"])
       m.add_country_holidays(country_name='CN')
       m.fit(self.train_ds)
       future = m.make_future_dataframe(periods=len(self.test_ds))  # concat future dates after history data
       self.forecast = m.predict(future)
       self.m = m

   def build_auto_prophet(self, **kwargs):

       if self.holidays_df is None:
           m = Prophet(changepoint_prior_scale=kwargs["changepoint_prior_scale"])
       else:
           m = Prophet(holidays=self.holidays_df, changepoint_prior_scale=kwargs["changepoint_prior_scale"])
       m.add_seasonality(name="yearly", period=365, fourier_order=kwargs["yearly_order"])
       m.add_country_holidays(country_name='CN')
       m.fit(self.train_ds)
       future = m.make_future_dataframe(periods=len(self.test_ds))  # concat future dates after history data
       self.forecast = m.predict(future)
       self.m = m

   def plot_result(self, folder, name):
       fig1 = self.m.plot(self.forecast)
       plt.scatter(self.test_ds["ds"].values, self.test_ds["y"], c='black', s=5)
       plt.title(name)
       if folder:
           paths.add_source("", f"{paths.prophet_model}/{folder}/plot_result")
           plt.savefig(f"{getattr(paths, folder)}/plot_result/result_{name}.png")
       else:
           plt.show()
       fig2 = self.m.plot_components(self.forecast)
       plt.title(name)
       if folder:
           plt.savefig(f"{getattr(paths, folder)}/plot_result/component_{name}.png")
       else:
           plt.show()

   def compare_result(self):
       compare_result = pd.merge(self.forecast, self.test_ds, on="ds")
       non_nan_data = compare_result[compare_result["y"].notnull()]
       mape = abs(non_nan_data["yhat"].sum() - non_nan_data["y"].sum()) / non_nan_data["y"].sum()
       return compare_result, [non_nan_data["yhat"].sum(), non_nan_data["y"].sum(), mape]

class Searcher:
    def __init__(self, parameters):
        self.names = parameters.keys()
        self.setups = [list(param) for param in parameters.values()]
        self.i = len(self.setups) - 1       # refer to which param, starting from the last param
        self.j = 0                          # refer to which value of param i
        self.best_mape = float("inf")
        self.best_params = None
        self.last_mape = 10000000000

    def run(self):
        current_setup = [s[0] for s in self.setups]
        # while self.best_mape > 0.1:
        while self.best_mape > 0.001:
            if self.i >= 0 and self.j >= len(self.setups[self.i]):                      # all values of this param have been tried
                self.j = 1
                self.i -= 1
            if self.i < 0:                                                               # no param could adjust
                break
            current_setup[self.i] = self.setups[self.i][self.j]
            params = dict(zip(self.names, current_setup))
            print(params, self.i, self.j)
            mape = self.func(**params)
            # prophet_search_model.build_prophet(**params)
            # _, sn_performance_result = prophet_search_model.compare_result()
            # mape = sn_performance_result[2]
            print(f"mape: {mape}")
            if mape < self.best_mape:
                self.best_mape = mape
                self.best_params = params
            # if mape > self.last_mape*1.2 and self.i > 0:        # leave changepoint full grid search
            if mape > self.last_mape*1.2:                     # cp_cut, consider all parameter is linear correlated with mape
                self.i -= 1
                self.j = 1
                continue
            self.last_mape = mape
            self.j += 1
        print(self.best_mape, self.best_params)

## Usage:
class MySearcher(Searcher):
    def __init__(self, prophet_model, parameters):
        super().__init__(parameters)
        self.model = prophet_model

    def func(self, **params):                   # minimize loss func
        self.model.build_prophet(**params)
        predict_result, sn_performance_result = self.model.compare_result()
        mape = sn_performance_result[2]
        return mape

class MySearcher_simple(Searcher):
    def __init__(self, prophet_model, parameters):
        super().__init__(parameters)
        self.model = prophet_model

    def func(self, **params):                   # minimize loss func
        self.model.build_auto_prophet(**params)
        predict_result, sn_performance_result = self.model.compare_result()
        mape = sn_performance_result[2]
        return mape


# def apply_prophet( tmall_inseason_style_sold):
#     # style_code = 880268 # cp:0.08 year:1 44 max/min 2803
#     style_code = 749571 # cp:1, yearly:1, 37, max/min 1492
#     style_ts = tmall_inseason_style_sold.query("style_cd == @style_code")
#     style_ds = style_ts[["sales_date", 'gross_sales_qty']]
#     style_ds.columns = ["ds", "y"]
#     style_ds["y"] = np.where(style_ds["ds"].isin(holidays_df["ds"].unique()), np.nan, style_ds["y"])
#     prophet_predict_model = Prophet_Model(style_ds, 2019, 'SU', holidays_df)
#     prophet_predict_model.train_ds["y"].describe()
#     m = Prophet(holidays=holidays_df, changepoint_prior_scale=0.08)
#     m.add_seasonality(name="yearly", period=365, fourier_order=1)
#     m.add_seasonality(name='quarterly', period=91.5, fourier_order=2)
#     m.add_seasonality(name='monthly', period=30.5, fourier_order=1)
#     m.add_seasonality(name='weekly', period=7, fourier_order=2)
#     m.add_country_holidays(country_name='CN')
#     m.fit(prophet_predict_model.train_ds)
#     future = m.make_future_dataframe(periods=len(prophet_predict_model.test_ds))  # concat future dates after history data
#     forecast = m.predict(future)
#     compare_result = pd.merge(forecast, prophet_predict_model.test_ds, on="ds")
#     non_nan_data = compare_result[compare_result["y"].notnull()]
#     mape = abs(non_nan_data["yhat"].sum() - non_nan_data["y"].sum()) / non_nan_data["y"].sum()
#     fig1 = m.plot(forecast)
#     plt.scatter(prophet_predict_model.test_ds["ds"].values, prophet_predict_model.test_ds["y"], c='black', s=5)
#     fig2 = m.plot_components(forecast)
#     plt.show()


def main(tmall_inseason_style_sold, include_rm=0, folder="prophet_sn_forecast"):
    """
    :param tmall_inseason_style_sold: style level sold daily data
    :param include_rm: if include retail moment days record then set 1, if remove the set 0 and the sales qty will be replace by nan
    :return: forecast total season(2019su) sold with prophet model, prophet params are auto searched according to Jan sold qty
    """
    parameters = {
        "changepoint_prior_scale": list(np.arange(0.05, 8, 0.5)),
        "weekly_order": list(np.arange(8, 0, -1)),
        "monthly_order": list(np.arange(8, 0, -1)),
        "quarterly_order": list(np.arange(8, 0, -1)),
        "yearly_order": list(np.arange(8, 0, -1))
    }   # for
    simple_parameters1 = {
        "yearly_order": list(np.arange(2, 0, -1)),
        "changepoint_prior_scale": list(np.arange(1, 0.01, -0.05))
    }       # for small sales qty at non-retail moments with smaller CV means need changepoint greater to balance the yearly trend, the yearly trend is not that obvious
    simple_parameters2 = {
        "yearly_order": list(np.arange(2, 0, -1)),
        "changepoint_prior_scale": list(np.arange(0.1, 0.01, -0.01))
    }       # for very smooth curve wit small sales qty at non-retail moments and large sales boost at retail moment
    folder_name = f"rm_{include_rm}_{folder}"
    paths.add_source(folder_name, f"{paths.prophet_model}/{folder_name}")
    styles_prophet_best_params = defaultdict(dict)
    result_df = pd.DataFrame()
    performance_list = []
    style_cd_list = tmall_inseason_style_sold["style_cd"].unique()
    for style_code in style_cd_list:
        style_ts = tmall_inseason_style_sold.query("style_cd == @style_code")
        style_ds = style_ts[["sales_date", 'gross_sales_qty']]
        style_ds.columns = ["ds", "y"]
        if include_rm != 1:
            style_ds["y"] = np.where(style_ds["ds"].isin(holidays_df["ds"].unique()), np.nan,
                                     style_ds["y"])  # remove abnormal high at retailmoments
        prophet_predict_model = Prophet_Model(style_ds, 2019, 'SU', holidays_df)
        if len(prophet_predict_model.test_ds) < 90 or style_ds[
            "ds"].min() > prophet_predict_model.train_period_latest_begins or len(prophet_predict_model.train_ds) < 365:
            continue
        # find appropriate params using previous season as test
        prophet_search_model = Prophet_Model(style_ds, 2019, 'Jan',
                                             holidays_df)  # search for the appropriate prophet model params for this style
        max_min = prophet_predict_model.train_ds["y"].describe()["max"] / \
                  prophet_predict_model.train_ds["y"].describe()["min"]
        if prophet_predict_model.train_ds["y"].describe()["50%"] < 55 or max_min > 1400:
            if prophet_predict_model.train_ds["y"].describe()["mean"] / prophet_predict_model.train_ds["y"].describe()["std"] >= 0.5 and max_min > 600:
                simple_parameters = simple_parameters1
            else:
                simple_parameters = simple_parameters2
            searcher = MySearcher_simple(prophet_search_model, simple_parameters)
            searcher.run()
            prophet_predict_model.build_auto_prophet(**searcher.best_params)
        else:
            searcher = MySearcher(prophet_search_model, parameters)
            searcher.run()
            prophet_predict_model.build_prophet(**searcher.best_params)

        predict_result, sn_performance_result = prophet_predict_model.compare_result()
        print(style_code, sn_performance_result)
        # save result
        prophet_predict_model.plot_result(folder_name, style_code)
        predict_result["style_code"] = style_code
        result_df = pd.concat([result_df, predict_result])
        sn_performance_result.append(style_code)
        performance_list.append(sn_performance_result)
        styles_prophet_best_params[style_code]["params"] = searcher.best_params
        styles_prophet_best_params[style_code]["mape"] = searcher.best_mape

    performance_df = pd.DataFrame(performance_list, columns=["yhat", "y", "mape", "style_code"])
    performance_df.to_excel(f"{getattr(paths, folder_name)}/performance_df.xlsx", index=False)
    result_df.to_excel(f"{getattr(paths, folder_name)}/prediction_data.xlsx", index=False)

    with open(f"{getattr(paths, folder_name)}/search_params.txt", "w") as f:
        parameters_float = {key: [x * 1.0 for x in value] for key, value in parameters.items()}
        json.dump(parameters_float, f)
    with open(f"{getattr(paths, folder_name)}/styles_params.txt", "w") as f:
        styles_prophet_best_params_f = styles_prophet_best_params.copy()
        for key, value in styles_prophet_best_params_f.items():
            value["params"] = {key2: value2 * 1.0 for key2, value2 in value["params"].items()}
        json.dump(styles_prophet_best_params_f, f)
    with open(f"{getattr(paths, folder_name)}/sim_search_params1.txt", "w") as f:
        simple_parameters1_float = {key: [x * 1.0 for x in value] for key, value in simple_parameters1.items()}
        json.dump(simple_parameters1_float, f)
    with open(f"{getattr(paths, folder_name)}/sim_search_params2.txt", "w") as f:
        simple_parameters2_float = {key: [x * 1.0 for x in value] for key, value in simple_parameters2.items()}
        json.dump(simple_parameters2_float, f)

    return performance_df, styles_prophet_best_params

import itertools

parameters = {
    "a": range(8),
    "b": range(10),
}

names = parameters.keys()
setups = itertools.product(*parameters.values())
for setup in setups:
    data = dict(zip(names, setup))
    print(data)


if __name__ == '__main__':
    paths.add_source("prophet_model","prophet_model")
    daily_sold = pd.read_excel(f'{paths.source}/{daily_sold_data}')
    rm_days = pd.read_excel(f"{paths.source}/{rm_daily_file_src}")      # retail moment facts
    # extract tmall inseason part
    tmall_inseason_sold = daily_sold.query("(platform == 'TMALL') & (inseason_flag1=='Y')")
    # aggregate sku to style level
    tmall_inseason_style_sold = tmall_inseason_sold.groupby(["platform", "style_cd", "sales_date"])['gross_sales_qty'].sum().reset_index().sort_values(by=["platform", "style_cd", "sales_date"])
    holidays_df = get_prophet_holiday_df(rm_days)
    folder = "multi_searcher_screen_max_min_50_cp_cut_3"
    performance_df, styles_prophet_best_params = main(tmall_inseason_style_sold, include_rm=0, folder=folder)






