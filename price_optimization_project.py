import os
from glob import glob
import pandas as pd
import datetime
import json
import numpy as np
from scipy.optimize import curve_fit
from config import *
from curve_functions import *

path_dict = {"source":"source_data","inseason_analysis":"inseason_analysis"}

class PathManager:
	def __init__(self, base_url,path_dict):
		self.base_url = base_url
		for name, value in path_dict.items():
			setattr(self, name, os.path.join(self.base_url, value))
			if not os.path.exists(os.path.join(self.base_url, value)):
				os.makedirs(os.path.join(self.base_url, value))

	def add_source(self,name,path):
		setattr(self,name,os.path.join(self.base_url, path))
		if not os.path.exists(os.path.join(self.base_url, path)):
			os.makedirs(os.path.join(self.base_url, path))

class DataRefactor:
	
	def __init__(self):
		self.input_data = None
		self.col_dict = {
		"prod_code":["STYLCOLOR_CD","sku_code","SKU","SKU_CODE","sku"],
		"category":['GBL_CAT_SUM_DESC',"CATEGORY"],
		"gender":['RETL_GNDR_GRP',"GENDER","gender"],
		"platform":['PLATFORM_DIMENSION','STORE_NM'],
		"MSRP":['CHN_LATEST_MSRP'],
		"classification":["SILH_DESC"]
		}
	
	def load_data(self,data,use_cols=None):
		self.input_data = data
		if use_cols:
			self.output_data = self.input_data[use_cols]

	def name_refactor(self):
		rename_dict = {}
		for final_col, init_col_list in self.col_dict.items():
			for init_col in init_col_list:
				if init_col in self.output_data.columns:
					rename_dict[init_col] = final_col
		self.output_data.rename(columns=rename_dict, inplace=True)

	def replace_col_values(self, replace_col, map_dict):
		self.output_data[replace_col] = self.output_data[replace_col].replace(map_dict)


def convert_y_or_n(data, cols:list):
	for col in cols:
		data[col] = data[col].replace({"Y":1, "N":0})

def group_season2_3(df):
	return df.sort_values()[:1]

def convert_inseason_flag_from_int(data, cols:list):
	for col in cols:
		data[col] = data[col].replace({1:"Y", 0:"N"})

# def left_join_with_check(base, new):
# 	after = pd.merge(base, new, left_on=left_on, right_on=right_on, how="left",copy=False)
# 	if len(base) == len(after):
# 		print ("pass!")
# 	else:
# 		print(f"before:{base.shape[0]} after:{after.shape[0]}")
# 	return after

def left_join_with_check(base, new, kwargs):
	after = pd.merge(base, new, how="left", copy=False, **kwargs)
	if len(base) == len(after):
		print ("pass!")
	else:
		print(f"before:{base.shape[0]} after:{after.shape[0]}")
	return after

def process_white_list(white_list):
	white = white_list[white_list["SEASON"]!="MIX SEASON"]
	white["season1"] = white["SEASON"].apply(lambda x: x[:4])
	white["season1"] = white["season1"].apply(lambda x: x[:2]+"20"+x[2:4])
	white["season_sku"] = white.apply(lambda x:x["season1"]+"_"+x["PRODUCT CODE"], axis=1)
	white_use = white[["season_sku","EVENT LIST"]]
	white_use = white_use[white_use["EVENT LIST"].notnull()]
	return white_use


def extract_duplicate_lines_by_key(data, key):
	a = data.drop_duplicates(subset=[key], keep="first")
	b = data.drop_duplicates(subset=[key], keep=False)
	repeat_first = a.append(b).drop_duplicates(keep=False)
	return data[data[key].isin(repeat_first[key].values)]


def process_buy_plan_data(buy_plan,target_season):
	# This is the buy plan one month before the season starts, we use this version as the input for our in season forecast
	# platform is double checked with Mya Ma, store.com=nike.com, CN FY11 E-Commerce=Tmall
	buy_plan["CCD Dt Bus Seasn Yr Cd"] = buy_plan["CCD Dt Bus Seasn Yr Cd"].str.strip()
	target_season_buy_plan = buy_plan[buy_plan["CCD Dt Bus Seasn Yr Cd"] == target_season]
	target_season_buy_plan = target_season_buy_plan[target_season_buy_plan["Cust Sold To Nm"].isin(buy_plan_channel_define.keys())]
	target_season_buy_plan["Cust Sold To Nm"].replace(buy_plan_channel_define,inplace=True)
	target_season_buy_plan_final = target_season_buy_plan.groupby(['Cust Sold To Nm', 'Prod Cd']).agg({'Cnfrmd Qty': sum}).reset_index()
	target_season_buy_plan_final.columns=["platform","prod_code","season_buy_qty"]
	return target_season_buy_plan_final


def process_traffic_data(traffic_raw):
	# weekly traffic(daily average in a week) by sub_platform from elaine
	traffic_raw.rename(columns={"platform":"sub_platform"}, inplace=True)
	traffic_raw["platform"] = np.where(traffic_raw.sub_platform.isin(['SNKRS APP','Nike.com','GROUP PURCHASE']),
									   "NIKE.COM",
									   np.where(traffic_raw.sub_platform.isin(['Tmall', 'TMALL Jordan', 'TMALL YA']),"TMALL",None))
	traffic = traffic_raw[traffic_raw["platform"].notnull()]
	traffic_grouped = traffic.groupby(["platform","week_end"]).sum().reset_index()
	return traffic_grouped

def extract_style_cd(df):
	return df.apply(lambda x: x[:-4])


def prepare_data_master(runtime=2):
	if runtime != 1:
		data_master5 = pd.read_csv(f"{paths.step_data}/data_master2020303.csv")
	else:
		# load raw data from sql server
		sold = pd.read_csv(f"{paths.source}/DIG_TMALL_NIKECOM_WEEKLY_SOLD_SU18_to_HO19.csv")
		attributes = pd.read_csv(f"{paths.source}/attributes_full_20200120.csv")
		inventory = pd.read_csv(f"{paths.source}/inventory_tmall_nikecom_su18_ho19.csv")
		booking = pd.read_excel(f"{paths.source}/TMALL_NIKECOM_BOOKING_SU18_to_HO19.xlsx")
		white_list = pd.read_excel(f"{paths.source}/white_list_sp20_from_mya.xlsx")

		# screen out swoosh.com
		sold = sold[sold["sub_platform"]!='SWOOSH.COM']

		inseason_flag_cols = [f"inseason_flag{i}" for i in range(1,4)] + [f"nd_inseason_flag{i}" for i in range(1,4)]
		convert_y_or_n(sold, cols=inseason_flag_cols)
		agg_func = {key: max for key in inseason_flag_cols}
		agg_func["season2"] = group_season2_3
		agg_func["season3"] = group_season2_3
		# agg_func["reg_msrp"] = group_season2_3
		agg_func["sales_qty"] = 'sum'
		agg_func["sales_amt"] = 'sum'

		group_sold_by_platform = sold.groupby(["platform","season1","prod_code","week_end"]).agg(agg_func).reset_index()
		convert_inseason_flag_from_int(group_sold_by_platform, cols=inseason_flag_cols)
		group_sold_by_platform.to_csv(f"{paths.step_data}/group_sold_by_platform.csv", index=False)

		# merge attribute
		data_refactor = DataRefactor()
		data_refactor.load_data(attributes, use_cols=attribute_use_cols)
		data_refactor.name_refactor()
		attributes_2 = data_refactor.output_data
		data_master = left_join_with_check(group_sold_by_platform, attributes_2, {"on": "prod_code"})
		data_master["season_sku"] = data_master.apply(lambda x: x["season1"]+ "_" + x["prod_code"], axis=1)

		# merge white list tag
		white_use = process_white_list(white_list)
		extract_duplicate_lines_by_key(white_use, "season_sku")
		data_master2 = left_join_with_check(data_master, white_use, {"on": "season_sku"})

		# booking
		data_refactor.load_data(booking, use_cols=booking_use_cols)
		data_refactor.name_refactor()
		booking2 = data_refactor.output_data
		booking2["season_sku"] = booking2.apply(lambda x:x["SEASON_DESC"] + "_" + x["prod_code"], axis=1)
		del booking2["prod_code"]
		data_master3 = left_join_with_check(data_master2, booking2, {"on": ["platform","season_sku"]})

		# inventory
		data_refactor = DataRefactor()
		data_refactor.load_data(inventory, use_cols=inventory_use_cols)
		data_refactor.name_refactor()
		data_refactor.replace_col_values("platform", platform_dict)
		inventory2 = data_refactor.output_data
		inventory2["week_begin"] = pd.to_datetime(inventory2["EOP_DT"])
		inventory_grouped = inventory2.groupby(["platform","prod_code","week_begin"]).sum().reset_index()

		data_master3["week_begin"]=pd.to_datetime(data_master3["week_end"])-datetime.timedelta(days=7)
		data_master4 = left_join_with_check(data_master3, inventory_grouped, {"on": ["platform","prod_code","week_begin"]})
		# data_master4.to_csv(f"{paths.step_data}/data_master20200120.csv", index=False)

		# process weekly traffic(daily average in a week) by sub_platform from elaine
		traffic_raw = pd.read_csv(f"{paths.source}/{traffic_platform}")
		traffic = process_traffic_data(traffic_raw)
		data_master5 = left_join_with_check(data_master4, traffic, {"on":["platform","week_end"]})

		# merge yearweek number
		calendar = pd.read_excel(f"{paths.source}/calendar_date_weekno.xlsx")
		data_master5["week_begin"] = pd.to_datetime(data_master5["week_begin"])
		calendar.rename(columns={"DATE":"week_begin"},inplace=True)
		data_master5 = left_join_with_check(data_master5, calendar[["week_begin", "YrWkNum"]], {"on": "week_begin"})
		data_master5.to_csv(f"{paths.step_data}/data_master2020303.csv", index=False)
	return data_master5


########################################  run inseason forecast model (adapt paper theory)     ######################################

def process_train_and_test_dataset(data_master5, train_season, target_season, run_time=2):
	"""
		Step 1. Data extraction
				** Data truncate for modeling
					1. su18 inseason ftw
					2. su18 and su19 common style(could use ada's and grace's comp instead)
					3. tmall
				** Data aggregate on style level(could use sku and model level instead)
				** Merge buy plan qty(Mya) as q0 and tmall traffic data to replace door# and extract weekly retail moment days
				** Transaction Time period should be in in-season period

		Step 2. Feature preparation
				1. weekly average selling price: avg_selling_price
				2. first week of each style: sn_wk0
				3. md of each week for each style: md
				4. selling week number since wk0: sales_wk

		if it's the first time of running code, run_time should be set to 1

	"""	
	def get_buy_plan_data(train_season):
		if run_time == 1:
			# use buy plan qty as q0(from Mya)
			su18_buy_plan = pd.read_excel(glob(f"{paths.source}/booking/{train_season}/*.xlsx")[0], sheet_name = 'Page1_1', usecols = ['Cust Sold To Nm', 'CCD Dt Bus Seasn Yr Cd', 'Prod Cd', 'Cnfrmd Qty'])
			su18_buy = process_buy_plan_data(su18_buy_plan, train_season)
			# convert sku level buy plan into style level buy plan
			su18_buy["style_cd"] = extract_style_cd(su18_buy["prod_code"])
			su18_buy_style = su18_buy.groupby(["platform","style_cd"]).sum().reset_index()
			with pd.ExcelWriter(f"{paths.step_data}/{train_season}_buy_processed.xlsx") as writer:
				su18_buy.to_excel(writer,"sku_level", index=False)
				su18_buy_style.to_excel(writer, "style_level", index=False)
		else:
			su18_buy_style = pd.read_excel(f"{paths.step_data}/{train_season}_buy_processed.xlsx", sheet_name="style_level")
		return su18_buy_style

	def get_style_traffic():
		style_traffic = pd.read_csv(f"{paths.source}/{traffic_style}")
		style_traffic["week_end"] = pd.to_datetime(style_traffic["week_end"])
		style_traffic["avg_daily_pdp_traffic"] = style_traffic["avg_daily_pdp_traffic"] * 7
		style_traffic["avg_daily_pv"] = style_traffic["avg_daily_pv"] * 7
		style_traffic.rename(columns={"styl_cd": "style_cd",
									  "avg_daily_pdp_traffic": "style_traffic_week",
									  "avg_daily_pv": "style_pv_week"}, inplace=True)
		del style_traffic['YrWkNo']
		return style_traffic

	def agg_sku_to_style(train_season):
		# get su2018 inseason ftw data as train dataset, since we are modeling on style-level(model-level), we are going to aggregate sku
		su18_inseason_ftw_sku = data_master5.query("(season1 == @train_season) & (inseason_flag1 == 'Y') & (PE == 'FTW')")
		su18_inseason_ftw_sku["style_cd"] = extract_style_cd(su18_inseason_ftw_sku["prod_code"])
		su18_inseason_ftw_sku["msrp_amt"] = su18_inseason_ftw_sku.eval("MSRP*sales_qty")
		su18_inseason_ftw_style = su18_inseason_ftw_sku.groupby(["platform","style_cd","week_end","week_begin","YrWkNum"]).agg({
			'sales_qty':sum, 
			'sales_amt':sum,
			"msrp_amt": sum,
			'STYLCOLOR_DESC':'first',
			'gender':'first',
			'category':'first',
			'PE':'first',
			'MSRP':pd.Series.mode,
			'Traffic':'mean',
			'EOP_QTY': sum}).reset_index()
		return su18_inseason_ftw_style

	def get_same_styles_codes_in_train_and_test():
		# find 18 19 common style and attributes, then find common ftw key styles
		su18_styles = set(data_master5.query("(season1 == @train_season) & (inseason_flag1 == 'Y')")["prod_code"].map(lambda x:x[:-4]))
		su19_styles = set(data_master5.query("(season1 == @target_season) & (inseason_flag1 == 'Y')")["prod_code"].map(lambda x:x[:-4]))
		common_styles = su18_styles & su19_styles
		return common_styles

	def filter_inseason_transaction_period(model_master_final, season):
		model_master_final["week_begin"] = pd.to_datetime(model_master_final["week_begin"])
		model_base = model_master_final[(model_master_final["week_begin"] >= pd.to_datetime(f"{season[2:]}-03-31")) &
		(model_master_final["week_begin"] < pd.to_datetime(f"{season[2:]}-07-01")) & 
		(model_master_final.sales_qty >= 0)].sort_values(by=["style_cd","week_begin"])
		return model_base


	def finalize_train_dataset_features(model_base):
		"""
			Step 2 make features required in the function curve
		"""
		model_base["avg_selling_price"] = model_base.eval("sales_amt/sales_qty")

		group_index = ['platform', 'style_cd','STYLCOLOR_DESC','gender','category', 'PE', 'MSRP']
		season_stats = model_base.groupby(group_index).agg({
			'YrWkNum': [min,max],
			'Traffic': max,
			'EOP_QTY': max
			})
		season_stats.columns=["sn_wk0","sn_wk_end", "sn_max_traffic","sn_max_inventory"]
		model_base = left_join_with_check(model_base,season_stats,{"left_on": group_index,"right_index": True})

		model_base["MSRP"] = model_base["MSRP"].astype(float)
		model_base["md"] = model_base.eval("avg_selling_price/MSRP")
		model_base = model_base[~model_base["style_cd"].isin(model_base.query("md>1").style_cd.unique())]		# delete abnormal markdown style
		model_base["sales_wk"] = model_base.eval("YrWkNum - sn_wk0")
		model_base["q0"] = model_base.eval("season_buy_qty / (sn_wk_end - sn_wk0)")
		weekly_max = model_base.groupby(["platform", "style_cd"]).agg({"sales_qty":[max, 'median', 'mean']})
		weekly_max.columns = ["max_wk_sold", "median_wk_sold", "avg_wk_sold"]	# theoretically q0 should not greater than max_wk_sold
		model_base = left_join_with_check(model_base, weekly_max,
										  {"left_on": ["platform", "style_cd"], "right_index": True})
		return model_base

	def extract_train_and_test(season):
		su18_buy_style = get_buy_plan_data(season)
		su18_inseason_ftw_style = agg_sku_to_style(season)
		common_styles = get_same_styles_codes_in_train_and_test()
		su18_inseason_ftw_style["common_style_su19"] = su18_inseason_ftw_style["style_cd"].isin(common_styles)	# True if a style in su2018 also appears in su2019
		# filter tmall and common styles shared with su2019
		model_master = su18_inseason_ftw_style[(su18_inseason_ftw_style["common_style_su19"]>0) & (su18_inseason_ftw_style["platform"] == "TMALL")]
		# merge tmall retail moment days
		retail_moment_data = pd.read_csv(f"{paths.source}/{retail_moment_file_src}")
		model_master = left_join_with_check(model_master, retail_moment_data, {"on":["platform", "YrWkNum"]})
		# merge buy plan(noted: whole season buy qty merge with weekly sales data)
		model_master_final = left_join_with_check(model_master, su18_buy_style, {"on":["platform", "style_cd"]})
		# merge style level traffic
		style_traffic = get_style_traffic()
		model_master_final["week_end"] = pd.to_datetime(model_master_final["week_end"])
		model_master_final = left_join_with_check(model_master_final, style_traffic, {"on":["style_cd","week_end"]})
		model_base = filter_inseason_transaction_period(model_master_final, season)
		model_base = finalize_train_dataset_features(model_base)
		model_base.to_excel(f"{paths.model_data}/model_master_{season}_by_style_same_code.xlsx",index=False)
		return model_base

	train_base = extract_train_and_test(train_season)
	test_base = extract_train_and_test(target_season)
	return train_base, test_base 

"""
	Step 3. Modeling: curve fitting
			1. traffic
			2. q0: 

"""
def curve_function(x, s0, t0, lambda1, lambda2, gamma):
	"""
		!!!! no traffic version
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
    
	# traffic = x[0]
	q0 = x[0]
	sales_wk = x[1]
	rm_days = x[2]
	md = x[3]

	RetailMoment = (1 + s0 * rm_days/7)
	MarketTrend = np.where(((sales_wk < t0) & (sales_wk >= (t0 - lambda1))), np.power((1 - np.power((t0 - sales_wk)/lambda1, 3)), 3),
				np.where(((sales_wk >= t0) & (sales_wk < (t0 + lambda2))), np.power((1 - np.power((sales_wk - t0)/lambda2, 3)),3), 0))
	PriceSensitivity = np.power(md, gamma)

	return q0 * MarketTrend * RetailMoment * PriceSensitivity


def get_params_init_and_boundarys(train_data):
	"""
		no traffic version, the initial version
	"""
	MAX_SOLD_WEEK = train_data["sales_wk"].max()
	ini_t0 = MAX_SOLD_WEEK / 2
	ini_lambda1, ini_lambda2 = ini_t0 , MAX_SOLD_WEEK / 2
	ini_s0 = 1.0
	ini_gamma = -1.0
	init_params = [ini_s0, ini_t0, ini_lambda1, ini_lambda2, ini_gamma]
	bounds = ([0, 0, 0, 0, -np.inf], [10, MAX_SOLD_WEEK, MAX_SOLD_WEEK, MAX_SOLD_WEEK, 0])
	return init_params, bounds


def run_curve_fit(train_dataset, get_init_params=None, curve_func=None, use_x=None):
	"""
	:param train_dataset:
	:param get_init_params: get initial guess and bounds for curve params
	:param curve_func: use what curve function
	:param use_x: features that in the curve as original input
	:return: in sample forecast result and curve params for each style
	"""

	get_init_params = get_init_params or get_params_init_and_boundarys
	curve_func = curve_func or curve_function
	use_x = use_x or ["q0", "sales_wk", "DaysRM", "md"]

	in_sample_fcst_result = pd.DataFrame()
	model_params = {}
	styles_set = train_dataset["style_cd"].unique()
	for style_code in styles_set:
		train_data = train_dataset.query("style_cd == @style_code")
		train_data["sales_wk"] += 1
		xdata = train_data[use_x].values.T
		ydata = train_data["sales_qty"].values
		params_guess, params_bounds = get_init_params(train_data)
		try:
			popt, pcov = curve_fit(curve_func, xdata, ydata, p0 = params_guess, bounds = params_bounds)
			predict_qty = pd.Series(curve_func(xdata, *popt),name="fcst_qty")
			result_data = pd.concat([train_data.reset_index(drop=True),predict_qty],axis=1)
			in_sample_fcst_result = in_sample_fcst_result.append(result_data,ignore_index=True)
			model_params[style_code] = list(popt)
		except:
			print(style_code)
			print(xdata)
	return in_sample_fcst_result, model_params


def predict_target_season(fcst_result, test_data, params, curve_func=None, use_x=None):
	curve_func = curve_func or curve_function
	use_x = use_x or ["q0", "sales_wk", "DaysRM", "md"]
	out_sample_fcst_result = pd.DataFrame()
	for style_code in fcst_result["style_cd"].unique():
		curve_param = params[style_code]
		predict_style_data = test_data.query("style_cd == @style_code")
		predict_style_data["sales_wk"] += 1
		xdata = predict_style_data[use_x].values.T
		y_predict = pd.Series(curve_func(xdata, *curve_param), name="fcst_qty")
		predict_style_data = pd.concat([predict_style_data.reset_index(drop=True), y_predict], axis=1)
		out_sample_fcst_result = out_sample_fcst_result.append(predict_style_data, ignore_index=True)
	return out_sample_fcst_result


def main(model_base, y, x_cols, curve_func, init_func, model_name):

	train_dataset = model_base.dropna(subset = x_cols + [y], axis=0)
	train_dataset = train_dataset.query("(sn_wk_end - sn_wk0) > 5")     # if total selling weeks in a season is fewer than 5 weeks, then kick out
	in_sample_fcst_result, model_params = run_curve_fit(train_dataset, get_init_params=init_func,
													curve_func=curve_func, use_x=x_cols)
	out_sample_fcst_result = predict_target_season(in_sample_fcst_result, test_base, model_params,
												   curve_func=curve_func, use_x=x_cols)
	paths.add_source(model_name, f"model_data/{model_name}")
	in_sample_fcst_result.to_excel(f"{getattr(paths, model_name)}/fcst_result_insample_{train_season}_{model_name}.xlsx", index=False)
	out_sample_fcst_result.to_excel(f"{getattr(paths, model_name)}/fcst_result_outsample_{target_season}_{model_name}.xlsx", index=False)
	with open(f"{getattr(paths, model_name)}/model_params_{train_season}_{model_name}.json","w") as f:
		json.dump(model_params,f)


if __name__ == '__main__':

	paths = PathManager("", path_dict)
	paths.add_source("step_data", "step_data")
	data_master = prepare_data_master(runtime=2)
	train_season = "SU2018"
	target_season = 'SU2019'
	paths.add_source("model_data", "model_data")
	model_base, test_base = process_train_and_test_dataset(data_master, train_season, target_season, run_time=2)	# run_time set to 1 if first run buy plan data

	y = "sales_qty"


	# linear md
	linear_md_settings = {"x_cols": ["style_traffic_week", "median_wk_sold", "sales_wk", "DaysRM", "md"],
						"curve_func": curve_function_linear_md, "init_func": get_init_and_bounds_linear_md,
						"model_name":"linear_md"}
	main(model_base, y, **linear_md_settings)

	# linear md & site traffic
	linear_md_site_traffic_settings = {"x_cols": ["Traffic", "median_wk_sold", "sales_wk", "DaysRM", "md"],
						"curve_func": curve_function_linear_md, "init_func": get_init_and_bounds_linear_md,
						"model_name":"linear_md_site_traffic"}
	main(model_base, y, **linear_md_site_traffic_settings)

	# attempt to try other models
	# # style traffic, buy plan q0
	# traffic_settings = {"x_cols": ["style_traffic_week", "q0", "sales_wk", "DaysRM", "md"],
	# 					"curve_func": curve_function_traffic, "init_func": get_init_and_bounds_traffic,
	# 					"model_name":"style_traffic_old_rm"}
	# main(model_base, y, **traffic_settings)
	#
	# # remove traffic, buy plan q0
	# no_traffic_settings = {"x_cols":["q0", "sales_wk", "DaysRM", "md"],
	# 					   "curve_func":curve_function, "init_func":get_params_init_and_boundarys,
	# 					   "model_name":"no_traffic_old_rm"}
	# main(model_base, y, **no_traffic_settings)
	#
	# # use real max weekly sold qty as q0
	# max_sold_q0_settings = {"x_cols": ["style_traffic_week", "max_wk_sold", "sales_wk", "DaysRM", "md"],
	# 					"curve_func": curve_function_traffic, "init_func": get_init_and_bounds_traffic,
	# 					"model_name":"max_sold_q0"}
	# main(model_base, y, **max_sold_q0_settings)
	#
	# # use real median weekly sold qty as q0
	# median_sold_q0_settings = {"x_cols": ["style_traffic_week", "median_wk_sold", "sales_wk", "DaysRM", "md"],
	# 					"curve_func": curve_function_traffic, "init_func": get_init_and_bounds_traffic,
	# 					"model_name":"median_sold_q0"}
	# main(model_base, y, **median_sold_q0_settings)
	#
	# # use avg weekly sold qty as q0
	# avg_sold_q0_settings = {"x_cols": ["style_traffic_week", "avg_wk_sold", "sales_wk", "DaysRM", "md"],
	# 					"curve_func": curve_function_traffic, "init_func": get_init_and_bounds_traffic,
	# 					"model_name":"avg_sold_q0"}
	# main(model_base, y, **avg_sold_q0_settings)
	#
	# # use real median sold qty as q0 and no traffic
	# med_sold_q0_no_traffic_settings = {"x_cols":["median_wk_sold", "sales_wk", "DaysRM", "md"],
	# 					   "curve_func":curve_function, "init_func":get_params_init_and_boundarys,
	# 					   "model_name":"med_sold_q0_no_traffic"}
	# main(model_base, y, **med_sold_q0_no_traffic_settings)




