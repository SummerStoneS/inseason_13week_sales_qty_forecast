import os
import pandas as pd
import datetime
from config import *

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
		"platform":['PLATFORM_DIMENSION','STORE_NM']
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

def left_join_with_check(base, new, left_on, right_on):
	after = pd.merge(base, new, left_on=left_on, right_on=right_on, how="left")
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

paths = PathManager("", path_dict)
paths.add_source("step_data", "step_data")

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
agg_func["reg_msrp"] = group_season2_3
agg_func["sales_qty"] = 'sum'
agg_func["sales_amt"] = 'sum'

group_sold_by_platform = sold.groupby(["platform","season1","prod_code","week_end"]).agg(agg_func).reset_index()
convert_inseason_flag_from_int(group_sold_by_platform, cols=inseason_flag_cols)
group_sold_by_platform.to_csv("group_sold_by_platform.csv", index=False)

# merge attribute
data_refactor = DataRefactor()
data_refactor.load_data(attributes, use_cols=attribute_use_cols)
data_refactor.name_refactor()
attributes_2 = data_refactor.output_data
data_master = left_join_with_check(group_sold_by_platform, attributes_2, left_on="prod_code", right_on="prod_code")
data_master["season_sku"] = data_master.apply(lambda x: x["season1"]+ "_" + x["prod_code"], axis=1)

# merge white list tag
white_use = process_white_list(white_list)
extract_duplicate_lines_by_key(white_use, "season_sku")
data_master2 = left_join_with_check(data_master, white_use, left_on="season_sku", right_on="season_sku")

# booking
data_refactor.load_data(booking, use_cols=booking_use_cols)
data_refactor.name_refactor()
booking2 = data_refactor.output_data
booking2["season_sku"] = booking2.apply(lambda x:x["SEASON_DESC"] + "_" + x["prod_code"], axis=1)
del booking2["prod_code"]
data_master3 = left_join_with_check(data_master2, booking2, left_on=["platform","season_sku"], right_on=["platform","season_sku"])

# inventory
data_refactor = DataRefactor()
data_refactor.load_data(inventory, use_cols=inventory_use_cols)
data_refactor.name_refactor()
data_refactor.replace_col_values("platform", platform_dict)
inventory2 = data_refactor.output_data
inventory2["week_begin"] = pd.to_datetime(inventory2["EOP_DT"])
inventory_grouped = inventory2.groupby(["platform","prod_code","week_begin"]).sum().reset_index()

data_master3["week_begin"]=pd.to_datetime(data_master3["week_end"])-datetime.timedelta(days=7)
data_master4 = left_join_with_check(data_master3, inventory_grouped, left_on=["platform","prod_code","week_begin"], right_on=["platform","prod_code","week_begin"])
data_master4.to_csv(f"{paths.step_data}/data_master20200120.csv", index=False)