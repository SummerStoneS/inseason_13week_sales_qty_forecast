import pandas as pd
# save_file

data_master_file = "data_master2020310.csv"     # sku level sales qty & attributes & site traffic & booking, this is the basic data for modeling

# source
retail_moment_file_src = "20200302_Dig_RetailMoment_Days_SP18_HO19_v1.0.csv"        # Elaine根据traffic整理出来的retail moment days
rm_days_advanced_file_src = "20200302_Dig_RetailMoment_Days_SP18_HO19_ComImported_MoreDaysAdded_v2.0.csv"       # 基于上一版和nd的promotion calendar修改，加上了更多小节日，可能并没有反映在traffic上

traffic_platform = "20200228_Traffic_by_SubPlatform_Week_v1.0.csv"
traffic_style = "20200305_Tmall_StyleLevel_Traffic_by_Week_v1.0.csv"

# grace comp list
comp_model_file = "IS FCST MAPPING_grace.xlsx"

attribute_use_cols = ['STYLCOLOR_CD', 'STYLCOLOR_DESC', 'PE', 'GBL_CAT_SUM_DESC', 'RETL_GNDR_GRP', 'Classification','CHN_LATEST_MSRP']
booking_use_cols = ['PLATFORM_DIMENSION', 'SEASON_DESC','STYLCOLOR_CD', 'or_booking', 'ar_booking']
inventory_use_cols = ['STORE_NM','STYLCOLOR_CD','EOP_QTY', 'EOP_DT','IN_TRANSIT_QTY', 'bz_wh_transit_qty']

platform_dict = {
'L&F S-SDC DIGITAL': "NIKE.COM", 
'Tmall': "TMALL", 
'Nike.com': "NIKE.COM", 
'NIKE.COM-NORTH': "NIKE.COM",
'TMALL Jordan': "TMALL", 
'TMALL YA': "TMALL", 
'GROUP PURCHASE': "NIKE.COM"}

buy_plan_channel_define = {
'CN FY11 E-Commerce':'TMALL',
'Store.com':'NIKE.COM',
'CN Digital Tmall':'TMALL', 
'GROUP PURCHASE':'NIKE.COM', 
'TMALL STORE-JORDAN': 'TMALL', 
'TMALL STORE-YA': 'TMALL'}


final_dataset_have_cols = ['platform', 'style_cd', 'week_end', 'week_begin', 'YrWkNum',
                           'sales_qty', 'sales_amt', 'msrp_amt', 'STYLCOLOR_DESC', 'gender',
                           'category', 'PE', 'MSRP', 'Traffic', 'EOP_QTY', 'common_style_su19',
                           'DaysRM', 'DaysRM_PreheatIncluded', 'season_buy_qty',
                           'style_traffic_week', 'style_pv_week', 'avg_selling_price', 'sn_wk0',
                           'sn_wk_end', 'sn_max_traffic', 'sn_max_inventory', 'md', 'sales_wk',
                           'q0', 'max_wk_sold', 'median_wk_sold', 'avg_wk_sold']

season_periods_dict = {"SP":("01-01","03-31"), "SU":("03-31","07-01"), "FA":("07-01","10-01"), "HO":("10-01", "12-31")}

agg_dict = {
    'sales_qty':sum,
    'sales_amt':sum,
    "msrp_amt": sum,
    'STYLCOLOR_DESC':'first',
    'gender':'first',
    'category':'first',
    'PE':'first',
    'MSRP': pd.Series.median,
    'Traffic':'mean',
    'EOP_QTY': sum}

agg_model_dict = {
    'sales_qty':sum,
    'sales_amt':sum,
    "msrp_amt": sum,
    'STYLCOLOR_DESC':'first',
    'gender':'first',
    'category':'first',
    'PE':'first',
    'MSRP': pd.Series.median,
    'Traffic':'mean',
    'EOP_QTY': sum,
    'DaysRM': 'mean',
    'DaysRM_PreheatIncluded': 'mean',
    'season_buy_qty': sum,
    'style_traffic_week': sum,
    'style_pv_week': sum}