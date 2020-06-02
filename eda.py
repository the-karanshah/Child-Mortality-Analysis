# used Dask

# EDA for all csvs in one run

# System:
# GCP Cluster Details:
# Master node: Standard (1 master, N workers)
# Machine type: e2-standard-2
# Number of GPUs: 0
# Primary disk type: pd-standard
# Primary disk size: 64GB
# Worker nodes: 4 (2 of which were up and running)
# Machine type: e2-highmem-4
# Number of GPUs: 0
# Primary disk type: pd-standard
# Primary disk size: 32GB
# Image Version: 1.4.27-debian9


import dask.dataframe as dd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

dlist = {'alcohol': 'float64',       'as_binned': 'float64',       'building_no': 'object',       'cart': 'float64',       'chew': 'float64',       'client_hh_id': 'float64',       'client_hl_id': 'float64',       'client_w_id': 'float64',       'cooking_fuel': 'float64',       'currently_dead_or_out_migrated': 'float64',       'date_of_birth': 'float64',       'date_of_intr': 'float64',       'diagnosed_for': 'float64',       'drinking_water_source': 'float64',       'ever_conceived': 'object',       'fid': 'float64',       'fidx': 'float64',       'headname': 'object',       'hh_expall_status': 'float64',       'hh_id': 'float64',       'hh_serial_no': 'float64',       'highest_qualification': 'float64',       'hl_expall_status': 'float64',       'hl_id': 'float64',       'house_status': 'float64',       'house_structure': 'object',       'household_have_electricity': 'float64',       'identifcation_code': 'object',       'is_bicycle': 'float64',       'is_car': 'float64',       'is_computer': 'float64',       'is_radio': 'float64',       'is_refrigerator': 'float64',       'is_scooter': 'float64',       'is_sewing_machine': 'float64',       'is_telephone': 'float64',       'is_television': 'float64',       'is_tractor': 'float64',       'is_washing_machine': 'float64',       'is_water_filter': 'float64',       'is_water_pump': 'float64',       'kitchen_availability': 'float64',       'land_possessed': 'float64',       'lighting_source': 'float64',       'month_of_intr': 'float64',       'no_of_dwelling_rooms': 'float64',       'owner_status': 'float64',       'relation_to_head': 'float64',       'smoke': 'float64',       'sn': 'object',       'status': 'float64',       'symptoms_pertaining_illness': 'float64',       'toilet_used': 'float64',       'treatment_source': 'float64',       'usual_residance': 'float64',       'w_expall_status': 'float64',       'w_id': 'float64',       'w_status': 'float64',       'year_of_intr': 'float64',       'date_of_marriage': 'float64',       'month_of_marriage': 'float64',       'occupation_status': 'float64',       'other_int_code': 'object',       'year_of_marriage': 'float64',       'cdoi': 'object',       'disability_status': 'float64',       'illness_type': 'float64',       'injury_treatment_type': 'float64',       'member_identity': 'float64',       'month_of_birth': 'float64',       'religion': 'float64',       'social_group_code': 'float64',       'year_of_birth': 'float64',       'sex': 'float64',       'age': 'float64',       'id': 'float64',       'marital_status': 'float64',       'schedule_id': 'float64',       'year': 'float64',       'fidh':'float64',       'householdstatus':'float64',       'housestatus':'float64',       'iscoveredbyhealthscheme':'float64',       'recordstatus':'float64',        'recordupdatedcount':'float64',        'residancial_status':'float64'}

columns_for_eda = ['state', 'district', 'rural', 
                   'age', 'marital_status', 'delivered_any_baby',
                   'born_alive_female', 'born_alive_male', 'born_alive_total', 'surviving_female', 'surviving_male', 'surviving_total', 'mother_age_when_baby_was_born', 
                   'is_tubectomy', 'is_vasectomy', 'is_copper_t', 'is_pills_daily', 'is_piils_weekly', 'is_emergency_contraceptive', 'is_condom', 'is_moder_methods', 
                   'is_contraceptive', 'is_periodic_abstinence', 'is_withdrawal', 'is_amenorrahoea', 'is_other_traditional_method',
                   'when_you_bcome_mother_last_time',
                   'is_any_fp_methos_used', 'fp_method_used', 'source_of_treatment_for_fp', 'how_long_using_this_method', 'method_type_used_in_last_5_yrs',
                   'is_anc_registered', 'anm_in_last_3_months', 'during_pregnancy', 'during_lactation',
                   'aware_abt_rti', 'aware_abt_hiv', 'aware_of_haf', 'aware_of_the_danger_signs',
                   'religion', 'social_group_code',
                   'currently_attending_school', 'reason_for_not_attending_school', 'highest_qualification', 'occupation_status',
                   'disability_status', 'injury_treatment_type', 'illness_type', 'treatment_source', 'sought_medical_care', 
                   'chew', 'smoke', 'alcohol',
                   'house_structure', 'drinking_water_source', 'is_water_filter', 'toilet_used', 'is_toilet_shared', 'household_have_electricity', 'lighting_source', 'cooking_fuel',
                   'ever_conceived', 'no_of_times_conceived', 'age_at_first_conception',
                   'counselled_for_menstrual_hyg', 'aware_abt_haf', 'aware_abt_ort_ors', 'aware_abt_ort_ors_zinc', 'aware_abt_danger_signs_new_born',
                   'iscoveredbyhealthscheme',	'healthscheme_1',	'healthscheme_2']

csv_name = '9'
df = dd.read_csv('data/' + csv_name + '.csv', sep='|',
       usecols = columns_for_eda,
       dtype=dlist,
       na_values=[str])
m_df = df[(df['age'] >= 16) & (df['delivered_any_baby'] == 1)]

m_df_eda = m_df[columns_for_eda]


columns_to_drop = ['fp_method_used', 'source_of_treatment_for_fp', 'how_long_using_this_method',
                   'method_type_used_in_last_5_yrs', 'anm_in_last_3_months', 'during_pregnancy',
                   'during_lactation', 'aware_abt_ort_ors_zinc', 'aware_abt_ort_ors',
                   'aware_abt_haf', 'aware_abt_danger_signs_new_born',
                   'currently_attending_school', 'reason_for_not_attending_school', 'treatment_source',
                   'sought_medical_care', 'is_toilet_shared', 'ever_conceived', 'no_of_times_conceived',
                   'age_at_first_conception', 'healthscheme_1', 'healthscheme_2', 'when_you_bcome_mother_last_time']
m_df_eda_filterd = m_df_eda.drop(columns_to_drop, axis=1)

columns_to_drop_where_rows_missing = ['house_structure', 'drinking_water_source','is_water_filter',
                  'toilet_used', 'household_have_electricity', 'lighting_source',
                  'cooking_fuel', 'born_alive_female', 'born_alive_male', 'born_alive_total',
                  'surviving_female', 'surviving_male', 'surviving_total', 'mother_age_when_baby_was_born',
                  'is_tubectomy', 'is_vasectomy', 'is_copper_t', 'is_pills_daily', 'is_piils_weekly', 'is_emergency_contraceptive',
                  'is_condom', 'is_moder_methods', 'is_contraceptive', 'is_periodic_abstinence',
                  'is_withdrawal', 'is_amenorrahoea', 'is_other_traditional_method', 'religion', 'social_group_code']

for column in columns_to_drop_where_rows_missing:
  m_df_eda_filterd = m_df_eda_filterd[m_df_eda_filterd[column].notnull()]

replacement_map = {'is_anc_registered': 2, 'aware_of_haf': 3, 'counselled_for_menstrual_hyg': 2,
               'iscoveredbyhealthscheme': 3, 'is_any_fp_methos_used': 2, 'aware_abt_hiv': 2,
               'aware_of_the_danger_signs': 2, 'aware_abt_rti': 2, 'alcohol': 0,
               'smoke': 0, 'chew': 0, 'disability_status': 0, 'injury_treatment_type': 0,
               'illness_type': 0, 'highest_qualification': 0, 'occupation_status': 16}

for column, replacement in replacement_map.items():
  m_df_eda_filterd[column] = m_df_eda_filterd[column].fillna(value=replacement)

modern_method_columns = ['is_tubectomy', 'is_vasectomy', 'is_copper_t', 'is_pills_daily',
                         'is_piils_weekly', 'is_emergency_contraceptive', 'is_condom',
                         'is_moder_methods']
traditional_method_columns = ['is_contraceptive', 'is_periodic_abstinence', 'is_withdrawal',
                              'is_amenorrahoea', 'is_other_traditional_method']


def check_if_modern_method_used(row):
  return 1 if any([row[method] == 1 for method in modern_method_columns]) else 0

def check_if_traditional_method_used(row):
  return 1 if any([row[method] == 1 for method in traditional_method_columns]) else 0

m_df_eda_filterd['modern_methods_used'] = m_df_eda_filterd.apply(lambda row: check_if_modern_method_used(row), axis=1)
m_df_eda_filterd['traditional_methods_used'] = m_df_eda_filterd.apply(lambda row: check_if_traditional_method_used(row), axis=1)
m_df_eda_filterd = m_df_eda_filterd.drop(modern_method_columns + traditional_method_columns, axis=1)
m_df_eda_filterd['children_lost'] = m_df_eda_filterd['born_alive_total'] - m_df_eda_filterd['surviving_total']

rating_cols = ['iscoveredbyhealthscheme', 'counselled_for_menstrual_hyg',
                'cooking_fuel', 'lighting_source', 'household_have_electricity', 'is_water_filter', 'chew', 'smoke', 'alcohol', 
                'disability_status', 'highest_qualification', 'social_group_code', 'aware_abt_rti', 'aware_abt_hiv', 'aware_of_haf', 'aware_of_the_danger_signs', 
                'is_any_fp_methos_used', 'is_anc_registered', 'delivered_any_baby', 'marital_status', 'rural']
drop_cols = ['occupation_status', 'injury_treatment_type', 'drinking_water_source', 'toilet_used', 'religion']

decode = {1:'1', 2:'0'}

married = {2:'0', 3:'1', 4:'2', 5:'3', 6:'7', 7:'8', 8:np.NaN} #Married but Gauna not performed-2,Married and Gauna performed-3,Remarried -4,Widow/Widower-5,Divorced-6,Separated-7,Not stated-8
cooking_fuel = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'0', 0:'0'} #Firewood-1,Crop Residue -2,Cow dung cake -3,Coal/lignite/Charcoal-4,Kerosene-5,LPG/PNG -6,Electricity-7,Biogas-8,Any other-9,No cooking-0
lighting_source = {4:'1', 2:'2', 1:'3', 3:'4', 5:'0', 6:'0'} #Electricity-1,Kerosene-2,Solar -3,Other Oils-4,Any other-5,No lighting-6
chew = {0:'0', 3:'1', 5:'2', 4:'3', 2:'4', 6:'5', 7:'6'} #Pan with tobacco-1,Pan without tobacco-2,Gutka/Pan masala with tobacco-3,Gutka/Pan masala without tobacco -4,Tobacco only-5,Ex – Chewer -6,Never chewed-7,Not known-0
smoke = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'} #Usual smoker-1,Occasional smoker-2,Ex – Smoker-3,Never smoked-4,Not known-0
alcohol = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'} #Usual drinker-1,Occasional drinker-2,Ex – drinker-3,Never drank-4,Not known-0
disability_status = {1:'1', 2:'1', 3:'1', 4:'1', 5:'1', 6:'1', 7:'1', 0:'0'} #Mental-1,Visual-2,Hearing-3,Speech-4,Locomotor-5,Multiple-6,No Disability-0 (Others--7 :used in First & Second updation Survey only: details for Codes 0 to 6 remained same during the First & Second updation Survey ) 
highest_qualification = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'8', 8:'9', 9:'7'} #Illiterate-0,Literate Without formal education-1,Literate With formal education-Below primary-2,Literate With formal education-Primary-3,Literate With formal education-Middle-4,Literate With formal education-Secondary/Matric (Class-X)-5,Literate With formal education-Hr. Secondary/Sr. Secondary/Pre-university(Class XII)-6,Literate With formal education-Graduate/B.Tech/B.B.A/MBBS/Equivalent-7,Literate With formal education-Post Grad/ M.Tech/M.B.A/ MD/Equivalent or higher-8, Literate With formal education-Non-technical/Technical diploma or certificate not equivalent to a degree-9
social_group_code = {1:'1', 2:'2', 3:'3'} #SC-1, ST-2, Others-3
house_structure = {1:'1', 2:'2', 3:'3', 4:'4'} #Pucca -1, Semi Pucca -2, Kuccha -3, Others -4
iscoveredbyhealthscheme = {1:'1', 2:'2', 3:'3'}
counselled_for_menstrual_hyg = {1:'1', 2:'0'}
rural = {1:'1', 2:'2'}

def convertToNo(x):
  return 'no' if x != 'yes' else x

def process(df):
  df = df.astype('int8')
  for col in rating_cols:
    if col == 'marital_status':
      df[col] = df[col].map(married).astype('int8')
    elif col == 'traditional_methods_used':
      continue
    elif col == 'modern_methods_used':
      continue
    elif col == 'iscoveredbyhealthscheme':
      df[col] = df[col].replace(iscoveredbyhealthscheme).astype('int8')
    elif col == 'counselled_for_menstrual_hyg':
      df[col] = df[col].replace(counselled_for_menstrual_hyg).astype('int8')
    elif col == 'cooking_fuel':
      df[col] = df[col].replace(cooking_fuel).astype('int8')
    elif col == 'lighting_source':
      df[col] = df[col].replace(lighting_source).astype('int8')
    elif col == 'household_have_electricity':
      df[col] = df[col].replace(decode).astype('int8')
    elif col == 'is_water_filter':
      df[col] = df[col].replace(decode).astype('int8')
    elif col == 'chew':
      df[col] = df[col].replace(chew).astype('int8')
    elif col == 'smoke':
      df[col] = df[col].replace(smoke).astype('int8')
    elif col == 'alcohol':
      df[col] = df[col].replace(alcohol).astype('int8')
    elif col == 'disability_status':
      df[col] = df[col].replace(disability_status).astype('int8')
    elif col == 'highest_qualification':
      df[col] = df[col].replace(highest_qualification).astype('int8')
    elif col == 'social_group_code':
      df[col] = df[col].replace(social_group_code).astype('int8')
    elif col == 'house_structure':
      df[col] = df[col].replace(social_group_code).astype('int8')
    elif col == 'aware_abt_rti' or col == 'aware_abt_hiv' or col == 'aware_of_haf' or col == 'aware_of_the_danger_signs' or col == 'is_any_fp_methos_used' or col == 'is_anc_registered' or col == 'delivered_any_baby':
      df[col] = df[col].replace(decode).astype('int8')
    elif col == 'rural':
      df[col] = df[col].replace(rural).astype('int8')
    
  # drop columns
  df = df.drop(drop_cols, axis=1)

  return df.astype('int8')
m_df_eda_filterd_p = process(m_df_eda_filterd)
# m_df_eda_filterd_p.to_csv("data/processed_"+csv_name+".csv", compute=True, single_file = True)

finalcsv = m_df_eda_filterd_p.compute()

features = finalcsv.loc[:, finalcsv.columns != 'children_lost']
labels = finalcsv.loc[:, finalcsv.columns == 'children_lost']

x = features.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

finalcsv = pd.DataFrame(x_scaled, columns=features.columns)
finalcsv['children_lost'] = labels.values

finalcsv.to_csv("data/processedTest2_"+csv_name+".csv", index=False)