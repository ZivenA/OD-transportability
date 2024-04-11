import pandas as pd

# Uncomment the section of the following file corresponding to the
# dataset you wish to run the models on.

# --- BEGIN FOREST FIRES SETUP ---
# defaults = {
#     'cause_H': 1,
#     'cause_PB': 0,
#     'cause_L': 0,
#     'cause_RE': 0,
#     'cause_U': 0,
#     '1': 0,
#     '2': 0,
#     '3': 0,
#     '4': 0,
#     'Crown': 0,
#     'Dump': 0,
#     'Forest': 0,
#     'Grass': 0,
#     'Ground': 0,
#     'IFR': 0,
#     'OFR': 0,
#     'Other': 0,
#     'Prescribed Burn': 0,
#     'Req For Assist': 0,
#     'Request For Ass*': 0,
#     'Surface': 0,
#     'Wildfire': 0,
#     'Summer': 1,
#     'Fall': 0,
#     'Winter': 0,
#     'Spring': 0,
#     'Size>-4': 1, 'Size>-2': 1, 'Size>-1': 1, 'Size>0': 1,
#     'Size>1': 0, 'Size>2': 0, 'Size>3': 0, 'Size>4': 0, 'Size>5': 0, 'Size>6': 0,
#     'after1940': 1, 'after1950': 1, 'after1960': 1, 'after1970': 1, 'after1980': 1, 'after1990': 1,
#     'after2000': 0, 'after2010': 0, 'after2020': 0
# }

# iterations_num_feats = [
#                         # 1,
#                         # 2,
#                         # 3,
#                         # 4,
#                         # 5,
#                         # 6,
#                         # 7,
#                         # 8,
#                         # 9,
#                         10
#                         ]

# source_target_zones = [
#     # ('ECOZONE', 8, 7),
#     # ('ECOZONE', 8, 14),
#     # ('ECOZONE', 9, 14),
#     # ('ECOZONE', 6, 14),
#     # ('ECOZONE', 9, 6),
#     # ('ECOZONE', 4, 9),
#     # ('ECOZONE', 9, 10),
#     # ('ECOZONE', 6, 10),
#     # ('ECOZONE', 12, 8),
#     # ('ECOZONE', 5, 7),
#     # ('ECOZONE', 5, 9),
#     # ('ECOZONE', 5, 6),
#     # ('ECOZONE', 6, 4),
#     # ('ECOZONE', 12, 4),
#     # ('ECOZONE', 11, 12),
#     # ('ECOZONE', 4, 11),
#     # ('ECOZONE', 11, 10),
#     # ('ECOZONE', 14, 12),
#     # ('ECOZONE', 9, 11),
#     # ('ECOZONE', 7, 11),
#     # ('Size>2', 0, 1),
#     # ('Size>2', 1, 0),
#     # ('after1950', 0, 1),
#     # ('after1950', 1, 0),
#     # ('Size>1', 1, 0),
#     # ('Size>1', 0, 1),
#     # ('Dump', 0, 1),
#     # ('Other', 0, 1), 
# ]


# data_filepath = "Datasets/fires_correct_encoded.csv"
# label_col = 'duration_days'

# simple_output_filename = "Results/simple_results_fires.csv"
# soph_xgb_filename = "Results/xgb_fires"
# xgb_ratio_filename = "Results/xgb_ratio_fires"
# one_split_simple_filename = "Results/single_split_fires.csv"

# def df_fetch_and_cleanup():
#     fp = data_filepath
#     df = pd.read_csv(fp, dtype=int)

#     df = df[~(df['duration_days'] < 0)] #remove error
#     df.loc[df['duration_days'] > 10, 'duration_days'] = -1 #large
#     df.loc[df['duration_days'] >= 0, 'duration_days'] = 0
#     df.loc[df['duration_days'] == -1, 'duration_days'] = 1

#     return df

### --- END FOREST FIRES SETUP ---

### --- BEGIN BANK_CHURN SETUP ---

# defaults = {
#     'Complain' : 0.0 ,
#     'NumOfProducts' : 1.0 ,
#     'PoorCr' : 0.0 ,
#     'IsActiveMember' : 1.0 ,
#     'AgeDecade' : 3.0 ,
#     'Geography' : 1.0 ,
#     'Balabove100k' : 0.0 ,
#     'HasCrCard' : 1.0 ,
#     'Salabove100k' : 1.0 ,
#     'FairCr' : 0.0 ,
#     'Tenure' : 2.0 ,
#     'Satisfaction Score' : 3.0 ,
#     'NoBalance' : 0.0 ,
#     'Card Type' : 3.0 ,
#     'Salabove150k' : 0.0 ,
#     'Gender' : 0.0 ,
#     'Salabove50k' : 1.0 ,
#     'Above400pt' : 1.0 ,
#     'Above800pt' : 0.0 ,
#     'VGoodCr' : 0.0 ,
#     'GoodCr' : 0.0 ,
#     'ExcellentCr' : 0.0 ,
# }

# iterations_num_feats = [
#                         1,
#                         2,
#                         3,
#                         4,
#                         5,
#                         6,
#                         7,
#                         8,
#                         9,
#                         10
#                         ]

# # (split_col, source, target)
# source_target_zones = [
#     ('Gender', 1.0, 0.0),
#     ('Gender', 0.0, 1.0),
#     ('Geography', 1.0, 2.0),
#     ('Geography', 2.0, 1.0),
#     ('Geography', 2.0, 3.0),
#     ('Geography', 3.0, 2.0),
#     ('Geography', 1.0, 3.0),
#     ('Geography', 3.0, 1.0),
#     ('HasCrCard', 1.0, 0.0),
#     ('HasCrCard', 0.0, 1.0),
#     # ('Complain', 1.0, 0.0),
#     # ('Complain', 0.0, 1.0),
# ]

# data_filepath = "Datasets/bank_churn_cat.csv"
# label_col = 'Exited'

# simple_output_filename = "Results/simple_results_bank.csv"
# soph_xgb_filename = "Results/xgb_bank"
# xgb_ratio_filename = "Results/xgb_ratio_bank"
# one_split_simple_filename = "Results/single_split_bank.csv"

# def df_fetch_and_cleanup():
#     fp = data_filepath
#     df = pd.read_csv(fp, dtype=float)

#     return df

#### --- END BANK CHURN SETUP ---


### --- BEGIN BINNED LODGEPOLE COVTYPE SETUP ---

# # Default values are needed for the simple single-variable models
# defaults = {
#     'Wilderness_Area4' : 0 ,
#     'Elevation_binned' : 2 ,
#     'Soil_Type2' : 0 ,
#     'Soil_Type4' : 0 ,
#     'Soil_Type12' : 0 ,
#     'Soil_Type32' : 0 ,
#     'Wilderness_Area1' : 0 ,
#     'Soil_Type38' : 0 ,
#     'Soil_Type39' : 0 ,
#     'Soil_Type22' : 0 ,
#     'Soil_Type29' : 0 ,
#     'Soil_Type40' : 0 ,
#     'Soil_Type23' : 0 ,
#     'Soil_Type20' : 0 ,
#     'Soil_Type31' : 0 ,
#     'Soil_Type30' : 0 ,
#     'Soil_Type33' : 0 ,
#     'Soil_Type24' : 0 ,
#     'Soil_Type34' : 0 ,
#     'Horizontal_Distance_To_Hydrology_binned' : 0 ,
#     'Soil_Type13' : 0 ,
#     'Wilderness_Area2' : 0 ,
#     'Horizontal_Distance_To_Roadways_binned' : 1 ,
#     'Soil_Type17' : 0 ,
#     'Hillshade_Noon_binned' : 0 ,
#     'Wilderness_Area3' : 0 ,
#     'Soil_Type27' : 0 ,
#     'Horizontal_Distance_To_Fire_Points_binned' : 0 ,
#     'Soil_Type11' : 0 ,
#     'Soil_Type16' : 0 ,
#     'Soil_Type3' : 0 ,
#     'Soil_Type21' : 0 ,
#     'Soil_Type10' : 0 ,
#     'Soil_Type35' : 0 ,
#     'Vertical_Distance_To_Hydrology_binned' : 0 ,
#     'Soil_Type28' : 0 ,
#     'Aspect_binned' : 0 ,
#     'Soil_Type6' : 0 ,
#     'Hillshade_9am_binned' : 1 ,
#     'Soil_Type14' : 0 ,
#     'Soil_Type1' : 0 ,
#     'Slope_binned' : 0 ,
#     'Soil_Type26' : 0 ,
#     'Soil_Type19' : 0 ,
#     'Hillshade_3pm_binned' : 1 ,
#     'Soil_Type9' : 0 ,
#     'Soil_Type5' : 0 ,
#     'Soil_Type18' : 0 ,
#     'Soil_Type36' : 0 ,
#     'Soil_Type37' : 0 ,
#     'Soil_Type25' : 0 ,
#     'Soil_Type8' : 0 ,
#     'Soil_Type7' : 0 ,
#     'Soil_Type15' : 0 ,
# }

# # The range in number-of-features-per-model we want to evaluate
# iterations_num_feats = [
#                         1,
#                         2,
#                         3,
#                         4,
#                         5,
#                         6,
#                         7,
#                         8,
#                         9,
#                         10
#                         ]

# # Selects the feature to split populations on as well as the values of that feature defining the source and target populations:
# # (split_col, source, target)
# source_target_zones = [
#     ('Elevation_binned', 0, 3),
#     ('Elevation_binned', 1, 2),
#     ('Hillshade_Noon_binned', 0, 3),
#     ('Hillshade_Noon_binned', 1, 2),
#     ('Soil_Type2', 0, 1),
#     ('Soil_Type4', 0, 1),
#     # ('Soil_Type12', 0, 1),
#     # ('Soil_Type16', 0, 1),
#     # ('Wilderness_Area3', 0, 1),
#     # ('Wilderness_Area1', 0, 1),
#     # ('Aspect_binned', 0, 3),
#     # ('Aspect_binned', 1, 2),
#     # ('Horizontal_Distance_To_Roadways_binned', 0, 3),
#     # ('Horizontal_Distance_To_Roadways_binned', 1, 2),
#     # ('Wilderness_Area4', 0, 1),
#     # ('Wilderness_Area4', 1, 0)
# ]

# simple_output_filename = "Results/simple_results_lodgepolebin.csv"
# soph_xgb_filename = "Results/xgb_lodgepolebin"
# xgb_ratio_filename = "Results/xgb_ratio_lodgepolebin"
# one_split_simple_filename = "Results/single_split_lodgepolebin.csv"

# data_filepath = "Datasets/covtype_lodgepole_binned.csv"
# label_col = 'Lodgepole'

# def df_fetch_and_cleanup():
#     fp = data_filepath
#     df = pd.read_csv(fp, dtype=int)

#     return df

# --- END BINNED LODGEPOLE COVTYPE SETUP ---

# --- BEGIN DIABETES SETUP ---

# Default values are needed for the simple single-variable models
defaults = {
    'HighBP': 1,
    'HighChol': 1,
    'CholCheck': 1,
    'Smoker': 0,
    'Stroke': 0,
    'HeartDiseaseorAttack': 0,
    'PhysActivity': 1,
    'Fruits': 1,
    'Veggies': 1,
    'HvyAlcoholConsump': 0,
    'AnyHealthcare': 1,
    'NoDocbcCost': 0,
    'GenHlth': 3,
    'DiffWalk': 0,
    'Sex': 0,
    'BMI_binned': 1,
    'MentHlth_binned': 0,
    'PhysHlth_binned': 0,
    'Age_binned': 0,
    'Education_binned': 2,
    'Income_binned': 2,
}

# The range in number-of-features-per-model we want to evaluate
iterations_num_feats = [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10
                        ]

# Selects the feature to split populations on as well as the values of that feature defining the source and target populations:
# (split_col, source, target)
source_target_zones = [
    ('HighChol', 0, 1),
    ('BMI_binned', 0, 3),
    ('BMI_binned', 1, 2),
    ('Sex', 0, 1),
    ('HighBP', 0, 1),
    ('HeartDiseaseorAttack', 0, 1),
    ('HvyAlcoholConsump', 0, 1),
    ('GenHlth', 1, 5),
    ('GenHlth', 3, 4),
    ('DiffWalk', 0, 1),
    ('Age_binned', 0, 3),
    ('Smoker', 0, 1),
    ('AnyHealthcare', 0, 1),
    # ('HighBP', 0, 1),
    # ('HighBP', 1, 0),
    # ('Soil_Type12', 0, 1),
    # ('Soil_Type16', 0, 1),
    # ('Wilderness_Area3', 0, 1),
    # ('Wilderness_Area1', 0, 1),
    # ('Aspect_binned', 0, 3),
    # ('Aspect_binned', 1, 2),
    # ('Horizontal_Distance_To_Roadways_binned', 0, 3),
    # ('Horizontal_Distance_To_Roadways_binned', 1, 2),
    # ('Wilderness_Area4', 0, 1),
    # ('Wilderness_Area4', 1, 0)
]

simple_output_filename = "Results/simple_results_diabetes.csv"
soph_xgb_filename = "Results/xgb_diabetes"
xgb_ratio_filename = "Results/xgb_ratio_diabetes"
one_split_simple_filename = "Results/single_split_diabetes.csv"

data_filepath = "Datasets/diabetes_binned.csv"
label_col = 'Diabetes_binary'

def df_fetch_and_cleanup():
    fp = data_filepath
    df = pd.read_csv(fp, dtype=int)

    return df