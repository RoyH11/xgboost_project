"""
File: config.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Configuration file containing hyperparameter settings and dataset paths 
for XGBoost model tuning.
"""

# Data paths
# color fundus photography (CFP) dataset
CSV_PATH = r"C:\Users\S237442\Documents\GitHub\xgboost_project\data\RetFound_LF_all_Automorph_with_fully_labelled.csv"

# # optical coherence tomography (OCT) dataset
# CSV_PATH = "L:\\Lab\\Roy\\AI-READi-Roy\\60_RETFound_stuff\\60_cfp_all\\30_XGBoost\\RETFound_LF_all_OCT_fully_labelled.csv"

# Parameters foler
PARAMETERS_FOLDER = r"C:\Users\S237442\Documents\GitHub\xgboost_project\hyperparameters"

# Model folder
MODEL_FOLDER = r"C:\Users\S237442\Documents\GitHub\xgboost_project\saved_models"

# Comparison setting
BINARY = True  # False for all 4, True for healthy vs unhealthy

# Machine region selection
MACHINE_REGION = "maestro2_3d_macula"

# Hyperparameter tuning phases

# Phase 1
LEARNING_RATE = 0.1  # 0.01 - 0.3
NUM_ROUND = 100 # 100 - 1000

# Phase 2
MAX_DEPTH = [3, 9] #6 # 3 - 10
MIN_CHILD_WEIGHT = [1, 10] # 1 # 1 - 10
GAMMA = [0, 2] # 0 # 0 - 5

# Phase 3
REG_LAMBDA = 1 # 1 - 10
REG_ALPHA = 0 # 0 - 10

# Phase 4
SUBSAMPLE = 1.0 # 0.5 - 1.0
COLSAMPLE_BYTREE = 1.0 # 0.5 - 1.0
COLSAMPLE_BYLEVEL = 1.0 # 0.5 - 1.0
