"""
File: config.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Configuration file containing hyperparameter settings and dataset paths 
for XGBoost model tuning.
"""

import os

# ---------------------------------------------------------------------
# Get the root directory of the project, and go back one level        #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#
                                                                      #
# Parameters foler                                                    #
PARAMETERS_FOLDER = os.path.join(ROOT_DIR, "hyperparameters")         #
                                                                      #
# Models folder                                                       #
MODELS_FOLDER = os.path.join(ROOT_DIR, "saved_models")                #
                                                                      #
# ---------------------------------------------------------------------

# color fundus photography (CFP) dataset                                
CSV_PATH = r"data\new_cfp_features_fully_labelled.csv"

# optical coherence tomography (OCT) dataset                        
# CSV_PATH = r"data\RetFound_LF_all_OCT_fully_labelled.csv"

# ---------------------------------------------------------------------
CSV_PATH = os.path.join(ROOT_DIR, CSV_PATH)                           #
# ---------------------------------------------------------------------
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
