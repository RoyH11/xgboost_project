import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import time
# import cupy as cp

CFP_CSV_PATH = r"RetFound_LF_all_Automorph_with_fully_labelled.csv"

OCT_CSV_PATH = r"L:\Lab\Roy\AI-READi-Roy\60_RETFound_stuff\60_cfp_all\30_XGBoost\RETFound_LF_all_OCT_fully_labelled.csv"

# False for all 4, True for health vs unhealthy
COMPARE = True 

MACHINE_REGION = "maestro2_3d_macula"

# defalut parameters
# ---------------------------------

# phase 1

# 0.01 - 0.3
LEARNING_RATE = 0.1 # [0.01, 0.1, 0.3]
# 100 - 1000
NUM_ROUND = 100 # [100, 500, 1000]
# ---------------------------------

# phase 2

# 3 - 10
MAX_DEPTH = 6 
# 1 - 10
MIN_CHILD_WEIGHT = 1 
# 0 - 5
GAMMA = 0 
# ---------------------------------

# phase 3

# 1 - 10
REG_LAMBDA = 1 
# 0 - 10
REG_ALPHA = 0 
# ---------------------------------

# phase 4

# 0.5 - 1.0
SUBSAMPLE = 1.0 
COLSAMPLE_BYTREE = 1.0 
COLSAMPLE_BYLEVEL = 1.0 
# ---------------------------------



# welcome message
def welcome_message():
    print("-------------------------------------")
    print("Welcome to the XGBoost Tuning Script!")
    print("-------------------------------------")



# Select machine region
def hard_code_machine_region(df):

    # get the subset of the df that corresponds to the selected machine region
    df_machine_region = df[df['machine_region'] == MACHINE_REGION]

    #print(df_machine_region.shape)

    # get the columns that contains "feature_"
    features_columns = df_machine_region.columns[df_machine_region.columns.str.contains("feature_")]

    # get the features and health conditions
    features = df_machine_region[features_columns]
    health_conditions = df_machine_region['study_group']

    print(f"User selected: {MACHINE_REGION}")
    print()
    # wait 0.1 seconds
    time.sleep(0.1)
    print(f"Features shape: {features.shape}")
    print(f"Health conditions shape: {health_conditions.shape}")

    return features, health_conditions



# Prepare data for xgboost
def prepare_data(features, health_conditions, health_condition_index):

    if health_condition_index:
        # change all conditions to unhealthy, if not healthy
        health_conditions = health_conditions.apply(lambda x: "unhealthy" if x != "healthy" else x)

    # encode the health conditions
    le = LabelEncoder()
    health_conditions = le.fit_transform(health_conditions)

    # split the data into training, validation, and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, health_conditions, test_size=0.2, random_state=42, stratify=health_conditions)

    return X_train, X_test, y_train, y_test



# Start or quit the program
def start_or_quit():
    print()
    time.sleep(0.1)
    print("Last chance to exit the program!")
    print("---------------------------------")
    time.sleep(0.1)
    print("1: I am ready, just do it!")
    print("0: I am not ready, quit the program")
    print("---------------------------------")
    # ask user to select a tuning phase
    start_bool = int(input("Selection: "))
    time.sleep(0.1)
    print("User selected: ", start_bool)
    print()

    return start_bool

    

# Train model
def train_model(X_train, X_test, y_train, y_test):

    def to_list(param): 
        if isinstance(param, list):
            return param
        else:
            return [param]

    fixed_params = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda'
    }

    param_grid = {
        'learning_rate': to_list(LEARNING_RATE),
        'n_estimators': to_list(NUM_ROUND),

        'max_depth': to_list(MAX_DEPTH),
        'min_child_weight': to_list(MIN_CHILD_WEIGHT),
        'gamma': to_list(GAMMA),

        'reg_lambda': to_list(REG_LAMBDA),
        'reg_alpha': to_list(REG_ALPHA),

        'subsample': to_list(SUBSAMPLE),
        'colsample_bytree': to_list(COLSAMPLE_BYTREE),
        'colsample_bylevel': to_list(COLSAMPLE_BYLEVEL)
    }
    
    # # move the data to GPU
    # X_train = cp.array(X_train)
    # X_test = cp.array(X_test)
    # y_train = cp.array(y_train)
    # y_test = cp.array(y_test)

    xgb_model = XGBClassifier(**fixed_params)

    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3, # 3-fold cross-validation
        verbose=1, # print the progress
        # n_jobs=-1 # use all processors
    )

    grid.fit(X_train, y_train)

    print("Best parameters found: ", grid.best_params_)
    print("Best accuracy found: ", grid.best_score_)

    return grid.best_params_, grid.best_score_


# main function
def main(): 
    # welcome message
    welcome_message()

    # load dataset
    df = pd.read_csv(CFP_CSV_PATH)

    # # select machine region
    # features, health_conditions = select_machine_region(df)

    # hard code machine region
    features, health_conditions = hard_code_machine_region(df)

    # select health condition
    health_condition_index = COMPARE # False for all 4, True for health vs all other

    # prepare data for xgboost
    X_train, X_test, y_train, y_test = prepare_data(features, health_conditions, health_condition_index)

    # last chance to exit the program
    start_bool = start_or_quit()

    if start_bool == 0:
        print("Exiting the program...")
        return

    # train model
    best_params, best_score = train_model(X_train, X_test, y_train, y_test)



if __name__ == "__main__":
    main()