"""
File: model_training.py
Author: Roy Huang
Email: ruoqiuhuang@gmial.com
Date: 2025-02-27
Description: Implements a grid search for hyperparameter tuning and model training 
on an XGBoost model with GPU support.
"""

import numpy as np
import itertools
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from config import *
from utils import save_trained_model, copy_and_update_params


def to_list(param):
    """
    Ensure that the parameter is a list.
    """
    if isinstance(param, list):
        return param
    else:
        return [param]
    

def generate_hyperparameter_combinations():
    """
    Generates all possible combinations of hyperparameters.
    """
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
    return [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]


def cross_validation_model(X_train, y_train, params, cv_folds=3): 
    """
    Performs k-fold cross-validation and return the mean accuracy.
    """
    # Convert the data into DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Update params for binary/multi-class classification
    updates = {
        'objective': 'binary:logistic' if BINARY else 'multi:softmax',
        'num_class': None if BINARY else len(np.unique(y_train)),
        'eval_metric': ['logloss', 'auc'] if BINARY else ['mlogloss', 'merror'],
        'tree_method': 'hist',
        'device': 'cuda'
    }
    params = copy_and_update_params(params, updates)

    # Perform k-fold cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=params['n_estimators'],
        nfold=cv_folds,
        stratified=True,
        early_stopping_rounds=10,
        metrics=params['eval_metric'],
        seed=42
    )

    # Extract the final results
    if BINARY:
        final_logloss = cv_results['test-logloss-mean'].iloc[-1]
        final_auc = cv_results['test-auc-mean'].iloc[-1]
        return final_logloss, final_auc
    else:
        final_mlogloss = cv_results['test-mlogloss-mean'].iloc[-1]
        final_merror = cv_results['test-merror-mean'].iloc[-1]
        return final_mlogloss, final_merror
    

def manual_grid_search(X_train, y_train, cv_folds=3):
    """
    Performs a manual grid search for hyperparameter tuning.
    """
    # Generate all possible hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations()

    best_score = -float('inf') if BINARY else float('inf') # higher auc is better, lower mlogloss is better
    best_params = None

    for params in hyperparameter_combinations: 
        print(f"Evaluating parameters: {params}")

        # Perform k-fold cross-validation
        if BINARY: 
            mean_logloss, mean_auc = cross_validation_model(X_train, y_train, params, cv_folds)
            print(f"Mean CV Log Loss: {mean_logloss:.4f}, Mean CV AUC: {mean_auc:.4f}")

            # Select best on the highest AUC
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params

        else:
            mean_mlogloss, mean_merror = cross_validation_model(X_train, y_train, params, cv_folds)
            print(f"Mean CV Multi-Class Log Loss: {mean_mlogloss:.4f}, Mean CV Multi-Class Error: {mean_merror:.4f}")

            # Select best on the lowest mlogloss
            if mean_mlogloss < best_score:
                best_score = mean_mlogloss
                best_params = params

    print("\nBest parameters found:", best_params)
    print("Best cross-validation score:", best_score)

    return best_params, best_score


def train_final_model(X_train, X_val, y_train, y_val, best_params, timestamp):
    """
    Train the final model using the best hyperparameters.
    """
    print("\nTraining final model with best hyperparameters...")

    # Update the best_params
    updates = {
        'objective': 'binary:logistic' if BINARY else 'multi:softmax',
        'num_class': None if BINARY else len(np.unique(y_train)),
        'eval_metric': ['logloss', 'auc'] if BINARY else ['mlogloss', 'merror'],
        'tree_method': 'hist',
        'device': 'cuda', 
        'early_stopping_rounds': 10
    }
    best_params = copy_and_update_params(best_params, updates)

    # Define the model
    final_model = xgb.XGBClassifier(
        **best_params
    )

    # Fit the model
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Get the best iteration
    best_iteration = final_model.best_iteration

    # retrain the model with the best iteration
    if best_iteration is not None:
        print(f"\nRetraining the model with the best iteration: {best_iteration}")
        final_model.n_estimators = best_iteration
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

    model_filename = save_trained_model(final_model, timestamp)

    print(f"\nFinal model trained and saved to '{model_filename}'.")

    return model_filename