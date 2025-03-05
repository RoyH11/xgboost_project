"""
File: main.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Main script for running the model training and evaluation pipeline.
"""

from data_processing import *
from model_training import *
from utils import *
import json
import time

def main():
    """
    Main excution script for XGBoost tuning. 
    """

    if not pre_run_check():
        return

    welcome_message()

    # Load the dataset
    df = load_dataset()

    # Select machine region
    features, health_conditions, recommended_split = select_machine_region(df)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(features, health_conditions, recommended_split)

    if not start_or_quit():
        return
    
    timestamp = create_timestamp()

    # Train model using manual grid search with cross-validation
    best_params, best_score = manual_grid_search(X_train, y_train)

    # Save the best hyperparameters and score to a JSON file
    best_params_file_path = save_hyperparameters(best_params, best_score, timestamp)

    # Train the final model using the best hyperparameters
    final_model = train_final_model(X_train, X_val, y_train, y_val, best_params)

    # Save the final model
    final_model_file_path = save_trained_model(final_model, timestamp)

    # Message to user
    training_complete_message(best_params_file_path, final_model_file_path)


if __name__ == "__main__":
    main()