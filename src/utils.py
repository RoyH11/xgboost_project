"""
File: utils.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Utility functions for user interaction
"""

import time 
import json
from config import *
import os


def pre_run_check():
    """
    Check if the parameters and models folders exist.
    Check if data path is valid.
    """
    if not os.path.exists(PARAMETERS_FOLDER):
        os.makedirs(PARAMETERS_FOLDER)

    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if not os.path.exists(CSV_PATH):
        print(f"Data path '{CSV_PATH}' does not exist.")
        print("Exiting...")
        return False
    
    return True

def welcome_message():
    """
    Display a welcome message to the user.
    """
    print("-------------------------------------")
    print("Welcome to the XGBoost Tuning Script!")
    print("-------------------------------------")
    time.sleep(0.1)


def start_or_quit(): 
    """
    Prompt the user to start or quit the script.
    """
    print("\nLast chance to exit the program!")
    print("-------------------------------------")
    time.sleep(0.1)
    print("1: I am ready, just do it!")
    print("0: Exit")
    print("-------------------------------------")

    try: 
        start_int = int(input("Selction: "))
    except ValueError:
        print("Invalid input, exiting...")
        return False
    
    if start_int != 1:
        print("\nExiting...")
        return False

    print("\nOK, let's GO!\n")
    return True


def create_timestamp():
    """
    Create a timestamp for saving files.
    """
    return time.strftime("%Y%m%d-%H%M%S")


def copy_and_update_params(params, updates):
    """
    Copy the parameters and update them with new values.
    """
    params_copy = params.copy()
    params_copy.update(updates)
    return params_copy


def save_hyperparameters(best_params, best_score, timestamp):
    """
    Save the best hyperparameters and score to a JSON file.
    """
    filename = f"best_hyperparameters_{timestamp}.json"

    file_path = os.path.join(PARAMETERS_FOLDER, filename)

    updates = {"best_score": best_score}
    best_params = copy_and_update_params(best_params, updates)

    with open(file_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\nBest hyperparameters saved to '{file_path}'.")

    return file_path


def save_trained_model(model, timestamp): 
    """
    Save the trained model to a file.
    """
    filename = f"xgboost_model_{timestamp}.json"

    file_path = os.path.join(MODELS_FOLDER, filename)

    model.save_model(file_path)

    print(f"\nTrained model saved to '{file_path}'.")

    return file_path


def training_complete_message(best_params_file_path, final_model_file_path):
    """
    Display a message to indicate that the training is complete.
    """
    print("\nTraining complete!")
    print("-------------------------------------")
    print(f"Best hyperparameters saved to {best_params_file_path}")
    print(f"Trained model saved to {final_model_file_path}")
    print("-------------------------------------")
    print("Goodbye!")