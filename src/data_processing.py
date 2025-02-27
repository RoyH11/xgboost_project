"""
File: data_processing.py
Author: Roy Huang
Date: 2025-02-27
Description: Module for loading datasets, selecting machine regions, and preparing data for XGBoost model tuning.
"""

import pandas as pd
import time 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import CSV_PATH, MACHINE_REGION, COMPARE

def load_data():
    """
    Load the dataset from the specified CSV file.
    """
    return pd.read_csv(CSV_PATH)

def select_machine_region(df):
    """
    Select the machine region from the dataset.
    """
    df_machine_region = df[df["machine_region"] == MACHINE_REGION]
    features_columns = df_machine_region.columns[df_machine_region.columns.str.startswith('feature_')]

    features = df_machine_region[features_columns]
    health_conditions = df_machine_region["study_group"]

    print(f"User selected: {MACHINE_REGION}\n")
    time.sleep(0.1)
    print(f"Features shape: {features.shape}")
    print(f"Health conditions shape: {health_conditions.shape}")

    return features, health_conditions

def prepare_data(features, health_conditions): 
    """
    Encode labels and splits data into train/test sets.
    """
    if COMPARE:
        health_conditions = health_conditions.apply(lambda x: "unhealthy" if x != "healthy" else x)

    le = LabelEncoder()
    health_conditions = le.fit_transform(health_conditions)

    X_train, X_test, y_train, y_test = train_test_split(
        features, health_conditions, test_size=0.2, random_state=42, stratify=health_conditions
    )

    return X_train, X_test, y_train, y_test