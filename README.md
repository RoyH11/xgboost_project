# XGBoost Model Training and Hyperparameter Tuning

## Overview
Originally developed for **diabetic retinopathy detection** using the [AI-READI](https://aireadi.org/dataset) dataset. The original CSV files contain latent features extracted from fundus images using [RETFound](https://github.com/rmaphoh/RETFound_MAE), with [AutoMorph](https://github.com/rmaphoh/AutoMorph) preprocessing applied beforehand.

While designed for this specific task, the pipeline can be easily adapted for other **binary or multi-class classification** problems by modifying the dataset preprocessing steps.


## Features
- **End-to-End XGBoost Integration**
    > Fully optimized for XGBoost, leveraging its built-in GPU acceleration.
- **GPU-Powered Grid Search**
    > Unlike sklearn's CPU-bound grid search, this pipeline uses XGBoost's native GPU support for faster hyperparameter tuning, significantly reducing training time on large datasets.
- **Early Stopping**
    > Prevents overfitting by monitoring validation performance
- **Model Persistence**
    > Automatically saves the best model and its hyperparameters for future use
- **Customizable Hyperparameters**
    > Easily adjust hyperparameter search ranges in the `config.py` file


## Installation
### **1. Set Up Conda Environment**
```bash
conda create --name xgboost_env python=3.9 -y
conda activate xgboost_env
```

### **2. Install Dependencies**
```bash
conda install -c conda-forge xgboost scikit-learn pandas numpy
```
*If any package is unavailable via Conda, use pip:*
```bash
pip install <package-name>
```


## Project Structure
```
.
├── src/
│   ├── main.py              # Main script to run the training pipeline
│   ├── model_training.py    # Functions for model training and hyperparameter tuning
│   ├── data_processing.py   # Dataset loading and preprocessing
│   ├── utils.py             # Utility functions (saving models, creating timestamps, etc.)
│   ├── config.py            # Configuration file (paths, hyperparameters)
├── data/                    # Dataset directory
├── hyperparameters/         # Stores best hyperparameters found during tuning
├── saved_models/            # Stores trained XGBoost models
├── README.md                # Project documentation (this file)
```


## Usage
### **1. Prepare the Dataset**
- Create a `data` directory in the project root
    ```bash
    mkdir data
    ```

- Place dataset CSV files in the `data` directory. The dataset should contain the columns 
`machine_region`, `study_group`, and `feature_0` to `feature_n`.

- Modify `config.py` to specify the dataset path:
    ```python
    CSV_PATH = r"data\your_dataset.csv"
    ```

### **2. Customize Hyperparameters**
Modify `config.py` to adjust hyperparameter search ranges:
```python
# Data paths
# color fundus photography (CFP) dataset
CSV_PATH = r"C:\Users\S237442\Documents\GitHub\xgboost_project\data\RetFound_LF_all_Automorph_with_fully_labelled.csv"


# Parameters foler
PARAMETERS_FOLDER = r"C:\Users\S237442\Documents\GitHub\xgboost_project\hyperparameters"

# Models folder
MODELS_FOLDER = r"C:\Users\S237442\Documents\GitHub\xgboost_project\saved_models"

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
```

### **3. Run the Training Pipeline**
```bash
python src/main.py
```
This script:
1. Loads the dataset
2. Splits it into training, validation, and test sets
3. Runs **manual grid search** for hyperparameter tuning
4. Saves the best hyperparameters
5. Trains the final model using the best parameters
6. Saves the trained model



---

## Future Improvements
- Automate hyperparameter tuning using Optuna.
- Implement logging instead of print statements.
- Add support for feature importance visualization.

---

## Author
**Roy Huang**  
📧 ruoqiuhuang@gmail.com

