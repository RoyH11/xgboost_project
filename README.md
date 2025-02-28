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
### **1. Place 

---

## Future Improvements
- Automate hyperparameter tuning using Optuna.
- Implement logging instead of print statements.
- Add support for feature importance visualization.

---

## Author
**Roy Huang**  
📧 ruoqiuhuang@gmail.com

