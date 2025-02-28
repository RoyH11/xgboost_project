# XGBoost Model Training and Hyperparameter Tuning (GPU-Optimized)

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
- Modify `config.py` to adjust hyperparameter search ranges:
    
    Select binary option and machine region
    ```python
    # Comparison setting
    BINARY = True  # False for all 4, True for healthy vs unhealthy

    # Machine region selection
    MACHINE_REGION = "maestro2_3d_macula"
    ```

- During each phase, specify the hyperparameter values as 
  either **a single value** or **a list of discrete options**, not as a continuous range. 
  The program will **only test the values you provide**—it does **not** 
  automatically search within a range.

    > For example, `LEARNING_RATE = 0.1` will only test a learning rate of 0.1,
    while `LEARNING_RATE = [0.01, 0.1, 0.3]` will test learning rates of 0.01, 0.1, and 0.3.

    ✅ Correct example (with recommended values): 
    ```python
    # Phase 1
    LEARNING_RATE = 0.1  # 0.01 - 0.3
    NUM_ROUND = 100 # 100 - 1000

    # Phase 2
    MAX_DEPTH = [3, 9] # 3 - 10
    MIN_CHILD_WEIGHT = [1, 5, 10] # 1 - 10
    GAMMA = [0, 2] # 0 - 5

    # Phase 3
    REG_LAMBDA = 1 # 1 - 10
    REG_ALPHA = 0 # 0 - 10

    # Phase 4
    SUBSAMPLE = 1.0 # 0.5 - 1.0
    COLSAMPLE_BYTREE = 1.0 # 0.5 - 1.0
    COLSAMPLE_BYLEVEL = 1.0 # 0.5 - 1.0
    ```

    > [!TIP]
    > Make sure to document the hyperparameters you tested each time you run the pipeline.

### **3. Run the Training Pipeline**
- In the project root, run the main script:
    ```bash
    python src/main.py
    ```

- Confirm you are ready: enter `1` to proceed or `0` to exit
    ```console
    Last chance to exit the program!
    -------------------------------------
    1: I am ready, just do it!
    0: Exit
    -------------------------------------
    Selction: 
    ```

- The pipeline will run through the following steps: 
    1. Loads the dataset
    2. Splits it into training, validation, and test sets
    3. Runs **manual grid search** for hyperparameter tuning
    4. Saves the best hyperparameters
    5. Trains the final model using the best parameters
    6. Saves the trained model

- After training: 
    ```console
    Training complete!
    -------------------------------------
    Best hyperparameters saved to <path_to_hyperparameters_file>
    Trained model saved to <path_to_saved_model>
    -------------------------------------
    Goodbye!
    ```
    The best hyperparameters and trained model are saved in the `hyperparameters` and `saved_models` directories, respectively. The file names include the timestamp of the training session.

### **4. Repeat [Step 2](#2-customize-hyperparameters) and [Step 3](#3-run-the-training-pipeline) for Further Hyperparameter Tuning**

### **5. Load the Best Model for Testing**
- Under construction


## Future Improvements
- ROC-AUC display
- Testing mode 


## Author
**Roy Huang**  
📧 ruoqiuhuang@gmail.com

