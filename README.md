# Mortality Risk Prediction - README

## Overview
This project predicts ICU patient mortality risk using machine learning models (Logistic Regression and XGBoost). The code includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Required Python libraries (see below)

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install the following manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```

## Dataset
Place the dataset file `ICU_Patient_Monitoring_Mortality.csv` in the `data/` directory.
The dataset can also be found here: https://www.kaggle.com/datasets/jayjoshi37/icu-patient-monitoring-and-mortality-prediction/data

## Running the Code
1. Open a terminal and navigate to the `src/` directory:
   ```bash
   cd src
   ```

2. Run the script:
   ```bash
   python ICU_mortality_ML.py
   ```

3. Outputs will be saved in the `outputs/` directory:
   - Plots (e.g., `feature_distributions.png`, `correlation_matrix.png`)
   - Model performance metrics (e.g., `model_summary.csv`)

## Key Outputs
- **Plots**: Visualizations for data analysis and model evaluation.
- **Model Summary**: A CSV file summarizing the performance of Logistic Regression and XGBoost models.

## Notes
- Ensure the `data/` and `outputs/` directories exist before running the script.
- The script automatically handles missing values and performs stratified train-test splits.

## Authors
Rafes Divas (Trent Levy, Nehemyah Green, Ledi Anggara)