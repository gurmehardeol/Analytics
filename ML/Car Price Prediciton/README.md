# Car Price Prediction - Machine Learning Pipeline

A comprehensive, modular machine learning pipeline for predicting car prices using multiple regression models. The pipeline is fully configurable via a central YAML configuration file.

## 📋 Project Structure

```
Car Price Prediciton/
├── car_price_prediction_with_missing.csv  # Raw dataset
├── config.yaml                             # Central configuration file
├── eda.py                                  # Exploratory Data Analysis script
├── preprocess.py                           # Data preprocessing script
├── modeling.py                             # ML modeling script
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── data/                                   # Shared data files (used across scripts)
│   └── processed_data.pkl                  # Processed dataset for modeling
└── outputs/                                # All script outputs
    ├── eda/                                # EDA output folder
    │   └── *.png                           # All EDA visualizations
    ├── preprocessing/                      # Preprocessing output folder
    │   ├── processed_data_inspection.csv   # CSV for manual inspection
    │   └── preprocessing_objects.pkl       # Encoders and scalers
    └── modeling/                           # Modeling output folder
        ├── models/                         # Trained models
        │   ├── linear_regression_model.pkl
        │   ├── random_forest_model.pkl
        │   └── xgboost_model.pkl
        └── plots/                          # Modeling visualizations
            └── *.png
```

## 🎯 Features

### 1. Exploratory Data Analysis (eda.py)
- **Comprehensive visualizations** using Seaborn and Matplotlib
- **Distribution analysis** for all features and target variable
- **Correlation heatmap** for numeric features
- **Feature vs Target plots** (scatter plots for numeric, box plots for categorical)
- **Automated plot generation** saved to `/plots` directory

### 2. Data Preprocessing (preprocess.py)
- **Config-driven preprocessing** - all parameters controlled via config.yaml
- **Missing value handling** with multiple strategies (mean, median, mode, drop)
- **Feature engineering** (e.g., car age from year)
- **Categorical encoding** (one-hot, label encoding)
- **Feature scaling** (StandardScaler, MinMaxScaler, RobustScaler)
- **Outlier detection and handling** (IQR method, capping/removal)
- **Saves processed data** as both pickle and CSV

### 3. Machine Learning Modeling (modeling.py)
- **Multiple models**:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor (optional)
- **Feature importance analysis** with visualizations
- **Decision tree visualization** from Random Forest
- **Comprehensive evaluation metrics**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (R-squared)
  - MAPE (Mean Absolute Percentage Error)
- **Cross-validation** for robust performance estimation
- **Actual vs Predicted plots** and residual analysis
- **Model comparison** and automatic best model selection
- **Saves trained models** using joblib

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyyaml joblib
```

### Optional (for XGBoost):
```bash
pip install xgboost
```

## 🚀 Usage

### Quick Start

Run the complete pipeline in sequence:

```bash
# 1. Exploratory Data Analysis
python eda.py

# 2. Data Preprocessing
python preprocess.py

# 3. Model Training and Evaluation
python modeling.py
```

### Configuration

All pipeline behavior is controlled via [config.yaml](config.yaml). Key configuration sections:

#### Data Paths
```yaml
data:
  raw_data: "car_price_prediction_with_missing.csv"
  processed_data: "data/processed_data.pkl"
```

#### Preprocessing Options
```yaml
preprocessing:
  missing_values:
    numeric_strategy: "median"      # mean, median, mode, drop
    categorical_strategy: "mode"    # mode, constant, drop
  
  encoding:
    method: "onehot"                # onehot, label, target
  
  scaling:
    method: "standard"              # standard, minmax, robust, none
  
  outliers:
    method: "iqr"                   # iqr, zscore, none
    handle: "cap"                   # cap, remove, none
```

#### Model Configuration
```yaml
modeling:
  models:
    linear_regression:
      enabled: true
    
    random_forest:
      enabled: true
      params:
        n_estimators: 100
        max_depth: 10
    
    xgboost:
      enabled: true
      params:
        n_estimators: 100
        learning_rate: 0.1
```

## 📊 Dataset

### Features
- **Brand**: Car manufacturer (Tesla, BMW, Audi, Ford, Honda, Toyota, Mercedes)
- **Model**: Specific car model
- **Year**: Manufacturing year
- **Engine Size**: Engine displacement in liters
- **Mileage**: Total distance traveled
- **Fuel Type**: Petrol, Diesel, Electric, Hybrid
- **Transmission**: Manual or Automatic
- **Condition**: New, Used, Like New

### Target Variable
- **Price**: Car sale price (continuous)

## 📈 Output Files

### EDA Outputs (`/outputs/eda/`)
All exploratory data analysis visualizations:
- `target_distribution.png` - Price distribution histogram and boxplot
- `numeric_distributions.png` - All numeric feature distributions
- `correlation_heatmap.png` - Feature correlation matrix
- `feature_correlations.png` - Feature correlation with target
- `*_distribution.png` - Individual categorical feature distributions
- `*_vs_price.png` - Scatter plots for numeric features vs price
- `*_vs_price_boxplot.png` - Box plots for categorical features vs price

### Preprocessing Outputs
**Shared Data** (`/data/` - used by modeling.py):
- `processed_data.pkl` - Complete processed dataset with encoders and scaler

**Preprocessing-specific** (`/outputs/preprocessing/`):
- `processed_data_inspection.csv` - Processed data in CSV format for manual inspection
- `preprocessing_objects.pkl` - Saved encoders and scaler for deployment

### Modeling Outputs (`/outputs/modeling/`)

**Models** (`/outputs/modeling/models/`):
- `linear_regression_model.pkl` - Trained Linear Regression model
- `random_forest_model.pkl` - Trained Random Forest model
- `xgboost_model.pkl` - Trained XGBoost model (if enabled)

**Plots** (`/outputs/modeling/plots/`):
- `linear_regression_coefficients.png` - LR coefficient importance
- `*_feature_importance.png` - Tree-based model feature importances
- `decision_tree_visualization.png` - Single tree from Random Forest
- `predictions_comparison.png` - Actual vs predicted for all models
- `residuals_plot.png` - Residual analysis for all models

## 🧪 Model Performance

The pipeline automatically compares all enabled models and displays:
- Training and testing metrics for each model
- Cross-validation scores
- Model comparison table
- Best performing model identification

Example output:
```
MODEL COMPARISON
================================================================================
              Model  Test MAE  Test RMSE  Test R2  Test MAPE
  Linear Regression   12500.5    15234.2    0.825      18.3
      Random Forest    8234.1    10456.7    0.912      12.1
            XGBoost    7845.3     9876.4    0.928      11.5

================================================================================
Best Model (by R2): XGBoost
================================================================================
```

## 🔧 Customization

### Adding New Features
1. Update `config.yaml` to include new features in appropriate categories
2. Modify preprocessing steps if needed in `preprocess.py`
3. Re-run the pipeline

### Adding New Models
1. Import the model in `modeling.py`
2. Add model configuration to `config.yaml`
3. Create a training function following the existing pattern
4. Add to the main pipeline execution

### Changing Preprocessing Steps
Edit `config.yaml` to modify:
- Missing value strategies
- Encoding methods
- Scaling techniques
- Outlier handling approaches

No code changes required!

## 📝 Code Quality

All scripts follow best practices:
- ✅ **Modular functions** with clear responsibilities
- ✅ **Comprehensive docstrings** for all functions
- ✅ **Type hints** where applicable
- ✅ **Error handling** and informative logging
- ✅ **Config-driven** - no hardcoded parameters
- ✅ **Clean output** with progress indicators
- ✅ **Ready for testing** - modular design

## 🤝 Contributing

To extend this pipeline:
1. Add new functions following existing patterns
2. Update configuration file with new parameters
3. Document new features in this README
4. Test thoroughly with the existing dataset

## 📄 License

This project is created for educational and demonstration purposes.

## 🙋 Support

For issues or questions:
1. Check the configuration file for parameter options
2. Review function docstrings for detailed documentation
3. Examine console output for detailed execution logs

---

**Note**: This pipeline is designed to be flexible and extensible. All major parameters are controlled through the configuration file, making it easy to experiment with different preprocessing and modeling strategies without modifying code.
