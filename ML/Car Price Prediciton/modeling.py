"""
Machine Learning Modeling for Car Price Prediction

This script trains and evaluates multiple ML models,
performs feature importance analysis, and saves trained models.
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_processed_data(config):
    """
    Load preprocessed data from pickle file.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (X, y, feature_names, preprocessing_objects)
    """
    print("\n" + "="*80)
    print("LOADING PROCESSED DATA")
    print("="*80)
    
    data_path = config['data']['processed_data']
    
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    X = processed_data['X']
    y = processed_data['y']
    feature_names = processed_data['feature_names']
    
    print(f"Data loaded successfully")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, feature_names


def create_output_directories(config):
    """
    Create output directories for models and plots.
    
    Args:
        config (dict): Configuration dictionary
    """
    modeling_output_dir = config['directories']['modeling_output']
    os.makedirs(modeling_output_dir, exist_ok=True)
    # Create subdirectories for organization
    os.makedirs(os.path.join(modeling_output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(modeling_output_dir, 'plots'), exist_ok=True)
    print("Modeling output directories created/verified")


def split_data(X, y, config):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    split_config = config['modeling']['train_test_split']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config['test_size'],
        random_state=split_config['random_state'],
        shuffle=split_config['shuffle']
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Test size ratio: {split_config['test_size']}")
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics


def train_linear_regression(X_train, X_test, y_train, y_test, config):
    """
    Train and evaluate Linear/Ridge Regression model.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (model, metrics, predictions)
    """
    model_config = config['modeling']['models']['linear_regression']
    use_ridge = model_config.get('use_ridge', False)
    
    if use_ridge:
        print("\n" + "="*80)
        print("TRAINING RIDGE REGRESSION MODEL (with L2 Regularization)")
        print("="*80)
        
        alpha = model_config['params'].get('alpha', 1.0)
        model = Ridge(alpha=alpha)
        print(f"Alpha (regularization strength): {alpha}")
    else:
        print("\n" + "="*80)
        print("TRAINING LINEAR REGRESSION MODEL")
        print("="*80)
        model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTesting Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    if config['modeling']['cross_validation']['enabled']:
        cv_folds = config['modeling']['cross_validation']['cv_folds']
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=cv_folds, 
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"\nCross-Validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return model, metrics, y_test_pred


def analyze_linear_regression_coefficients(model, feature_names, config):
    """
    Analyze and visualize linear regression coefficients.
    
    Args:
        model: Trained Linear Regression model
        feature_names (list): List of feature names
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("LINEAR REGRESSION COEFFICIENT ANALYSIS")
    print("="*80)
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    # Sort by absolute coefficient value
    coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    # Display top features
    top_n = config['modeling']['feature_importance']['top_n']
    print(f"\nTop {top_n} Most Important Features (by coefficient magnitude):")
    print(coefficients.head(top_n)[['Feature', 'Coefficient']].to_string(index=False))
    
    # Plot coefficients
    if config['modeling']['feature_importance']['plot']:
        plots_dir = os.path.join(config['directories']['modeling_output'], 'plots')
        
        plt.figure(figsize=(12, 8))
        top_features = coefficients.head(top_n)
        
        colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
        plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors, alpha=0.7)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Linear Regression Coefficients', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'linear_regression_coefficients.png'), 
                   dpi=config['eda']['dpi'], bbox_inches='tight')
        plt.close()
        print(f"\nSaved: linear_regression_coefficients.png")


def train_random_forest(X_train, X_test, y_train, y_test, config):
    """
    Train and evaluate Random Forest model.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (model, metrics, predictions)
    """
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*80)
    
    # Get model parameters
    params = config['modeling']['models']['random_forest']['params']
    
    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTesting Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    if config['modeling']['cross_validation']['enabled']:
        cv_folds = config['modeling']['cross_validation']['cv_folds']
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=cv_folds, 
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"\nCross-Validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return model, metrics, y_test_pred


def analyze_feature_importance(model, feature_names, model_name, config):
    """
    Analyze and visualize feature importance for tree-based models.
    
    Args:
        model: Trained tree-based model
        feature_names (list): List of feature names
        model_name (str): Name of the model
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print(f"{model_name.upper()} FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Get feature importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance
    importances = importances.sort_values('Importance', ascending=False)
    
    # Display top features
    top_n = config['modeling']['feature_importance']['top_n']
    print(f"\nTop {top_n} Most Important Features:")
    print(importances.head(top_n).to_string(index=False))
    
    # Plot feature importance
    if config['modeling']['feature_importance']['plot']:
        plots_dir = os.path.join(config['directories']['modeling_output'], 'plots')
        
        plt.figure(figsize=(12, 8))
        top_features = importances.head(top_n)
        
        plt.barh(top_features['Feature'], top_features['Importance'], 
                color='steelblue', alpha=0.7)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} {model_name} Feature Importances', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filename = f'{model_name.lower().replace(" ", "_")}_feature_importance.png'
        plt.savefig(os.path.join(plots_dir, filename), 
                   dpi=config['eda']['dpi'], bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {filename}")


def visualize_tree(model, feature_names, config):
    """
    Visualize a decision tree from Random Forest.
    
    Args:
        model: Trained Random Forest model
        feature_names (list): List of feature names
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("VISUALIZING DECISION TREE")
    print("="*80)
    
    plots_dir = os.path.join(config['directories']['modeling_output'], 'plots')
    tree_config = config['modeling']['tree_visualization']
    
    # Get tree to visualize
    tree_index = tree_config['tree_index']
    max_depth = tree_config['max_depth']
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Plot tree
    plot_tree(model.estimators_[tree_index], 
             feature_names=feature_names,
             filled=True,
             rounded=True,
             max_depth=max_depth,
             fontsize=10)
    
    plt.title(f'Decision Tree #{tree_index} from Random Forest (Max Depth: {max_depth})',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, 'decision_tree_visualization.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: decision_tree_visualization.png")


def train_xgboost(X_train, X_test, y_train, y_test, config):
    """
    Train and evaluate XGBoost model.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (model, metrics, predictions) or None if XGBoost not available
    """
    if not XGBOOST_AVAILABLE:
        print("\n" + "="*80)
        print("XGBOOST NOT AVAILABLE - SKIPPING")
        print("="*80)
        return None, None, None
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    # Get model parameters
    params = config['modeling']['models']['xgboost']['params']
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTesting Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    if config['modeling']['cross_validation']['enabled']:
        cv_folds = config['modeling']['cross_validation']['cv_folds']
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=cv_folds, 
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"\nCross-Validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return model, metrics, y_test_pred


def plot_predictions(y_test, predictions_dict, config):
    """
    Plot actual vs predicted values for all models.
    
    Args:
        y_test: True test values
        predictions_dict (dict): Dictionary of model predictions
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("PLOTTING PREDICTIONS")
    print("="*80)
    
    plots_dir = os.path.join(config['directories']['modeling_output'], 'plots')
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        axes[idx].scatter(y_test, y_pred, alpha=0.5, color='steelblue')
        
        # Plot perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                      'r--', linewidth=2, label='Perfect Prediction')
        
        axes[idx].set_xlabel('Actual Price', fontsize=12)
        axes[idx].set_ylabel('Predicted Price', fontsize=12)
        axes[idx].set_title(f'{model_name}\nActual vs Predicted', 
                           fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predictions_comparison.png'), 
               dpi=config['eda']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: predictions_comparison.png")


def plot_residuals(y_test, predictions_dict, config):
    """
    Plot residuals for all models.
    
    Args:
        y_test: True test values
        predictions_dict (dict): Dictionary of model predictions
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("PLOTTING RESIDUALS")
    print("="*80)
    
    plots_dir = os.path.join(config['directories']['modeling_output'], 'plots')
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        residuals = y_test - y_pred
        
        axes[idx].scatter(y_pred, residuals, alpha=0.5, color='coral')
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=2)
        
        axes[idx].set_xlabel('Predicted Price', fontsize=12)
        axes[idx].set_ylabel('Residuals', fontsize=12)
        axes[idx].set_title(f'{model_name}\nResidual Plot', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_plot.png'), 
               dpi=config['eda']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: residuals_plot.png")


def compare_models(metrics_dict):
    """
    Compare performance of all models.
    
    Args:
        metrics_dict (dict): Dictionary of model metrics
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        if metrics is not None:
            comparison_data.append({
                'Model': model_name,
                'Test MAE': metrics['test']['MAE'],
                'Test RMSE': metrics['test']['RMSE'],
                'Test R2': metrics['test']['R2'],
                'Test MAPE': metrics['test']['MAPE']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best model based on R2
    best_idx = comparison_df['Test R2'].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    print(f"\n{'='*80}")
    print(f"Best Model (by R2): {best_model}")
    print(f"{'='*80}")


def save_models(models_dict, config):
    """
    Save trained models to disk.
    
    Args:
        models_dict (dict): Dictionary of trained models
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    models_dir = os.path.join(config['directories']['modeling_output'], 'models')
    
    for model_name, model in models_dict.items():
        if model is not None:
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            filepath = os.path.join(models_dir, filename)
            
            joblib.dump(model, filepath)
            print(f"Saved: {filename}")


def main():
    """
    Main function to execute modeling pipeline.
    """
    print("="*80)
    print("MACHINE LEARNING MODELING - CAR PRICE PREDICTION")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Create output directories
    create_output_directories(config)
    
    # Load processed data
    X, y, feature_names = load_processed_data(config)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    # Storage for models, metrics, and predictions
    models = {}
    metrics = {}
    predictions = {}
    
    # Train Linear Regression
    if config['modeling']['models']['linear_regression']['enabled']:
        lr_model, lr_metrics, lr_pred = train_linear_regression(
            X_train, X_test, y_train, y_test, config
        )
        model_name = 'Ridge Regression' if config['modeling']['models']['linear_regression'].get('use_ridge', False) else 'Linear Regression'
        models[model_name] = lr_model
        metrics[model_name] = lr_metrics
        predictions[model_name] = lr_pred
        
        analyze_linear_regression_coefficients(lr_model, feature_names, config)
    
    # Train Random Forest
    if config['modeling']['models']['random_forest']['enabled']:
        rf_model, rf_metrics, rf_pred = train_random_forest(
            X_train, X_test, y_train, y_test, config
        )
        models['Random Forest'] = rf_model
        metrics['Random Forest'] = rf_metrics
        predictions['Random Forest'] = rf_pred
        
        analyze_feature_importance(rf_model, feature_names, 'Random Forest', config)
        visualize_tree(rf_model, feature_names, config)
    
    # Train XGBoost
    if config['modeling']['models']['xgboost']['enabled']:
        xgb_model, xgb_metrics, xgb_pred = train_xgboost(
            X_train, X_test, y_train, y_test, config
        )
        if xgb_model is not None:
            models['XGBoost'] = xgb_model
            metrics['XGBoost'] = xgb_metrics
            predictions['XGBoost'] = xgb_pred
            
            analyze_feature_importance(xgb_model, feature_names, 'XGBoost', config)
    
    # Plot predictions and residuals
    plot_predictions(y_test, predictions, config)
    plot_residuals(y_test, predictions, config)
    
    # Compare models
    compare_models(metrics)
    
    # Save models
    save_models(models, config)
    
    # Print feature statistics
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    print(f"Total features used: {X.shape[1]}")
    print(f"Feature names (first 10): {feature_names[:10]}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names) - 10} more")
    
    print("\n" + "="*80)
    print("MODELING COMPLETED SUCCESSFULLY")
    print("="*80)
    modeling_output = config['directories']['modeling_output']
    print(f"\nModels saved to: {os.path.join(modeling_output, 'models')}")
    print(f"Plots saved to: {os.path.join(modeling_output, 'plots')}")
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80)
    
    # Check if all models performed poorly
    all_r2_scores = [m['test']['R2'] for m in metrics.values() if m is not None]
    if all_r2_scores and max(all_r2_scores) < 0.3:
        print("\n⚠️  Model performance is below expectations. Consider:")
        print("  1. Enable data validation to remove inconsistent records")
        print("     Set: preprocessing.validation.remove_invalid_records: true")
        print("  2. Check data quality - look for logical inconsistencies")
        print("  3. Try different encoding methods (target encoding enabled)")
        print("  4. Add more domain-specific features")
        print("  5. Collect better quality data if possible")
    else:
        print("\n✓ Model performance looks reasonable!")


if __name__ == "__main__":
    main()
