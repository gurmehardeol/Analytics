"""
Exploratory Data Analysis (EDA) for Car Price Prediction

This script performs comprehensive EDA on the car price dataset,
including visualizations and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import warnings
warnings.filterwarnings('ignore')


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


def load_data(config):
    """
    Load the dataset from CSV file.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_path = config['data']['raw_data']
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    return df


def create_output_directory(config):
    """
    Create output directory for plots if it doesn't exist.
    
    Args:
        config (dict): Configuration dictionary
    """
    eda_output_dir = config['directories']['eda_output']
    os.makedirs(eda_output_dir, exist_ok=True)
    print(f"EDA output directory created/verified: {eda_output_dir}")


def basic_data_info(df, config):
    """
    Display basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("BASIC DATA INFORMATION")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    
    print("\nColumn Data Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    target = config['columns']['target']
    if target in df.columns:
        print(f"\nTarget Variable ({target}) Statistics:")
        print(df[target].describe())


def plot_target_distribution(df, config):
    """
    Plot the distribution of the target variable (Price).
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    target = config['columns']['target']
    plots_dir = config['directories']['eda_output']
    
    plt.figure(figsize=config['eda']['figure_size'])
    sns.set_style(config['eda']['plot_style'])
    
    # Create subplot with histogram and box plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df[target].dropna(), bins=50, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Price', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Distribution of {target}', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[target].dropna(), vert=True)
    axes[1].set_ylabel('Price', fontsize=12)
    axes[1].set_title(f'Box Plot of {target}', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'target_distribution.png'), 
                dpi=config['eda']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: target_distribution.png")


def plot_numeric_distributions(df, config):
    """
    Plot distributions of all numeric features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    numeric_features = config['columns']['numeric_features']
    plots_dir = config['directories']['eda_output']
    
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(numeric_features):
        if feature in df.columns:
            axes[idx].hist(df[feature].dropna(), bins=30, color='lightgreen', edgecolor='black')
            axes[idx].set_xlabel(feature, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numeric_distributions.png'), 
                dpi=config['eda']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: numeric_distributions.png")


def plot_categorical_distributions(df, config):
    """
    Plot count plots for categorical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    categorical_features = config['columns']['categorical_features']
    plots_dir = config['directories']['eda_output']
    
    for feature in categorical_features:
        if feature in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Get value counts and limit to top 15 categories if too many
            value_counts = df[feature].value_counts()
            if len(value_counts) > 15:
                value_counts = value_counts.head(15)
                df_plot = df[df[feature].isin(value_counts.index)]
                title_suffix = " (Top 15)"
            else:
                df_plot = df
                title_suffix = ""
            
            sns.countplot(data=df_plot, y=feature, 
                         palette=config['eda']['color_palette'],
                         order=df_plot[feature].value_counts().index)
            plt.xlabel('Count', fontsize=12)
            plt.ylabel(feature, fontsize=12)
            plt.title(f'Distribution of {feature}{title_suffix}', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            filename = f'{feature.lower().replace(" ", "_")}_distribution.png'
            plt.savefig(os.path.join(plots_dir, filename), 
                       dpi=config['eda']['dpi'], bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")


def plot_correlation_heatmap(df, config):
    """
    Plot correlation heatmap for numeric features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    plots_dir = config['directories']['eda_output']
    target = config['columns']['target']
    numeric_features = config['columns']['numeric_features']
    
    # Select numeric columns including target
    numeric_cols = numeric_features + [target]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), 
                dpi=config['eda']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: correlation_heatmap.png")


def plot_feature_vs_target(df, config):
    """
    Plot relationships between features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    target = config['columns']['target']
    numeric_features = config['columns']['numeric_features']
    plots_dir = config['directories']['eda_output']
    
    # Scatter plots for numeric features vs target
    for feature in numeric_features:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Remove NaN values for cleaner plotting
            plot_df = df[[feature, target]].dropna()
            
            plt.scatter(plot_df[feature], plot_df[target], alpha=0.5, color='steelblue')
            plt.xlabel(feature, fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.title(f'{feature} vs {target}', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(plot_df[feature], plot_df[target], 1)
            p = np.poly1d(z)
            plt.plot(plot_df[feature].sort_values(), 
                    p(plot_df[feature].sort_values()), 
                    "r--", alpha=0.8, linewidth=2, label='Trend line')
            plt.legend()
            
            plt.tight_layout()
            filename = f'{feature.lower().replace(" ", "_")}_vs_price.png'
            plt.savefig(os.path.join(plots_dir, filename), 
                       dpi=config['eda']['dpi'], bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")


def plot_categorical_vs_target(df, config):
    """
    Plot box plots for categorical features vs target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    target = config['columns']['target']
    categorical_features = config['columns']['categorical_features']
    plots_dir = config['directories']['eda_output']
    
    for feature in categorical_features:
        if feature in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Get value counts and limit to top 10 categories if too many
            value_counts = df[feature].value_counts()
            if len(value_counts) > 10:
                top_categories = value_counts.head(10).index
                df_plot = df[df[feature].isin(top_categories)]
                title_suffix = " (Top 10 Categories)"
            else:
                df_plot = df
                title_suffix = ""
            
            # Remove NaN values
            df_plot = df_plot[[feature, target]].dropna()
            
            sns.boxplot(data=df_plot, y=feature, x=target, 
                       palette=config['eda']['color_palette'],
                       order=df_plot.groupby(feature)[target].median().sort_values(ascending=False).index)
            plt.ylabel(feature, fontsize=12)
            plt.xlabel(target, fontsize=12)
            plt.title(f'{feature} vs {target}{title_suffix}', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            filename = f'{feature.lower().replace(" ", "_")}_vs_price_boxplot.png'
            plt.savefig(os.path.join(plots_dir, filename), 
                       dpi=config['eda']['dpi'], bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")


def correlation_with_target(df, config):
    """
    Analyze and visualize correlation of features with target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
    """
    target = config['columns']['target']
    numeric_features = config['columns']['numeric_features']
    plots_dir = config['directories']['eda_output']
    
    # Calculate correlations
    correlations = {}
    for feature in numeric_features:
        if feature in df.columns:
            corr = df[[feature, target]].corr().iloc[0, 1]
            correlations[feature] = corr
    
    # Sort by absolute correlation
    correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    print("\n" + "="*80)
    print("CORRELATION WITH TARGET VARIABLE")
    print("="*80)
    for feature, corr in correlations.items():
        print(f"{feature:20s}: {corr:+.4f}")
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    features = list(correlations.keys())
    corr_values = list(correlations.values())
    
    colors = ['red' if x < 0 else 'green' for x in corr_values]
    plt.barh(features, corr_values, color=colors, alpha=0.7)
    plt.xlabel('Correlation with Price', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Correlation with Target Variable', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_correlations.png'), 
                dpi=config['eda']['dpi'], bbox_inches='tight')
    plt.close()
    print(f"\nSaved: feature_correlations.png")


def main():
    """
    Main function to execute EDA pipeline.
    """
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS - CAR PRICE PREDICTION")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    create_output_directory(config)
    
    # Load data
    df = load_data(config)
    
    # Basic data information
    basic_data_info(df, config)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Generate visualizations
    plot_target_distribution(df, config)
    plot_numeric_distributions(df, config)
    plot_categorical_distributions(df, config)
    plot_correlation_heatmap(df, config)
    plot_feature_vs_target(df, config)
    plot_categorical_vs_target(df, config)
    correlation_with_target(df, config)
    
    print("\n" + "="*80)
    print("EDA COMPLETED SUCCESSFULLY")
    print(f"All plots saved to: {config['directories']['eda_output']}")
    print("="*80)


if __name__ == "__main__":
    main()
