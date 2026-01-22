"""
Data Preprocessing for Car Price Prediction

This script handles data cleaning, transformation, and preprocessing
based on configuration settings.
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
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
    Load the raw dataset from CSV file.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_path = config['data']['raw_data']
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    return df


def validate_data(df, config):
    """
    Validate data for logical consistency and quality issues.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Validated dataframe
    """
    if not config['preprocessing']['validation']['enabled']:
        print("Data validation skipped")
        return df
    
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    df_clean = df.copy()
    issues_found = []
    
    if config['preprocessing']['validation']['check_logical_consistency']:
        # Check Tesla-Electric consistency
        if 'Brand' in df_clean.columns and 'Fuel Type' in df_clean.columns:
            tesla_non_electric = df_clean[
                (df_clean['Brand'] == 'Tesla') & 
                (df_clean['Fuel Type'] != 'Electric')
            ]
            if len(tesla_non_electric) > 0:
                issues_found.append(f"Tesla with non-Electric fuel: {len(tesla_non_electric)} records")
        
        # Check for unrealistic year values
        if 'Year' in df_clean.columns:
            current_year = config['preprocessing']['feature_engineering']['current_year']
            future_cars = df_clean[df_clean['Year'] > current_year]
            if len(future_cars) > 0:
                issues_found.append(f"Cars from future years: {len(future_cars)} records")
            
            very_old_cars = df_clean[df_clean['Year'] < 1990]
            if len(very_old_cars) > 0:
                issues_found.append(f"Cars older than 1990: {len(very_old_cars)} records")
        
        # Check for unrealistic mileage
        if 'Mileage' in df_clean.columns:
            negative_mileage = df_clean[df_clean['Mileage'] < 0]
            if len(negative_mileage) > 0:
                issues_found.append(f"Negative mileage: {len(negative_mileage)} records")
        
        # Check for unrealistic prices
        target = config['columns']['target']
        if target in df_clean.columns:
            negative_price = df_clean[df_clean[target] < 0]
            if len(negative_price) > 0:
                issues_found.append(f"Negative prices: {len(negative_price)} records")
    
    if issues_found:
        print("\nData Quality Issues Detected:")
        for issue in issues_found:
            print(f"  ⚠️  {issue}")
        
        if config['preprocessing']['validation']['remove_invalid_records']:
            # Remove invalid records
            initial_rows = len(df_clean)
            
            if 'Brand' in df_clean.columns and 'Fuel Type' in df_clean.columns:
                df_clean = df_clean[~(
                    (df_clean['Brand'] == 'Tesla') & 
                    (df_clean['Fuel Type'] != 'Electric')
                )]
            
            if 'Year' in df_clean.columns:
                current_year = config['preprocessing']['feature_engineering']['current_year']
                df_clean = df_clean[
                    (df_clean['Year'] >= 1990) & 
                    (df_clean['Year'] <= current_year)
                ]
            
            if 'Mileage' in df_clean.columns:
                df_clean = df_clean[df_clean['Mileage'] >= 0]
            
            target = config['columns']['target']
            if target in df_clean.columns:
                df_clean = df_clean[df_clean[target] >= 0]
            
            removed_rows = initial_rows - len(df_clean)
            print(f"\n✓ Removed {removed_rows} invalid records")
            print(f"Dataset shape after validation: {df_clean.shape}")
        else:
            print("\nNote: Invalid records kept. Set 'remove_invalid_records: true' to remove them.")
    else:
        print("✓ No data quality issues detected")
    
    return df_clean


def create_output_directories(config):
    """
    Create output directories for processed data and preprocessing outputs.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create shared data directory
    os.makedirs(config['directories']['data'], exist_ok=True)
    # Create preprocessing-specific output directory
    os.makedirs(config['directories']['preprocessing_output'], exist_ok=True)
    print("Output directories created/verified")


def handle_missing_values(df, config):
    """
    Handle missing values in the dataset based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    print("\n" + "="*80)
    print("HANDLING MISSING VALUES")
    print("="*80)
    
    df_clean = df.copy()
    
    # Get configuration
    numeric_strategy = config['preprocessing']['missing_values']['numeric_strategy']
    categorical_strategy = config['preprocessing']['missing_values']['categorical_strategy']
    
    numeric_features = config['columns']['numeric_features']
    categorical_features = config['columns']['categorical_features']
    target = config['columns']['target']
    
    # Handle numeric features
    if numeric_strategy != 'drop':
        numeric_cols = [col for col in numeric_features if col in df_clean.columns]
        if numeric_cols:
            imputer = SimpleImputer(strategy=numeric_strategy)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            print(f"Numeric features imputed using '{numeric_strategy}' strategy")
    
    # Handle categorical features
    if categorical_strategy != 'drop':
        categorical_cols = [col for col in categorical_features if col in df_clean.columns]
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                if categorical_strategy == 'mode':
                    mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
                elif categorical_strategy == 'constant':
                    fill_value = config['preprocessing']['missing_values']['constant_fill_value']
                    df_clean[col].fillna(fill_value, inplace=True)
        print(f"Categorical features imputed using '{categorical_strategy}' strategy")
    
    # Handle target variable missing values
    if target in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=[target])
        dropped_rows = initial_rows - len(df_clean)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing target values")
    
    # Drop any remaining rows with missing values if strategy is 'drop'
    if numeric_strategy == 'drop' or categorical_strategy == 'drop':
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        dropped_rows = initial_rows - len(df_clean)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing values")
    
    print(f"Dataset shape after handling missing values: {df_clean.shape}")
    
    return df_clean


def create_features(df, config):
    """
    Create new features based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_new = df.copy()
    feature_eng_config = config['preprocessing']['feature_engineering']
    
    # Create car age feature
    if feature_eng_config['create_age_feature']:
        if 'Year' in df_new.columns:
            current_year = feature_eng_config['current_year']
            df_new['Car_Age'] = current_year - df_new['Year']
            print(f"✓ Created 'Car_Age' feature from Year (current year: {current_year})")
            
            # Add to numeric features for later processing
            if 'Car_Age' not in config['columns']['numeric_features']:
                config['columns']['numeric_features'].append('Car_Age')
    
    # Create brand tier (luxury vs economy)
    if feature_eng_config['create_brand_tier']:
        if 'Brand' in df_new.columns:
            luxury_brands = ['BMW', 'Mercedes', 'Audi', 'Tesla']
            df_new['Is_Luxury_Brand'] = df_new['Brand'].isin(luxury_brands).astype(int)
            print(f"✓ Created 'Is_Luxury_Brand' feature")
            
            if 'Is_Luxury_Brand' not in config['columns']['numeric_features']:
                config['columns']['numeric_features'].append('Is_Luxury_Brand')
    
    # Create mileage per year
    if feature_eng_config['create_mileage_per_year']:
        if 'Mileage' in df_new.columns and 'Car_Age' in df_new.columns:
            # Avoid division by zero
            df_new['Mileage_Per_Year'] = df_new['Mileage'] / (df_new['Car_Age'] + 1)
            print(f"✓ Created 'Mileage_Per_Year' feature")
            
            if 'Mileage_Per_Year' not in config['columns']['numeric_features']:
                config['columns']['numeric_features'].append('Mileage_Per_Year')
    
    # Create interaction features
    if feature_eng_config['create_interaction_features']:
        if 'Car_Age' in df_new.columns and 'Mileage' in df_new.columns:
            df_new['Age_Mileage_Interaction'] = df_new['Car_Age'] * df_new['Mileage'] / 100000
            print(f"✓ Created 'Age_Mileage_Interaction' feature")
            
            if 'Age_Mileage_Interaction' not in config['columns']['numeric_features']:
                config['columns']['numeric_features'].append('Age_Mileage_Interaction')
        
        if 'Engine Size' in df_new.columns and 'Car_Age' in df_new.columns:
            df_new['Engine_Age_Interaction'] = df_new['Engine Size'] * df_new['Car_Age']
            print(f"✓ Created 'Engine_Age_Interaction' feature")
            
            if 'Engine_Age_Interaction' not in config['columns']['numeric_features']:
                config['columns']['numeric_features'].append('Engine_Age_Interaction')
    
    print(f"Total features after engineering: {df_new.shape[1]}")
    
    return df_new


def handle_outliers(df, config):
    """
    Handle outliers in numeric features based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    print("\n" + "="*80)
    print("HANDLING OUTLIERS")
    print("="*80)
    
    df_clean = df.copy()
    outlier_config = config['preprocessing']['outliers']
    method = outlier_config['method']
    
    if method == 'none':
        print("Outlier handling skipped (method: none)")
        return df_clean
    
    numeric_features = config['columns']['numeric_features']
    target = config['columns']['target']
    numeric_cols = [col for col in numeric_features + [target] if col in df_clean.columns]
    
    if method == 'iqr':
        threshold = outlier_config['threshold']
        handle = outlier_config['handle']
        
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                if handle == 'cap':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    print(f"{col}: Capped {outliers_count} outliers")
                elif handle == 'remove':
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    print(f"{col}: Removed {outliers_count} outliers")
    
    elif method == 'zscore':
        from scipy import stats
        threshold = outlier_config.get('zscore_threshold', 3)
        handle = outlier_config['handle']
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            outliers_mask = z_scores > threshold
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                if handle == 'remove':
                    df_clean = df_clean[z_scores <= threshold]
                    print(f"{col}: Removed {outliers_count} outliers using Z-score")
    
    print(f"Dataset shape after outlier handling: {df_clean.shape}")
    
    return df_clean


def target_encode_feature(df, col, target_col, smoothing=1.0, min_samples_leaf=1):
    """
    Target encode a categorical feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        col (str): Column to encode
        target_col (str): Target column name
        smoothing (float): Smoothing parameter
        min_samples_leaf (int): Minimum samples per category
        
    Returns:
        tuple: (encoded series, encoding map)
    """
    # Calculate global mean
    global_mean = df[target_col].mean()
    
    # Calculate category means and counts
    agg = df.groupby(col)[target_col].agg(['mean', 'count'])
    
    # Calculate smoothed means
    smoothed_means = (
        (agg['count'] * agg['mean'] + smoothing * global_mean) / 
        (agg['count'] + smoothing)
    )
    
    # Create encoding map
    encoding_map = smoothed_means.to_dict()
    
    # Apply encoding
    encoded = df[col].map(encoding_map).fillna(global_mean)
    
    return encoded, encoding_map


def encode_categorical_features(df, config):
    """
    Encode categorical features based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (encoded dataframe, encoders dictionary)
    """
    print("\n" + "="*80)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*80)
    
    df_encoded = df.copy()
    encoders = {}
    
    method = config['preprocessing']['encoding']['method']
    categorical_features = config['columns']['categorical_features']
    categorical_cols = [col for col in categorical_features if col in df_encoded.columns]
    target = config['columns']['target']
    
    if method == 'target':
        # Target encoding for high-cardinality features
        target_config = config['preprocessing']['encoding']['target_encoding']
        smoothing = target_config['smoothing']
        min_samples_leaf = target_config['min_samples_leaf']
        high_card_threshold = target_config['high_cardinality_threshold']
        
        for col in categorical_cols:
            n_categories = df_encoded[col].nunique()
            
            if n_categories > high_card_threshold:
                # Use target encoding for high-cardinality features
                encoded_col, encoding_map = target_encode_feature(
                    df_encoded, col, target, smoothing, min_samples_leaf
                )
                df_encoded[f'{col}_TargetEnc'] = encoded_col
                encoders[col] = {'type': 'target', 'map': encoding_map, 'global_mean': df_encoded[target].mean()}
                print(f"Target encoded '{col}' ({n_categories} categories) → {col}_TargetEnc")
                
                # Remove original column
                df_encoded = df_encoded.drop(columns=[col])
            else:
                # Use label encoding for low-cardinality features
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                encoders[col] = {'type': 'label', 'encoder': encoder}
                print(f"Label encoded '{col}' ({n_categories} categories)")
    
    elif method == 'onehot':
        handle_unknown = config['preprocessing']['encoding']['handle_unknown']
        drop_first = config['preprocessing']['encoding']['drop_first']
        
        for col in categorical_cols:
            # Convert to string to handle any numeric categories
            df_encoded[col] = df_encoded[col].astype(str)
            
            encoder = OneHotEncoder(handle_unknown=handle_unknown, 
                                   drop='first' if drop_first else None,
                                   sparse_output=False)
            
            encoded_features = encoder.fit_transform(df_encoded[[col]])
            
            # Create column names
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            if drop_first and len(feature_names) > 1:
                feature_names = feature_names[1:]
            
            # Create dataframe with encoded features
            encoded_df = pd.DataFrame(encoded_features, 
                                     columns=feature_names,
                                     index=df_encoded.index)
            
            # Drop original column and concatenate encoded features
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            
            encoders[col] = encoder
            print(f"One-hot encoded '{col}' into {len(feature_names)} features")
    
    elif method == 'label':
        for col in categorical_cols:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder
            print(f"Label encoded '{col}' into numeric values")
    
    print(f"Dataset shape after encoding: {df_encoded.shape}")
    
    return df_encoded, encoders


def create_polynomial_features(df, config):
    """
    Create polynomial features for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Dataframe with polynomial features
    """
    feature_eng_config = config['preprocessing']['feature_engineering']
    
    if not feature_eng_config.get('create_polynomial_features', False):
        return df
    
    print("\n" + "="*80)
    print("CREATING POLYNOMIAL FEATURES")
    print("="*80)
    
    df_poly = df.copy()
    degree = feature_eng_config.get('polynomial_degree', 2)
    
    # Select only original numeric features (not engineered ones)
    original_numeric = ['Year', 'Engine Size', 'Mileage']
    numeric_cols = [col for col in original_numeric if col in df_poly.columns]
    
    if numeric_cols:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df_poly[numeric_cols])
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Add only the new polynomial features (exclude original features)
        new_features = [name for name in poly_feature_names if name not in numeric_cols]
        new_feature_idx = [i for i, name in enumerate(poly_feature_names) if name not in numeric_cols]
        
        poly_df = pd.DataFrame(
            poly_features[:, new_feature_idx],
            columns=new_features,
            index=df_poly.index
        )
        
        # Drop original numeric columns and concatenate
        df_poly = df_poly.drop(columns=numeric_cols)
        df_poly = pd.concat([df_poly, poly_df], axis=1)
        
        print(f"✓ Created {len(new_features)} polynomial features (degree={degree})")
        print(f"Total features: {df_poly.shape[1]}")
    
    return df_poly


def scale_features(df, config):
    """
    Scale/normalize numeric features based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (scaled dataframe, scaler object)
    """
    print("\n" + "="*80)
    print("SCALING FEATURES")
    print("="*80)
    
    df_scaled = df.copy()
    method = config['preprocessing']['scaling']['method']
    
    if method == 'none':
        print("Feature scaling skipped (method: none)")
        return df_scaled, None
    
    # Identify numeric columns (excluding target)
    target = config['columns']['target']
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from scaling if present
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    # Remove ID column if present
    id_col = config['columns'].get('id_column')
    if id_col and id_col in numeric_cols:
        numeric_cols.remove(id_col)
    
    if not numeric_cols:
        print("No numeric columns to scale")
        return df_scaled, None
    
    # Select scaler based on configuration
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        print(f"Unknown scaling method: {method}. Skipping scaling.")
        return df_scaled, None
    
    # Fit and transform
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    print(f"Applied '{method}' scaling to {len(numeric_cols)} numeric features")
    
    return df_scaled, scaler


def prepare_final_dataset(df, config):
    """
    Prepare final dataset by separating features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (X, y) features and target
    """
    print("\n" + "="*80)
    print("PREPARING FINAL DATASET")
    print("="*80)
    
    target = config['columns']['target']
    id_col = config['columns'].get('id_column')
    
    # Separate target
    y = df[target].copy()
    
    # Remove target and ID from features
    X = df.drop(columns=[target], errors='ignore')
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y


def save_processed_data(X, y, encoders, scaler, config):
    """
    Save processed data and preprocessing objects.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        encoders (dict): Encoders dictionary
        scaler: Scaler object
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    # Shared data directory (for files used by other scripts)
    shared_data_dir = config['directories']['data']
    # Preprocessing-specific output directory
    preprocessing_output_dir = config['directories']['preprocessing_output']
    
    # Save processed data to SHARED location (used by modeling.py)
    processed_data = {
        'X': X,
        'y': y,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    output_path = config['data']['processed_data']
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed data saved to: {output_path} (shared)")
    
    # Save CSV for inspection to preprocessing output folder
    csv_path = os.path.join(preprocessing_output_dir, 'processed_data_inspection.csv')
    processed_df = pd.concat([X, y], axis=1)
    processed_df.to_csv(csv_path, index=False)
    print(f"Inspection CSV saved to: {csv_path}")
    
    # Save preprocessing objects to preprocessing output folder
    preprocessing_objects = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    objects_path = os.path.join(preprocessing_output_dir, 'preprocessing_objects.pkl')
    with open(objects_path, 'wb') as f:
        pickle.dump(preprocessing_objects, f)
    
    print(f"Preprocessing objects saved to: {objects_path}")


def main():
    """
    Main function to execute preprocessing pipeline.
    """
    print("="*80)
    print("DATA PREPROCESSING - CAR PRICE PREDICTION")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Create output directories
    create_output_directories(config)
    
    # Load data
    df = load_data(config)
    
    # Preprocessing pipeline
    df = validate_data(df, config)
    df = handle_missing_values(df, config)
    df = create_features(df, config)
    df = handle_outliers(df, config)
    df_encoded, encoders = encode_categorical_features(df, config)
    df_poly = create_polynomial_features(df_encoded, config)
    df_scaled, scaler = scale_features(df_poly, config)
    
    # Prepare final dataset
    X, y = prepare_final_dataset(df_scaled, config)
    
    # Save processed data
    save_processed_data(X, y, encoders, scaler, config)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nSummary:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Target variable: {config['columns']['target']}")
    print(f"  - Output location: {config['data']['processed_data']}")


if __name__ == "__main__":
    main()
