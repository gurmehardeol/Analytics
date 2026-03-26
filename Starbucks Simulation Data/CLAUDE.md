# Starbucks Customer Segmentation Pipeline

## Dataset
- File: data/starbucks_orders.csv
- Goal: Cluster customers, profile them, then build classifiers

## Pipeline — run in sequence (each must complete before the next)

### Step 1
Use the eda-agent to perform visual EDA on the dataset.
Wait for it to complete and report findings.

### Step 2  
Use the preprocessing-agent to encode and normalize the data.
Wait for outputs/processed/data_scaled.csv to be created.

### Step 3
Use the datascience-agent to run clustering, profiling, and classification.
This agent runs all 3 phases internally.

## Output convention
All outputs go to the outputs/ directory. Never overwrite existing files.