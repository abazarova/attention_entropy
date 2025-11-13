import pandas as pd
import argparse
from pathlib import Path

def find_best_hyperparameters(csv_file, metric_column='val_mean', optimization='maximize'):
    """
    Find the best hyperparameters based on validation metrics.
    
    Args:
        csv_file (str): Path to the CSV file with results
        metric_column (str): Column name for the validation metric (default: 'val_mean')
        optimization (str): 'maximize' or 'minimize' depending on the metric
    
    Returns:
        dict: Dictionary containing best hyperparameters and corresponding test metrics
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    required_columns = ['preprocess_name', 'method_alpha', 'method_aggregation', 'val_mean', 'test_mean']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Find the best row based on validation metric
    if optimization == 'maximize':
        best_idx = df[metric_column].idxmax()
    elif optimization == 'minimize':
        best_idx = df[metric_column].idxmin()
    else:
        raise ValueError("optimization must be 'maximize' or 'minimize'")
    
    best_row = df.loc[best_idx]
    
    # Extract best hyperparameters and metrics
    best_result = {
        'preprocess_name': best_row['preprocess_name'],
        'method_alpha': best_row['method_alpha'],
        'method_aggregation': best_row['method_aggregation'],
        'best_val_metric': best_row[metric_column],
        'corresponding_test_metric': best_row['test_mean'],
        'optimization': optimization,
        'total_configurations': len(df)
    }
    
    return best_result, df

def analyze_hyperparameter_sensitivity(df, hyperparameter='method_alpha'):
    """
    Analyze how sensitive the performance is to different hyperparameter values.
    
    Args:
        df (pd.DataFrame): Results dataframe
        hyperparameter (str): Hyperparameter to analyze ('method_alpha' or 'method_aggregation')
    
    Returns:
        pd.DataFrame: Aggregated statistics for each hyperparameter value
    """
    
    if hyperparameter not in df.columns:
        raise ValueError(f"Hyperparameter {hyperparameter} not found in dataframe")
    
    sensitivity_analysis = df.groupby(hyperparameter).agg({
        'val_mean': ['mean', 'std', 'min', 'max'],
        'test_mean': ['mean', 'std', 'min', 'max'],
    }).round(4)
    
    return sensitivity_analysis

def main():
    parser = argparse.ArgumentParser(description='Find best hyperparameters from results CSV')
    parser.add_argument('csv_file', help='Path to the CSV file with results')
    parser.add_argument('--metric', default='val_mean', 
                       help='Metric column to optimize (default: val_mean)')
    parser.add_argument('--optimization', default='maximize', 
                       choices=['maximize', 'minimize'],
                       help='Whether to maximize or minimize the metric (default: maximize)')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Perform sensitivity analysis on hyperparameters')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} does not exist")
        return
    
    try:
        # Find best hyperparameters
        best_result, df = find_best_hyperparameters(
            args.csv_file, 
            args.metric, 
            args.optimization
        )
        
        print("=" * 60)
        print("BEST HYPERPARAMETER CONFIGURATION")
        print("=" * 60)
        print(f"Dataset: {best_result['preprocess_name']}")
        print(f"Best Alpha: {best_result['method_alpha']}")
        print(f"Best Aggregation: {best_result['method_aggregation']}")
        print(f"Best Validation Metric: {best_result['best_val_metric']:.4f}")
        print(f"Corresponding Test Metric: {best_result['corresponding_test_metric']:.4f}")
        print(f"Optimization: {best_result['optimization']}")
        print(f"Total configurations evaluated: {best_result['total_configurations']}")
        print("=" * 60)
        
        # Show top 3 configurations
        print("\nTOP 3 CONFIGURATIONS:")
        print("-" * 40)
        if args.optimization == 'maximize':
            top_configs = df.nlargest(3, args.metric)
        else:
            top_configs = df.nsmallest(3, args.metric)
            
        for i, (idx, row) in enumerate(top_configs.iterrows(), 1):
            print(f"{i}. Alpha: {row['method_alpha']}, "
                  f"Aggregation: {row['method_aggregation']}, "
                  f"Val: {row['val_mean']:.4f}, "
                  f"Test: {row['test_mean']:.4f}")
        
        # Perform sensitivity analysis if requested
        if args.sensitivity:
            print("\n" + "=" * 60)
            print("HYPERPARAMETER SENSITIVITY ANALYSIS")
            print("=" * 60)
            
            # Analyze alpha sensitivity
            print("\nALPHA SENSITIVITY:")
            alpha_sensitivity = analyze_hyperparameter_sensitivity(df, 'method_alpha')
            print(alpha_sensitivity)
            
            # Analyze aggregation sensitivity
            print("\nAGGREGATION SENSITIVITY:")
            agg_sensitivity = analyze_hyperparameter_sensitivity(df, 'method_aggregation')
            print(agg_sensitivity)
            
    except Exception as e:
        print(f"Error processing file: {e}")

# Function to use in other scripts
def get_best_configuration(csv_file, metric_column='val_mean', optimization='maximize'):
    """
    Convenience function to get best configuration for use in other scripts.
    
    Returns:
        tuple: (best_alpha, best_aggregation, test_metric)
    """
    best_result, _ = find_best_hyperparameters(csv_file, metric_column, optimization)
    return (
        best_result['method_alpha'],
        best_result['method_aggregation'],
        best_result['corresponding_test_metric']
    )

if __name__ == "__main__":
    main()