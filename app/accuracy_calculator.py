"""Accuracy calculator for ground truth Excel data."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Cache for the ground truth DataFrame (loaded once)
_ground_truth_cache: Dict[str, pd.DataFrame] = {}


def _get_ground_truth_data(excel_path: str) -> pd.DataFrame:
    """
    Get ground truth DataFrame (cached - reads only once).
    
    Args:
        excel_path: Path to Excel file
        
    Returns:
        DataFrame with the ground truth data
    """
    if excel_path not in _ground_truth_cache:
        print(f"Loading Excel file: {excel_path}")
        _ground_truth_cache[excel_path] = pd.read_excel(excel_path)
        print(f"Loaded {len(_ground_truth_cache[excel_path])} rows\n")
    
    return _ground_truth_cache[excel_path]


def _calculate_model_accuracy_scores(
    dataframe: pd.DataFrame,
    target_score: int = 5,
    model_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate accuracy for each model based on target_score.
    
    Logic:
    1. Filters rows where model's prediction equals target_score
    2. Calculates what percentage of those rows have score_pred == 1 (correct)
    
    Args:
        dataframe: DataFrame with the data
        target_score: Target score to filter by (default: 5)
        model_names: List of model names to process. If None, processes all models.
    
    Returns:
        Dictionary mapping model names to accuracy percentages
    """
    # If models not specified, get all model columns (exclude Question, Unnamed: 0, score_pred)
    if model_names is None:
        model_names = [
            col for col in dataframe.columns 
            if col not in ['Question', 'Unnamed: 0', 'score_pred']
        ]
    
    accuracy_results = {}
    
    print(f"Calculating accuracy for predictions == {target_score}...")
    print()
    
    for model_name in model_names:
        # Check if column exists
        if model_name not in dataframe.columns:
            print(f"Warning: {model_name} not found in Excel")
            continue
        
        # Step 1: Filter rows where model's prediction equals target_score
        filtered_dataframe = dataframe[dataframe[model_name] == target_score].copy()
        
        if len(filtered_dataframe) == 0:
            accuracy_results[model_name] = 0.0
            print(f"{model_name:50s} 0.00% (0 rows with prediction == {target_score})")
            continue
        
        # Step 2: Calculate percentage where score_pred == 1 (correct answers)
        correct_predictions_count = filtered_dataframe['score_pred'].sum()  # Count where score_pred == 1
        total_predictions_count = len(filtered_dataframe)
        accuracy_percentage = (correct_predictions_count / total_predictions_count) * 100 if total_predictions_count > 0 else 0.0
        
        accuracy_results[model_name] = accuracy_percentage
        
        print(f"{model_name:50s} {accuracy_percentage:6.2f}% ({correct_predictions_count}/{total_predictions_count} correct)")
    
    return accuracy_results


def calculate_accuracy_from_excel(
    excel_path: str,
    target_score: int = 5,
    model_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Main function to process the Excel file and calculate accuracy.
    
    Args:
        excel_path: Path to input Excel file
        target_score: Target score to filter by (default: 5)
        model_names: List of models to process (None = all)
    
    Returns:
        Dictionary mapping model names to accuracy percentages
    
    Example:
        # Calculate accuracy for predictions == 5
        results = calculate_accuracy_from_excel(
            excel_path="dataset/inference_on_pretrained_model.xlsx",
            target_score=5
        )
    """
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    print(f"Processing: {Path(excel_path).name}")
    print(f"Target score: {target_score}")
    print()
    
    # Get cached or read Excel file ONCE (from cache)
    dataframe = _get_ground_truth_data(excel_path)
    
    # Calculate accuracy
    accuracy_results = _calculate_model_accuracy_scores(dataframe, target_score, model_names)
    
    if not accuracy_results:
        print("No results found")
        return accuracy_results
    
    # Print summary
    print("\n" + "="*80)
    print("ACCURACY RESULTS (sorted by performance)")
    print("="*80)
    
    # Sort by accuracy (descending)
    sorted_model_results = sorted(accuracy_results.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, accuracy_percentage in sorted_model_results:
        print(f"{model_name:50s} {accuracy_percentage:6.2f}%")
    
    return accuracy_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python accuracy_calculator.py <excel_file_path> [target_score]")
        print("\nCalculates accuracy for each model where:")
        print("  - Model's prediction equals target_score")
        print("  - score_pred == 1 (correct answers)")
        print("\nExample:")
        print("  python accuracy_calculator.py dataset/inference_on_pretrained_model.xlsx 5")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    target_score = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Excel file is read only once (cached)
    results = calculate_accuracy_from_excel(excel_file, target_score)

