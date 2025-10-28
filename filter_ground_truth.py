"""
Filter and calculate accuracy for ground truth Excel data.

This script:
1. Reads the Excel file with model predictions (1-5) and score_pred (0 or 1)
2. For each model, filters rows where model's prediction equals target_score
3. Calculates accuracy as percentage of rows where score_pred == 1
4. Returns accuracy percentages for each model
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Cache for the ground truth DataFrame (loaded once)
_ground_truth_cache: Dict[str, pd.DataFrame] = {}


def get_ground_truth_data(excel_path: str) -> pd.DataFrame:
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


def calculate_models_accuracy(
    df: pd.DataFrame,
    target_score: int = 5,
    models: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate accuracy for each model based on target_score.
    
    Logic:
    1. Filters rows where model's prediction equals target_score
    2. Calculates what percentage of those rows have score_pred == 1 (correct)
    
    Args:
        df: DataFrame with the data
        target_score: Target score to filter by (default: 5)
        models: List of model names to process. If None, processes all models.
    
    Returns:
        Dictionary mapping model names to accuracy percentages
    """
    # If models not specified, get all model columns (exclude Question, Unnamed: 0, score_pred)
    if models is None:
        models = [col for col in df.columns if col not in ['Question', 'Unnamed: 0', 'score_pred']]
    
    results = {}
    
    print(f"Calculating accuracy for predictions == {target_score}...")
    print()
    
    for model_name in models:
        # Check if column exists
        if model_name not in df.columns:
            print(f"Warning: {model_name} not found in Excel")
            continue
        
        # Step 1: Filter rows where model's prediction equals target_score
        filtered_df = df[df[model_name] == target_score].copy()
        
        if len(filtered_df) == 0:
            results[model_name] = 0.0
            print(f"{model_name:50s} 0.00% (0 rows with prediction == {target_score})")
            continue
        
        # Step 2: Calculate percentage where score_pred == 1 (correct answers)
        correct_count = filtered_df['score_pred'].sum()  # Count where score_pred == 1
        total_count = len(filtered_df)
        percentage = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        
        results[model_name] = percentage
        
        print(f"{model_name:50s} {percentage:6.2f}% ({correct_count}/{total_count} correct)")
    
    return results


def process_excel_file(
    excel_path: str,
    target_score: int = 5,
    models: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Main function to process the Excel file and calculate accuracy.
    
    Args:
        excel_path: Path to input Excel file
        target_score: Target score to filter by (default: 5)
        models: List of models to process (None = all)
    
    Returns:
        Dictionary mapping model names to accuracy percentages
    
    Example:
        # Calculate accuracy for predictions == 5
        results = process_excel_file(
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
    df = get_ground_truth_data(excel_path)
    
    # Calculate accuracy
    results = calculate_models_accuracy(df, target_score, models)
    
    if not results:
        print("No results found")
        return results
    
    # Print summary
    print("\n" + "="*80)
    print("ACCURACY RESULTS (sorted by performance)")
    print("="*80)
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, percentage in sorted_results:
        print(f"{model_name:50s} {percentage:6.2f}%")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python filter_ground_truth.py <excel_file_path> [target_score]")
        print("\nCalculates accuracy for each model where:")
        print("  - Model's prediction equals target_score")
        print("  - score_pred == 1 (correct answers)")
        print("\nExample:")
        print("  python filter_ground_truth.py dataset/inference_on_pretrained_model.xlsx 5")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    target_score = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Excel file is read only once (cached)
    results = process_excel_file(excel_file, target_score)

