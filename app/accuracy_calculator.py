"""Accuracy calculator for benchmarking dataset."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Cache for the benchmarking dataset DataFrame (loaded once)
_benchmark_dataset_cache: Dict[str, pd.DataFrame] = {}

# Module logger
logger = logging.getLogger(__name__)


def _get_benchmark_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Get benchmarking dataset DataFrame (cached - reads only once).
    
    Args:
        dataset_path: Path to benchmarking dataset file
        
    Returns:
        DataFrame with the benchmarking dataset data
    """
    if dataset_path not in _benchmark_dataset_cache:
        logger.info(f"Loading benchmarking dataset: {dataset_path}")
        _benchmark_dataset_cache[dataset_path] = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(_benchmark_dataset_cache[dataset_path])} rows")
    
    return _benchmark_dataset_cache[dataset_path]


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
        dataframe: DataFrame with the benchmarking dataset
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
    
    logger.info(f"Calculating accuracy for predictions == {target_score}...")
    
    for model_name in model_names:
        # Check if column exists
        if model_name not in dataframe.columns:
            logger.warning(f"Model column not found in benchmarking dataset: {model_name}")
            continue
        
        # Step 1: Filter rows where model's prediction equals target_score
        filtered_dataframe = dataframe[dataframe['Score_'+str(model_name)] == target_score].copy()
        
        if len(filtered_dataframe) == 0:
            accuracy_results[model_name] = 0.0
            logger.info(f"{model_name:50s} 0.00% (0 rows with prediction == {target_score})")
            continue
        
        # Step 2: Calculate percentage where score_pred == 1 (correct answers)
        correct_predictions_count = filtered_dataframe['Result_'+str(model_name)].sum()  # Count where score_pred == 1
        total_predictions_count = len(filtered_dataframe)
        accuracy_percentage = (correct_predictions_count / total_predictions_count) * 100 if total_predictions_count > 0 else 0.0
        
        accuracy_results[model_name] = accuracy_percentage
        
        logger.info(
            f"{model_name:50s} {accuracy_percentage:6.2f}% "
            f"({correct_predictions_count}/{total_predictions_count} correct)"
        )
    
    return accuracy_results


def calculate_accuracy_from_dataset(
    dataset_path: str,
    target_score: int = 5,
    model_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Main function to process the benchmarking dataset and calculate accuracy.
    
    Args:
        dataset_path: Path to input benchmarking dataset file
        target_score: Target score to filter by (default: 5)
        model_names: List of models to process (None = all)
    
    Returns:
        Dictionary mapping model names to accuracy percentages (unsorted)
    
    Example:
        # Calculate accuracy for predictions == 5
        results = calculate_accuracy_from_dataset(
            dataset_path="dataset/inference_on_pretrained_model.xlsx",
            target_score=5
        )
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Benchmarking dataset file not found: {dataset_path}")
    
    logger.info(f"Processing benchmarking dataset: {Path(dataset_path).name}")
    logger.info(f"Target score: {target_score}")
    
    # Get cached or read benchmarking dataset ONCE (from cache)
    dataframe = _get_benchmark_dataset(dataset_path)
    
    # Calculate accuracy and return raw results
    return _calculate_model_accuracy_scores(dataframe, target_score, model_names)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        logger.error("Usage: python accuracy_calculator.py <dataset_file_path> [target_score]")
        logger.info("Calculates accuracy for each model where:")
        logger.info("  - Model's prediction equals target_score")
        logger.info("  - score_pred == 1 (correct answers)")
        logger.info("Example:  python accuracy_calculator.py dataset/inference_on_pretrained_model.xlsx 5")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    target_score = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Benchmarking dataset is read only once (cached)
    results = calculate_accuracy_from_dataset(dataset_file, target_score)
    
    # Log sorted results
    logger.info("\n" + "="*80)
    logger.info("ACCURACY RESULTS (sorted by performance)")
    logger.info("="*80)
    
    sorted_model_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, accuracy_percentage in sorted_model_results:
        logger.info(f"{model_name:50s} {accuracy_percentage:6.2f}%")

