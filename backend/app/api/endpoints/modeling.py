from fastapi import APIRouter, HTTPException , Query
from typing import Optional, List
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from app.services.ml_engine import MLEngine
from app.services.data_processor import DataProcessor
from app.models.schemas import ModelingRequest, ModelResult, TaskType
from app.core.config import settings
from app.core.exceptions import ModelTrainingException

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
ml_engine = MLEngine()
data_processor = DataProcessor()


def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

@router.post("/{dataset_id}/train")
async def train_models(
    dataset_id: str,
    target_column: str = Query(..., description="Target column for prediction"),
    task_type: str = Query("auto", description="Type of ML task: auto, classification, or regression"),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Test set size (0.1-0.5)"),
    algorithms: Optional[List[str]] = Query(None, description="List of algorithms to use"),
    hyperparameter_tuning: bool = Query(True, description="Enable hyperparameter tuning"),
    max_trials: int = Query(10, ge=5, le=50, description="Maximum optimization trials")
):
    """Train machine learning models with AutoML pipeline"""
    try:
        logger.info(f"=== Model Training Started ===")
        logger.info(f"Dataset ID: {dataset_id}")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Task type: {task_type}")
        logger.info(f"Test size: {test_size}")
        logger.info(f"Algorithms: {algorithms}")
        logger.info(f"Hyperparameter tuning: {hyperparameter_tuning}")
        logger.info(f"Max trials: {max_trials}")
        
        # Find and load dataset
        file_path = None
        for ext in settings.ALLOWED_EXTENSIONS:
            test_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.{ext}"
            if test_path.exists():
                file_path = test_path
                break
        
        if not file_path:
            logger.error(f"Dataset not found: {dataset_id}")
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        logger.info(f"Loading dataset from: {file_path}")
        df = data_processor.load_dataset(str(file_path))
        
        # Validate dataset
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset is empty. Please upload a valid dataset."
            )
        
        logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate target column
        if target_column not in df.columns:
            available_cols = df.columns.tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found. Available columns: {available_cols}"
            )
        
        # Check target column validity
        target_data = df[target_column]
        
        if target_data.isnull().all():
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' contains only null values"
            )
        
        # Check for sufficient unique values
        unique_values = target_data.nunique()
        if unique_values < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Target column must have at least 2 unique values, found {unique_values}"
            )
        
        # Validate task type
        valid_task_types = ["auto", "classification", "regression"]
        if task_type not in valid_task_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task type '{task_type}'. Must be one of: {valid_task_types}"
            )
        
        # Check class distribution for classification
        if unique_values < 20:  # Likely classification
            value_counts = target_data.value_counts()
            logger.info(f"Target column class distribution: {value_counts.to_dict()}")
            
            classes_with_single_sample = value_counts[value_counts < 2]
            if len(classes_with_single_sample) > 0:
                logger.warning(f"Classes with only 1 sample will be removed: {classes_with_single_sample.index.tolist()}")
                
                # Check if we'll have enough data after removal
                total_samples_to_remove = classes_with_single_sample.sum()
                remaining_samples = len(df) - total_samples_to_remove
                
                if remaining_samples < 10:
                    raise HTTPException(
                        status_code=400,
                        detail=f"After removing low-count classes, only {remaining_samples} samples remain. Need at least 10."
                    )
        
        # Validate algorithms if provided
        if algorithms:
            valid_algos = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting', 
                          'logistic_regression', 'linear_regression', 'ridge', 'lasso']
            invalid_algos = [a for a in algorithms if a not in valid_algos]
            if invalid_algos:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid algorithms: {invalid_algos}. Valid options: {valid_algos}"
                )
        
        # Run AutoML pipeline
        logger.info("Starting AutoML pipeline...")
        ml_results = ml_engine.auto_ml_pipeline(
            df=df,
            target_column=target_column,
            task_type=task_type,
            test_size=test_size,
            algorithms=algorithms,
            hyperparameter_tuning=hyperparameter_tuning,
            max_trials=max_trials
        )
        
        logger.info("AutoML pipeline completed successfully")
        
        # Convert NumPy types
        logger.info("Converting NumPy types to native Python types...")
        safe_results = convert_numpy_types(ml_results)
        
        logger.info("=== Model Training Completed ===")
        
        return {
            "dataset_id": dataset_id,
            "training_results": safe_results,
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except HTTPException:
        raise
    except ModelTrainingException as e:
        error_message = str(e.message if hasattr(e, 'message') else str(e))
        logger.error(f"Model training error: {error_message}")
        raise HTTPException(status_code=422, detail=error_message)
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/{dataset_id}/suggestions")
async def get_modeling_suggestions(dataset_id: str):
    """Get suggestions for modeling parameters"""
    try:
        logger.info(f"Getting modeling suggestions for dataset: {dataset_id}")
        
        # Find and load dataset
        file_path = None
        for ext in settings.ALLOWED_EXTENSIONS:
            test_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.{ext}"
            if test_path.exists():
                file_path = test_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = data_processor.load_dataset(str(file_path))
        
        # Analyze columns for suggestions
        suggestions = {
            'potential_targets': [],
            'recommended_task_type': 'auto',
            'recommended_algorithms': [],
            'recommended_test_size': 0.2
        }
        
        # Find potential target columns
        for col in df.columns:
            unique_count = df[col].nunique()
            null_ratio = df[col].isnull().sum() / len(df)
            
            # Good target column criteria
            if null_ratio < 0.1 and 2 <= unique_count <= len(df) * 0.9:
                suggestions['potential_targets'].append({
                    'column': col,
                    'unique_values': int(unique_count),
                    'null_percentage': float(null_ratio * 100),
                    'recommended': True
                })
        
        # Recommend algorithms based on dataset size
        if len(df) < 1000:
            suggestions['recommended_algorithms'] = ['random_forest', 'xgboost']
        else:
            suggestions['recommended_algorithms'] = ['xgboost', 'lightgbm', 'random_forest']
        
        # Adjust test size based on dataset size
        if len(df) < 100:
            suggestions['recommended_test_size'] = 0.15
        elif len(df) > 10000:
            suggestions['recommended_test_size'] = 0.25
        
        return suggestions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
