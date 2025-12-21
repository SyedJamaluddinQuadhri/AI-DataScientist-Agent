from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Optional
import uuid
import os
from pathlib import Path
import logging

from app.services.data_processor import DataProcessor
from app.models.schemas import DatasetInfo, DataQualityReport
from app.core.config import settings
from app.core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
data_processor = DataProcessor()

# Add this helper function to convert NumPy types
def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types"""
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

@router.post("/upload", response_model=dict)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process dataset"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower().lstrip('.')
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, 
                detail="File too large"
            )
        
        # Generate unique filename
        dataset_id = str(uuid.uuid4())
        filename = f"{dataset_id}.{file_extension}"
        file_path = Path(settings.UPLOAD_DIR) / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Load and analyze dataset
        df = data_processor.load_dataset(str(file_path))
        
        # Handle empty datasets
        if len(df) == 0:
            raise DataProcessingException("Dataset is empty. Please upload a valid dataset with data.")
        
        quality_report = data_processor.analyze_data_quality(df)
        
        # Create response
        dataset_info = {
            "dataset_id": dataset_id,
            "filename": filename,
            "original_filename": file.filename,
            "file_size": file_size,
            "rows": len(df),
            "columns": len(df.columns),
            "file_type": file_extension,
            "quality_report": quality_report,
            "column_info": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "sample_values": df[col].head(3).tolist()
                }
                for col in df.columns
            ],
            "preview": df.head(10).to_dict('records')
        }
        
        # Convert all NumPy types to Python native types before returning
        safe_dataset_info = convert_numpy_types(dataset_info)
        
        logger.info(f"Dataset uploaded successfully: {dataset_id}")
        return safe_dataset_info
        
    except DataProcessingException as e:
        logger.error(f"Data processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Update other endpoints similarly...


@router.get("/{dataset_id}/info")
async def get_dataset_info(dataset_id: str):
    """Get dataset information"""
    try:
        file_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"  # Assuming CSV for simplicity
        
        if not file_path.exists():
            # Try other extensions
            for ext in settings.ALLOWED_EXTENSIONS:
                test_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.{ext}"
                if test_path.exists():
                    file_path = test_path
                    break
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        df = data_processor.load_dataset(str(file_path))
        quality_report = data_processor.analyze_data_quality(df)
        
        return {
            "dataset_id": dataset_id,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "quality_report": quality_report,
            "preview": df.head(10).to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, rows: int = 10):
    """Get dataset preview"""
    try:
        # Find dataset file
        file_path = None
        for ext in settings.ALLOWED_EXTENSIONS:
            test_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.{ext}"
            if test_path.exists():
                file_path = test_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load and return preview
        df = data_processor.load_dataset(str(file_path))
        preview = df.head(rows).to_dict('records')
        
        return {
            "dataset_id": dataset_id,
            "preview": preview,
            "total_rows": len(df),
            "columns": df.columns.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{dataset_id}/preprocess")
async def preprocess_dataset(dataset_id: str, target_column: Optional[str] = None):
    """Preprocess dataset"""
    try:
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
        
        # Preprocess data
        df_processed, preprocessing_report = data_processor.preprocess_data(
            df, target_column=target_column
        )
        
        # Save preprocessed dataset
        processed_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}_processed.csv"
        df_processed.to_csv(processed_path, index=False)
        
        return {
            "dataset_id": dataset_id,
            "preprocessing_report": preprocessing_report,
            "processed_shape": df_processed.shape,
            "processed_columns": df_processed.columns.tolist(),
            "preview": df_processed.head(5).to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/suggestions")
async def get_target_suggestions(dataset_id: str):
    """Get target column suggestions"""
    try:
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
        suggestions = data_processor.suggest_target_column(df)
        
        return {
            "dataset_id": dataset_id,
            "target_suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete dataset and associated files"""
    try:
        deleted_files = []
        
        # Delete all files related to this dataset
        upload_dir = Path(settings.UPLOAD_DIR)
        for file_path in upload_dir.glob(f"{dataset_id}*"):
            file_path.unlink()
            deleted_files.append(str(file_path))
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {
            "dataset_id": dataset_id,
            "deleted_files": deleted_files,
            "status": "deleted"
        }
        
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
