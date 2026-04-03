from fastapi import APIRouter, HTTPException
from typing import Optional
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from app.services.eda_engine import EDAEngine
from app.services.data_processor import DataProcessor
from app.models.schemas import EDARequest, EDAResult
from app.core.config import settings
from app.core.exceptions import DataProcessingException
from fastapi.responses import Response
import io
from datetime import datetime


logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
eda_engine = EDAEngine()
data_processor = DataProcessor()

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
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

# ===== NOW THE ENDPOINT FUNCTION =====
@router.post("/{dataset_id}/generate-report")
async def generate_report(
    dataset_id: str,
    format_type: str = "pdf",
    include_eda: bool = True,
    include_modeling: bool = True,
    model_id: str = None
):
    """Generate comprehensive report in specified format"""
    try:
        logger.info(f"=== Report Generation Started ===")
        logger.info(f"Dataset ID: {dataset_id}")
        logger.info(f"Format: {format_type}")
        
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
        
        logger.info(f"Found dataset: {file_path}")
        
        # Load dataset
        df = data_processor.load_dataset(str(file_path))
        logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Prepare report data
        report_data = {
            'metadata': {
                'report_type': 'comprehensive',
                'generated_at': datetime.now().isoformat(),
                'dataset_id': dataset_id,
                'dataset_name': file_path.name
            },
            'dataset_info': {
                'dataset_id': dataset_id,
                'rows': len(df),
                'columns': len(df.columns),
                'original_filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower().lstrip('.')
            }
        }
        
        # Add EDA results if requested
        if include_eda:
            try:
                logger.info("Performing EDA for report...")
                eda_results = eda_engine.perform_comprehensive_eda(df)
                report_data['eda_results'] = eda_results
                logger.info("EDA completed successfully")
            except Exception as e:
                logger.warning(f"EDA generation failed: {str(e)}")
                report_data['eda_results'] = {'error': str(e)}
        
        # Initialize report generator
        from app.services.report_generator import ReportGenerator
        report_generator = ReportGenerator()
        
        # Load model results if requested
        models_results_data = {}
        if include_modeling and model_id:
            try:
                model_dir = Path(f"models/{model_id}")
                if (model_dir / "results.json").exists():
                    import json
                    with open(model_dir / "results.json", 'r') as f:
                        models_results_data = json.load(f)
                        logger.info(f"Loaded model results for {model_id}")
            except Exception as e:
                logger.warning(f"Could not load model results for {model_id}: {str(e)}")

        # Generate report
        logger.info(f"Generating {format_type} report...")
        report = report_generator.generate_comprehensive_report(
            dataset_info=report_data['dataset_info'],
            eda_results=report_data.get('eda_results', {}),
            model_results=models_results_data,
            format_type=format_type
        )
        
        logger.info(f"Report generated successfully")
        
        # Prepare response based on format
        if format_type == 'pdf':
            media_type = 'application/pdf'
            filename = f"ai_ds_report_{dataset_id}.pdf"
            content = report['content']
            logger.info(f"PDF size: {len(content)} bytes")
        elif format_type == 'html':
            media_type = 'text/html'
            filename = f"ai_ds_report_{dataset_id}.html"
            content = report['content']
            if isinstance(content, str):
                content = content.encode('utf-8')
            logger.info(f"HTML size: {len(content)} bytes")
        else:  # json
            media_type = 'application/json'
            filename = f"ai_ds_report_{dataset_id}.json"
            safe_report = convert_numpy_types(report)
            content = json.dumps(safe_report, indent=2).encode('utf-8')
            logger.info(f"JSON size: {len(content)} bytes")
        
        logger.info(f"=== Report Generation Completed: {filename} ===")
        
        # Return file as download
        return Response(
            content=content,
            media_type=media_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': media_type,
                'Content-Length': str(len(content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.post("/{dataset_id}/eda", response_model=dict)
async def perform_eda(dataset_id: str, 
                     target_column: Optional[str] = None,
                     custom_prompt: Optional[str] = None):
    """Perform comprehensive Exploratory Data Analysis"""
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
        
        # Validate dataset
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}"
            )
        
        # Perform comprehensive EDA
        eda_results = eda_engine.perform_comprehensive_eda(
            df, target_column=target_column, custom_prompt=custom_prompt
        )
        safe_eda_results = convert_numpy_types(eda_results)
        
        
        
        return {
            "dataset_id": dataset_id,
            "eda_results": safe_eda_results,
            "target_column": target_column,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
        
    except DataProcessingException as e:
        logger.error(f"EDA error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EDA analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")


@router.get("/{dataset_id}/summary")
async def get_dataset_summary(dataset_id: str):
    """Get dataset summary statistics"""
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
        
        # Generate summary
        summary = {
            "basic_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
            },
            "missing_values": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
            },
            "data_types": {
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
            },
            "basic_statistics": df.describe(include='all').to_dict()
        }
        
        return convert_numpy_types({
            "dataset_id": dataset_id,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/correlations")
async def get_correlations(dataset_id: str):
    """Get correlation analysis"""
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
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {
                "dataset_id": dataset_id,
                "message": "No numeric columns found for correlation analysis"
            }
        
        # Calculate correlations
        correlation_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        return convert_numpy_types({
            "dataset_id": dataset_id,
            "correlation_matrix": correlation_matrix.round(3).to_dict(),
            "highly_correlated_pairs": high_corr_pairs,
            "correlation_statistics": {
                "max_correlation": float(correlation_matrix.abs().max().max()),
                "min_correlation": float(correlation_matrix.min().min()),
                "mean_abs_correlation": float(correlation_matrix.abs().mean().mean())
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting correlations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/outliers")
async def detect_outliers(dataset_id: str):
    """Detect outliers in the dataset"""
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
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        outlier_analysis = {}
        
        for col in numeric_columns:
            data = df[col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            outlier_analysis[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_indices': outliers.index.tolist()[:20],  # Limit to 20
                'outlier_values': outliers.head(20).tolist()
            }
        
        return convert_numpy_types({
            "dataset_id": dataset_id,
            "outlier_analysis": outlier_analysis,
            "total_outliers": sum(analysis['outlier_count'] for analysis in outlier_analysis.values())
        })
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{dataset_id}/custom-analysis")
async def custom_analysis(dataset_id: str, prompt: str):
    """Perform custom analysis based on user prompt"""
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
        
        # This is a placeholder for custom analysis
        # In a real implementation, you might use LLMs to interpret the prompt
        analysis_results = {
            "prompt": prompt,
            "dataset_shape": df.shape,
            "analysis_type": "custom",
            "results": {
                "message": "Custom analysis functionality would be implemented here",
                "suggestions": [
                    "Implement natural language query processing",
                    "Use LLM to interpret user prompts",
                    "Generate custom visualizations based on requests"
                ]
            }
        }
        
        return convert_numpy_types({
            "dataset_id": dataset_id,
            "custom_analysis": analysis_results
        })
        
    except Exception as e:
        logger.error(f"Error in custom analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
