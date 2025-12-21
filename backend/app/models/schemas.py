from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class DatasetInfo(BaseModel):
    filename: str
    size: int
    rows: int
    columns: int
    file_type: str
    upload_time: datetime

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]

class DataQualityReport(BaseModel):
    total_rows: int
    total_columns: int
    missing_values_count: int
    duplicate_rows: int
    columns_info: List[ColumnInfo]
    data_types: Dict[str, str]
    memory_usage: str

class EDARequest(BaseModel):
    dataset_id: str
    custom_prompt: Optional[str] = None
    include_correlation_analysis: bool = True
    include_distribution_analysis: bool = True
    include_outlier_detection: bool = True

class EDAResult(BaseModel):
    dataset_id: str
    summary_stats: Dict[str, Any]
    correlations: Optional[Dict[str, Any]] = None
    outliers: Optional[Dict[str, List]] = None
    insights: List[str]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]

class ModelingRequest(BaseModel):
    dataset_id: str
    task_type: TaskType
    target_column: str
    feature_columns: Optional[List[str]] = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=3, le=10)
    algorithms: Optional[List[str]] = None
    hyperparameter_tuning: bool = True
    max_trials: int = Field(default=50, ge=10, le=200)

class ModelResult(BaseModel):
    model_id: str
    algorithm: str
    task_type: TaskType
    performance_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    learning_curves: Optional[Dict[str, Any]] = None
    hyperparameters: Dict[str, Any]

class VisualizationRequest(BaseModel):
    dataset_id: str
    chart_type: str
    columns: List[str]
    title: Optional[str] = None
    custom_config: Optional[Dict[str, Any]] = None

class ReportRequest(BaseModel):
    dataset_id: str
    include_eda: bool = True
    include_modeling: bool = True
    custom_sections: Optional[List[str]] = None
    format_type: str = Field(default="html", pattern="^(html|pdf|markdown)$")
