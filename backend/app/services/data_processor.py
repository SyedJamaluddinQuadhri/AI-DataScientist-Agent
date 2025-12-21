import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from app.core.exceptions import DataProcessingException
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataProcessor:
    """Advanced data processing and preprocessing engine"""
    
    def __init__(self):
        self.supported_formats = {
            'csv': self._read_csv,
            'xlsx': self._read_excel,
            'xls': self._read_excel,
            'json': self._read_json,
            'parquet': self._read_parquet,
            'arff': self._read_arff,
            'tsv': self._read_tsv
        }
        
    def load_dataset(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load dataset from various file formats"""
        try:
            file_extension = Path(file_path).suffix.lower().lstrip('.')
            
            if file_extension not in self.supported_formats:
                raise DataProcessingException(
                    f"Unsupported file format: {file_extension}",
                    {"supported_formats": list(self.supported_formats.keys())}
                )
            
            df = self.supported_formats[file_extension](file_path, **kwargs)
            logger.info(f"Successfully loaded dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise DataProcessingException(f"Failed to load dataset: {str(e)}")
    
    def _read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file with intelligent parameter detection"""
        # Try different encodings and separators
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, **kwargs)
                    if df.shape[1] > 1:  # Valid separation
                        return df
                except:
                    continue
        
        # Fallback to default
        return pd.read_csv(file_path, **kwargs)
    
    def _read_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Excel file"""
        return pd.read_excel(file_path, **kwargs)
    
    def _read_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read JSON file"""
        try:
            return pd.read_json(file_path, **kwargs)
        except:
            # Try reading as lines
            return pd.read_json(file_path, lines=True, **kwargs)
    
    def _read_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet file"""
        return pd.read_parquet(file_path, **kwargs)
    
    def _read_arff(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read ARFF file"""
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Convert byte strings to regular strings
        for col in df.select_dtypes(include=['object']):
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
        
        return df
    
    def _read_tsv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read TSV file"""
        return pd.read_csv(file_path, sep='\t', **kwargs)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        try:
        # Handle empty dataframe
            if len(df) == 0:
                raise DataProcessingException("Cannot analyze empty dataset")
        
            quality_report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values_count': int(df.isnull().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum()),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'columns_info': [],
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_percentage': {},
                'unique_counts': {},
                'constant_columns': [],
                'high_cardinality_columns': [],
                'potential_id_columns': []
            }

        # Calculate missing percentage safely
            total_cells = len(df) * len(df.columns)
            if total_cells > 0:
                quality_report['missing_percentage'] = {
                    col: float((df[col].isnull().sum() / len(df) * 100))
                    for col in df.columns
                }
        
        # Calculate unique counts
            quality_report['unique_counts'] = {
                col: int(df[col].nunique()) for col in df.columns
            }
        
        # Analyze each column
            for col in df.columns:
                null_count = int(df[col].isnull().sum())
                null_percentage = float(null_count / len(df) * 100) if len(df) > 0 else 0.0
                unique_count = int(df[col].nunique())
            
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'null_count': null_count,
                    'null_percentage': null_percentage,
                    'unique_count': unique_count,
                    'sample_values': [
                        None if pd.isna(val) else (int(val) if isinstance(val, np.integer) else float(val) if isinstance(val, np.floating) else val)
                        for val in df[col].dropna().head(5).tolist()
                    ]
                }
            
            # Check for constant columns
                if unique_count <= 1:
                    quality_report['constant_columns'].append(col)
            
            # Check for high cardinality
                cardinality_ratio = unique_count / len(df) if len(df) > 0 else 0
                if cardinality_ratio > 0.9 and unique_count > 100:
                    quality_report['high_cardinality_columns'].append(col)
            
            # Check for potential ID columns
                if (unique_count == len(df) or 
                    col.lower() in ['id', 'index', 'key'] or
                    'id' in col.lower()):
                    quality_report['potential_id_columns'].append(col)
            
                quality_report['columns_info'].append(col_info)
        
            return quality_report
        
        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            raise DataProcessingException(f"Data quality analysis failed: {str(e)}")

    
    def preprocess_data(self, df: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       strategy: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Advanced preprocessing with multiple strategies"""
        try:
            preprocessing_report = {
                'steps_applied': [],
                'columns_dropped': [],
                'columns_transformed': [],
                'imputation_strategy': {},
                'encoding_strategy': {}
            }
            
            df_processed = df.copy()
            
            # Step 1: Handle constant columns
            constant_cols = [col for col in df_processed.columns 
                           if df_processed[col].nunique() <= 1]
            if constant_cols:
                df_processed = df_processed.drop(columns=constant_cols)
                preprocessing_report['columns_dropped'].extend(constant_cols)
                preprocessing_report['steps_applied'].append('removed_constant_columns')
            
            # Step 2: Handle missing values
            df_processed, imputation_info = self._handle_missing_values(
                df_processed, strategy
            )
            preprocessing_report['imputation_strategy'] = imputation_info
            preprocessing_report['steps_applied'].append('missing_value_imputation')
            
            # Step 3: Encode categorical variables
            df_processed, encoding_info = self._encode_categorical_variables(
                df_processed, target_column
            )
            preprocessing_report['encoding_strategy'] = encoding_info
            preprocessing_report['steps_applied'].append('categorical_encoding')
            
            # Step 4: Handle outliers (optional)
            if strategy == 'auto' or 'outlier' in strategy:
                df_processed = self._handle_outliers(df_processed)
                preprocessing_report['steps_applied'].append('outlier_treatment')
            
            return df_processed, preprocessing_report
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise DataProcessingException(f"Preprocessing failed: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame, 
                              strategy: str) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with intelligent strategies"""
        imputation_info = {}
        df_imputed = df.copy()
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct == 0:
                continue
            
            if missing_pct > 0.5:
                # Drop columns with >50% missing values
                df_imputed = df_imputed.drop(columns=[col])
                imputation_info[col] = 'dropped_high_missing'
                continue
            
            # Choose imputation strategy based on data type
            if df[col].dtype in ['int64', 'float64']:
                if missing_pct < 0.1:
                    # Use KNN imputation for low missing percentage
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                    imputation_info[col] = 'knn_imputation'
                else:
                    # Use median for moderate missing percentage
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                    imputation_info[col] = 'median_imputation'
            else:
                # Use mode for categorical variables
                mode_value = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else 'Unknown'
                df_imputed[col] = df_imputed[col].fillna(mode_value)
                imputation_info[col] = 'mode_imputation'
        
        return df_imputed, imputation_info
    
    def _encode_categorical_variables(self, df: pd.DataFrame, 
                                    target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables intelligently"""
        encoding_info = {}
        df_encoded = df.copy()
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col == target_column:
                # Label encode target variable
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoding_info[col] = 'label_encoding'
            else:
                unique_count = df[col].nunique()
                
                if unique_count <= 10:
                    # One-hot encode low cardinality categorical variables
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                    encoding_info[col] = 'one_hot_encoding'
                else:
                    # Label encode high cardinality categorical variables
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    encoding_info[col] = 'label_encoding'
        
        return df_encoded, encoding_info
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        df_outliers = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_outliers[col] = np.where(df_outliers[col] < lower_bound, lower_bound, df_outliers[col])
            df_outliers[col] = np.where(df_outliers[col] > upper_bound, upper_bound, df_outliers[col])
        
        return df_outliers
    
    def suggest_target_column(self, df: pd.DataFrame) -> List[str]:
        """Suggest potential target columns based on data characteristics"""
        suggestions = []
        
        for col in df.columns:
            # Binary classification target
            if df[col].nunique() == 2:
                suggestions.append({
                    'column': col,
                    'task_type': 'binary_classification',
                    'confidence': 0.9
                })
            
            # Multi-class classification target
            elif df[col].nunique() <= 20 and df[col].dtype == 'object':
                suggestions.append({
                    'column': col,
                    'task_type': 'multiclass_classification',
                    'confidence': 0.7
                })
            
            # Regression target
            elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 20:
                suggestions.append({
                    'column': col,
                    'task_type': 'regression',
                    'confidence': 0.6
                })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions
