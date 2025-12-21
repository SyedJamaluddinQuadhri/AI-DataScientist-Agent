import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLUtils:
    """Advanced machine learning utilities and helper functions"""
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        self.feature_selectors = {
            'f_classif': f_classif,
            'f_regression': f_regression,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression
        }
    
    def create_cv_strategy(self, task_type: str, n_splits: int = 5, random_state: int = 42):
        """Create appropriate cross-validation strategy"""
        if task_type == 'classification':
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect and categorize feature types"""
        feature_types = {
            'numeric': [],
            'categorical': [],
            'binary': [],
            'datetime': [],
            'text': [],
            'constant': [],
            'high_cardinality': []
        }
        
        for col in df.columns:
            col_data = df[col]
            
            # Check for constant features
            if col_data.nunique() <= 1:
                feature_types['constant'].append(col)
                continue
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_types['datetime'].append(col)
                continue
            
            # Check for numeric
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.nunique() == 2:
                    feature_types['binary'].append(col)
                else:
                    feature_types['numeric'].append(col)
                continue
            
            # Check for categorical
            if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                unique_ratio = col_data.nunique() / len(col_data)
                
                if col_data.nunique() == 2:
                    feature_types['binary'].append(col)
                elif unique_ratio > 0.5 and col_data.nunique() > 50:
                    feature_types['high_cardinality'].append(col)
                elif self._is_text_column(col_data):
                    feature_types['text'].append(col)
                else:
                    feature_types['categorical'].append(col)
        
        return feature_types
    
    def suggest_preprocessing_steps(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Suggest preprocessing steps based on data characteristics"""
        suggestions = {
            'scaling': None,
            'feature_selection': None,
            'dimensionality_reduction': None,
            'encoding': [],
            'missing_value_strategy': {},
            'outlier_treatment': False,
            'feature_engineering': []
        }
        
        feature_types = self.detect_feature_types(df)
        
        # Scaling suggestions
        numeric_cols = feature_types['numeric']
        if numeric_cols:
            # Check for different scales
            scales = df[numeric_cols].std()
            if scales.max() / scales.min() > 100:
                suggestions['scaling'] = 'standard'
            elif df[numeric_cols].min().min() < 0:
                suggestions['scaling'] = 'standard'
            else:
                suggestions['scaling'] = 'minmax'
        
        # Feature selection suggestions
        n_features = len(df.columns) - (1 if target_column else 0)
        n_samples = len(df)
        
        if n_features > n_samples:
            suggestions['feature_selection'] = 'univariate'
        elif n_features > 50:
            suggestions['feature_selection'] = 'model_based'
        
        # Dimensionality reduction
        if n_features > 20 and n_samples > 100:
            suggestions['dimensionality_reduction'] = 'pca'
        
        # Encoding suggestions
        if feature_types['categorical']:
            suggestions['encoding'].append('one_hot_encoding')
        if feature_types['high_cardinality']:
            suggestions['encoding'].append('target_encoding')
        
        # Missing value strategy
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0:
                if missing_pct > 0.5:
                    suggestions['missing_value_strategy'][col] = 'drop_column'
                elif df[col].dtype in ['int64', 'float64']:
                    if missing_pct < 0.1:
                        suggestions['missing_value_strategy'][col] = 'knn_imputation'
                    else:
                        suggestions['missing_value_strategy'][col] = 'median_imputation'
                else:
                    suggestions['missing_value_strategy'][col] = 'mode_imputation'
        
        # Outlier treatment
        if numeric_cols:
            outlier_counts = self._detect_outliers_count(df[numeric_cols])
            if outlier_counts > len(df) * 0.05:  # More than 5% outliers
                suggestions['outlier_treatment'] = True
        
        # Feature engineering suggestions
        if feature_types['datetime']:
            suggestions['feature_engineering'].append('datetime_features')
        if len(numeric_cols) > 1:
            suggestions['feature_engineering'].append('polynomial_features')
        if feature_types['text']:
            suggestions['feature_engineering'].append('text_features')
        
        return suggestions
    
    def create_feature_selector(self, method: str, k: int = 10, task_type: str = 'classification'):
        """Create feature selector based on method and task type"""
        if method == 'univariate':
            if task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
            return SelectKBest(score_func=score_func, k=k)
        
        elif method == 'mutual_info':
            if task_type == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
            return SelectKBest(score_func=score_func, k=k)
        
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
    
    def perform_dimensionality_reduction(self, X: np.ndarray, method: str = 'pca', 
                                       n_components: Optional[int] = None) -> Tuple[np.ndarray, Any]:
        """Perform dimensionality reduction"""
        if method == 'pca':
            if n_components is None:
                # Determine optimal number of components
                pca_temp = PCA()
                pca_temp.fit(X)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= 0.95) + 1
                n_components = min(n_components, X.shape[1] // 2)
            
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            
        elif method == 'tsne':
            n_components = min(n_components or 2, 3)
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
            X_reduced = reducer.fit_transform(X)
            
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        return X_reduced, reducer
    
    def calculate_feature_importance_ensemble(self, models: List[Any], feature_names: List[str]) -> Dict[str, float]:
        """Calculate ensemble feature importance from multiple models"""
        importance_scores = {}
        
        for feature in feature_names:
            importance_scores[feature] = 0.0
        
        valid_models = 0
        
        for model in models:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    importance_scores[feature] += importances[i]
                valid_models += 1
            elif hasattr(model, 'coef_'):
                coefs = model.coef_
                if coefs.ndim > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)
                else:
                    coefs = np.abs(coefs)
                for i, feature in enumerate(feature_names):
                    importance_scores[feature] += coefs[i]
                valid_models += 1
        
        # Average the importance scores
        if valid_models > 0:
            for feature in importance_scores:
                importance_scores[feature] /= valid_models
        
        # Normalize to sum to 1
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for feature in importance_scores:
                importance_scores[feature] /= total_importance
        
        # Sort by importance
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def detect_data_drift(self, X_train: pd.DataFrame, X_new: pd.DataFrame, 
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Detect data drift between training and new data"""
        from scipy import stats
        
        drift_results = {
            'overall_drift': False,
            'drifted_features': [],
            'drift_scores': {},
            'recommendations': []
        }
        
        for col in X_train.columns:
            if col in X_new.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    # Use Kolmogorov-Smirnov test for numeric features
                    ks_stat, p_value = stats.ks_2samp(X_train[col].dropna(), X_new[col].dropna())
                    drift_score = ks_stat
                else:
                    # Use chi-square test for categorical features
                    train_counts = X_train[col].value_counts(normalize=True)
                    new_counts = X_new[col].value_counts(normalize=True)
                    
                    # Align the value counts
                    all_values = set(train_counts.index) | set(new_counts.index)
                    train_aligned = [train_counts.get(val, 0) for val in all_values]
                    new_aligned = [new_counts.get(val, 0) for val in all_values]
                    
                    try:
                        chi2_stat, p_value = stats.chisquare(new_aligned, train_aligned)
                        drift_score = chi2_stat / len(all_values)  # Normalize
                    except:
                        drift_score = 0.0
                
                drift_results['drift_scores'][col] = drift_score
                
                if drift_score > threshold:
                    drift_results['drifted_features'].append(col)
                    drift_results['overall_drift'] = True
        
        # Generate recommendations
        if drift_results['overall_drift']:
            drift_results['recommendations'].append("Data drift detected. Consider retraining the model.")
            drift_results['recommendations'].append("Monitor model performance closely.")
            
            if len(drift_results['drifted_features']) > len(X_train.columns) * 0.5:
                drift_results['recommendations'].append("Significant drift detected. Full model retraining recommended.")
        
        return drift_results
    
    def generate_synthetic_data(self, X: pd.DataFrame, y: pd.Series, 
                              method: str = 'smote', random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data for imbalanced datasets"""
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
            elif method == 'adasyn':
                from imblearn.over_sampling import ADASYN
                adasyn = ADASYN(random_state=random_state)
                X_resampled, y_resampled = adasyn.fit_resample(X, y)
                
            else:
                raise ValueError(f"Unsupported synthetic data generation method: {method}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            
        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping synthetic data generation.")
            return X, y
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {str(e)}")
            return X, y
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if a column contains text data"""
        if series.dtype != 'object':
            return False
        
        # Sample some values and check average length
        sample_values = series.dropna().head(100)
        if len(sample_values) == 0:
            return False
        
        avg_length = sample_values.astype(str).str.len().mean()
        return avg_length > 20  # Consider as text if average length > 20 characters
    
    def _detect_outliers_count(self, df: pd.DataFrame) -> int:
        """Count total outliers in numeric DataFrame using IQR method"""
        outlier_count = 0
        
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count += len(outliers)
        
        return outlier_count
    
    def save_preprocessor(self, preprocessor: Any, filepath: str):
        """Save preprocessing pipeline"""
        try:
            joblib.dump(preprocessor, filepath)
            logger.info(f"Preprocessor saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {str(e)}")
    
    def load_preprocessor(self, filepath: str) -> Any:
        """Load preprocessing pipeline"""
        try:
            preprocessor = joblib.load(filepath)
            logger.info(f"Preprocessor loaded from {filepath}")
            return preprocessor
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {str(e)}")
            return None
