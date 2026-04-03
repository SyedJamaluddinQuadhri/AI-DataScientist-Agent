import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import optuna
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
import joblib
import pickle
from typing import Dict, List, Any, Tuple, Optional
import logging
import uuid
from pathlib import Path
import json

from app.core.exceptions import ModelTrainingException
from app.core.config import settings

logger = logging.getLogger(__name__)

class MLEngine:
    """Advanced Machine Learning Engine with AutoML capabilities"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Define algorithm mappings
        self.classification_algorithms = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'xgboost': xgb.XGBClassifier,
            'lightgbm': lgb.LGBMClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'naive_bayes': GaussianNB,
            'decision_tree': DecisionTreeClassifier
        }
        
        self.regression_algorithms = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor,
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'svm': SVR,
            'knn': KNeighborsRegressor,
            'decision_tree': DecisionTreeRegressor
        }
        
        # Hyperparameter spaces for optimization
        self.hyperparameter_spaces = {
            'random_forest': {
                'n_estimators': (50, 300),
                'max_depth': (3, 10),
                'min_samples_split': (5, 20),
                'min_samples_leaf': (2, 10)
            },
            'xgboost': {
                'n_estimators': (50, 300),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.2),
                'subsample': (0.6, 0.9),
                'colsample_bytree': (0.6, 0.9),
                'reg_alpha': (0.1, 10.0),
                'reg_lambda': (0.1, 10.0)
            },
            'lightgbm': {
                'n_estimators': (50, 300),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.2),
                'feature_fraction': (0.6, 0.9),
                'bagging_fraction': (0.6, 0.9),
                'reg_alpha': (0.1, 10.0),
                'reg_lambda': (0.1, 10.0)
            }
        }
    
    def auto_ml_pipeline(self, df: pd.DataFrame,
                        target_column: str,
                        task_type: str = 'auto',
                        test_size: float = 0.2,
                        algorithms: list = None,
                        hyperparameter_tuning: bool = True,
                        max_trials: int = 50) -> dict:
        """Complete AutoML pipeline with robust handling of imbalanced classes"""
        try:
            # Prepare data
            X, y, task_type = self._prepare_data(df, target_column, task_type)
            
            # Validate minimum samples
            if len(X) < 10:
                raise ModelTrainingException(
                    "Dataset too small. Need at least 10 samples for training.",
                    task_type
                )
            
            # Handle classification with class imbalance
            if task_type == 'classification':
                class_counts = pd.Series(y).value_counts()
                min_class_count = class_counts.min()
                total_samples = len(y)
                
                logger.info(f"Class distribution before filtering: {class_counts.to_dict()}")
                
                # Strategy 1: Remove classes with only 1 sample
                if min_class_count < 2:
                    classes_to_remove = class_counts[class_counts < 2].index.tolist()
                    logger.warning(f"Removing classes with < 2 samples: {classes_to_remove}")
                    
                    # Filter out these classes
                    mask = ~y.isin(classes_to_remove)
                    X = X[mask]
                    y = y[mask]
                    
                    # Recheck class counts
                    class_counts = pd.Series(y).value_counts()
                    min_class_count = class_counts.min()
                    
                    logger.info(f"Class distribution after filtering: {class_counts.to_dict()}")
                    logger.info(f"Removed {(~mask).sum()} samples from low-count classes")
                
                # Validate we still have enough data
                if len(X) < 10:
                    raise ModelTrainingException(
                        f"After removing low-count classes, only {len(X)} samples remain. Need at least 10.",
                        task_type
                    )
                
                # Strategy 2: Adjust test_size if needed
                n_classes = len(class_counts)
                min_samples_needed = n_classes * 2  # At least 2 per class
                
                if len(X) < min_samples_needed:
                    raise ModelTrainingException(
                        f"Need at least {min_samples_needed} samples for {n_classes} classes, but only have {len(X)}.",
                        task_type
                    )
                
                # Calculate maximum safe test_size
                max_safe_test_size = (len(X) - n_classes) / len(X)
                if test_size > max_safe_test_size:
                    old_test_size = test_size
                    test_size = max(0.1, max_safe_test_size - 0.05)
                    logger.warning(f"Adjusted test_size from {old_test_size:.2f} to {test_size:.2f} for small dataset")
                
                # Strategy 3: Try stratified split, fallback to regular if it fails
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size, 
                        random_state=42, 
                        stratify=y
                    )
                    logger.info("Using stratified train-test split")
                except ValueError as e:
                    logger.warning(f"Stratified split failed: {e}. Using regular split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size, 
                        random_state=42
                    )
                    logger.info("Using regular train-test split")
                
            else:
                # Regression - regular split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=42
                )
            
            # Validate splits
            if len(X_train) < 3:
                raise ModelTrainingException(
                    f"Training set too small ({len(X_train)} samples). Need at least 3.",
                    task_type
                )
            
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select algorithms
            if not algorithms:
                algorithms = self._select_default_algorithms(task_type)
            
            # Adjust CV folds based on dataset size and class distribution
            if task_type == 'classification':
                # CV folds should not exceed the smallest class count in training data
                train_class_counts = pd.Series(y_train).value_counts()
                min_train_class = train_class_counts.min()
                cv_folds = min(5, min_train_class, len(X_train) // 10)
                cv_folds = max(2, cv_folds)  # At least 2 folds
            else:
                cv_folds = min(5, len(X_train) // 10)
                cv_folds = max(2, cv_folds)
            
            logger.info(f"Using {cv_folds}-fold cross-validation")
            
            # Train and evaluate models
            model_results = {}
            best_model = None
            best_score = -np.inf if task_type == 'regression' else 0
            
            for algo_name in algorithms:
                logger.info(f"Training {algo_name}...")
                
                try:
                    # Get model class
                    if task_type == 'classification':
                        model_class = self.classification_algorithms.get(algo_name)
                    else:
                        model_class = self.regression_algorithms.get(algo_name)
                    
                    if model_class is None:
                        logger.warning(f"Algorithm {algo_name} not found, skipping...")
                        continue
                    
                    # Hyperparameter tuning
                    if hyperparameter_tuning and algo_name in self.hyperparameter_spaces:
                        best_params = self._optimize_hyperparameters(
                            model_class, X_train_scaled, y_train, 
                            algo_name, task_type, max_trials, cv_folds
                        )
                        model = model_class(**best_params)
                    else:
                        model = model_class(random_state=42)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    results = self._evaluate_model(
                        model, X_train_scaled, X_test_scaled, 
                        y_train, y_test, task_type
                    )
                    
                    # Feature importance
                    feature_importance = self._get_feature_importance(
                        model, X.columns.tolist()
                    )
                    
                    model_results[algo_name] = {
                        'model': model,
                        'performance_metrics': results,
                        'feature_importance': feature_importance,
                        'hyperparameters': model.get_params() if hasattr(model, 'get_params') else {}
                    }
                    
                    # Update best model
                    primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
                    if results.get(primary_metric, 0) > best_score:
                        best_score = results[primary_metric]
                        best_model = algo_name
                        
                except Exception as e:
                    logger.error(f"Error training {algo_name}: {str(e)}")
                    continue
            
            if not model_results:
                raise ModelTrainingException(
                    "All models failed to train. Check your data quality and try again.",
                    task_type
                )
            
            # Generate comprehensive report
            ml_report = {
                'task_type': task_type,
                'target_column': target_column,
                'feature_columns': X.columns.tolist(),
                'dataset_info': {
                    'total_samples': len(df),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'n_features': X.shape[1],
                    'cv_folds_used': cv_folds,
                    'classes_removed': classes_to_remove if task_type == 'classification' and min_class_count < 2 else []
                },
                'model_results': {k: {**v, 'model': None} for k, v in model_results.items()},
                'best_model': best_model,
                'model_comparison': self._create_model_comparison(model_results, task_type),
                'recommendations': self._generate_ml_recommendations(model_results, task_type)
            }
            
            # Save best model
            if best_model:
                import uuid
                model_id = str(uuid.uuid4())
                self._save_model(model_results[best_model]['model'], scaler, model_id, ml_report)
                ml_report['model_id'] = model_id
            
            return ml_report
            
        except ModelTrainingException:
            raise
        except Exception as e:
            logger.error(f"Error in AutoML pipeline: {str(e)}")
            raise ModelTrainingException(f"AutoML pipeline failed: {str(e)}")


    def _prepare_data(self, df: pd.DataFrame, target_column: str, 
                     task_type: str) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Prepare data for modeling"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Auto-detect task type if not specified
        if task_type == 'auto':
            if y.dtype == 'object' or y.nunique() < 20:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Handle categorical variables in features
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            # Simple label encoding for now
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if task_type == 'classification' and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        
        return X, y, task_type
    
    def _select_default_algorithms(self, task_type: str) -> List[str]:
        """Select default algorithms based on task type"""
        if task_type == 'classification':
            return ['random_forest', 'xgboost', 'logistic_regression', 'gradient_boosting']
        else:
            return ['random_forest', 'xgboost', 'linear_regression', 'gradient_boosting']
    
    def _optimize_hyperparameters(self, model_class, X_train: np.ndarray, 
                                 y_train: np.ndarray, algo_name: str, 
                                 task_type: str, max_trials: int, cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            params = {}
            param_space = self.hyperparameter_spaces.get(algo_name, {})
            
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if param_name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            
            # Add random state for reproducibility
            params['random_state'] = 42
            
            try:
                # Create and train model
                model = model_class(**params)
                
                # Adjust CV folds if needed
                n_splits = min(cv_folds, len(y_train) // 2)
                if n_splits < 2:
                    n_splits = 2
                
                # Cross-validation score
                if task_type == 'classification':
                    # Check if we can do stratified CV
                    class_counts = pd.Series(y_train).value_counts()
                    if class_counts.min() >= n_splits:
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    else:
                        cv = n_splits  # Use regular KFold
                else:
                    cv = n_splits
                
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv,
                    scoring='accuracy' if task_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                
                return cv_scores.mean()
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0  # Return poor score for failed trials
        
        try:
            # Optimize with timeout
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=min(max_trials, 20), timeout=300, show_progress_bar=False)
            
            return study.best_params
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {str(e)}. Using defaults.")
            return {'random_state': 42}

        
        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_trials, timeout=300)  # 5 minute timeout
        
        return study.best_params
    
    def _evaluate_model(self, model, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray, 
                       task_type: str) -> Dict[str, float]:
        """Evaluate model performance"""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if task_type == 'classification':
            # Classification metrics
            results = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, average='weighted'),
                'recall': recall_score(y_test, y_test_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
                'train_accuracy': accuracy_score(y_train, y_train_pred)
            }
            
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            results['confusion_matrix'] = cm.tolist()
            
        else:
            # Regression metrics
            results = {
                'r2_score': r2_score(y_test, y_test_pred),
                'mean_squared_error': mean_squared_error(y_test, y_test_pred),
                'mean_absolute_error': mean_absolute_error(y_test, y_test_pred),
                'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_r2_score': r2_score(y_train, y_train_pred)
            }
        
        return results
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, importance in enumerate(importances):
                importance_dict[feature_names[i]] = float(importance)
        elif hasattr(model, 'coef_'):
            # For linear models
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs)
            for i, coef in enumerate(coefs):
                importance_dict[feature_names[i]] = float(coef)
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def _get_shap_values(self, model, X_sample: np.ndarray) -> Dict[str, Any]:
        """Calculate SHAP values for model interpretability"""
        if not globals().get('SHAP_AVAILABLE', False):
            return {}
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample[:50])  # Limit to 50 samples for performance
            
            return {
                'shap_values': shap_values.values.tolist(),
                'base_values': shap_values.base_values.tolist(),
                'data': shap_values.data.tolist()
            }
        except Exception as e:
            logger.warning(f"Could not calculate SHAP values: {str(e)}")
            return {}
    
    def _create_model_comparison(self, model_results: Dict, task_type: str) -> Dict[str, Any]:
        """Create model comparison summary"""
        comparison_data = []
        primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
        
        for algo_name, results in model_results.items():
            if 'performance_metrics' in results:
                comparison_data.append({
                    'algorithm': algo_name,
                    'primary_metric': results['performance_metrics'].get(primary_metric, 0),
                    'all_metrics': results['performance_metrics']
                })
        
        # Sort by primary metric
        comparison_data.sort(key=lambda x: x['primary_metric'], reverse=True)
        
        return {
            'ranking': comparison_data,
            'best_algorithm': comparison_data[0]['algorithm'] if comparison_data else None,
            'metric_comparison': {
                algo['algorithm']: algo['primary_metric'] 
                for algo in comparison_data
            }
        }
    
    def _generate_ml_recommendations(self, model_results: Dict, task_type: str) -> List[str]:
        """Generate ML recommendations based on results"""
        recommendations = []
        
        if not model_results:
            recommendations.append("No models were successfully trained. Check data quality and preprocessing.")
            return recommendations
        
        # Performance-based recommendations
        primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
        best_score = max(
            results['performance_metrics'].get(primary_metric, 0) 
            for results in model_results.values() 
            if 'performance_metrics' in results
        )
        
        if task_type == 'classification':
            if best_score < 0.7:
                recommendations.append("Low accuracy detected. Consider feature engineering, more data, or different algorithms.")
            elif best_score > 0.9:
                recommendations.append("High accuracy achieved. Check for overfitting with cross-validation.")
        else:
            if best_score < 0.5:
                recommendations.append("Low R² score. Consider feature engineering, polynomial features, or ensemble methods.")
            elif best_score > 0.95:
                recommendations.append("Very high R² score. Verify results and check for data leakage.")
        
        # Algorithm-specific recommendations
        if 'xgboost' in model_results and model_results['xgboost']['performance_metrics'].get(primary_metric, 0) == best_score:
            recommendations.append("XGBoost performed best. Consider further hyperparameter tuning.")
        
        if 'random_forest' in model_results and model_results['random_forest']['performance_metrics'].get(primary_metric, 0) == best_score:
            recommendations.append("Random Forest performed best. It provides good feature importance insights.")
        
        return recommendations
    
    def _save_model(self, model, scaler, model_id: str, ml_report: Dict = None):
        """Save trained model and scaler"""
        model_dir = Path(f"models/{model_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, model_dir / "model.pkl")
        
        # Save scaler
        joblib.dump(scaler, model_dir / "scaler.pkl")
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'model_type': type(model).__name__,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save results/report
        if ml_report:
            def convert_numpy_types(obj):
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

            safe_report = convert_numpy_types(ml_report)
            with open(model_dir / "results.json", 'w') as f:
                json.dump(safe_report, f, indent=2)
        
        logger.info(f"Model saved with ID: {model_id}")
    
    def predict(self, model_id: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with saved model"""
        try:
            model_dir = Path(f"models/{model_id}")
            
            # Load model and scaler
            model = joblib.load(model_dir / "model.pkl")
            scaler = joblib.load(model_dir / "scaler.pkl")
            
            # Preprocess and predict
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise ModelTrainingException(f"Prediction failed: {str(e)}")
