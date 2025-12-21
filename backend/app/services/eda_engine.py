import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Any, Tuple
import logging
import base64
import io
import json
from app.services.visualization_engine import VisualizationEngine
from app.core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)

class EDAEngine:
    """Advanced Exploratory Data Analysis Engine"""
    
    def __init__(self):
        self.visualizations = []

        
    def perform_comprehensive_eda(self, df: pd.DataFrame, 
                                target_column: str = None,
                                custom_prompt: str = None) -> Dict[str, Any]:
        """Perform comprehensive EDA"""
        try:
            eda_results = {
                'dataset_overview': {},
                'descriptive_statistics': {},
                'missing_value_analysis': {},
                'correlation_analysis': {},
                'distribution_analysis': {},
                'outlier_analysis': {},
                'categorical_analysis': {},
                'feature_relationships': {},
                'data_insights': [],
                'visualizations': [],
                'recommendations': []
            }
        
        # Safely execute each analysis step
            try:
                eda_results['dataset_overview'] = self._get_dataset_overview(df)
            except Exception as e:
                logger.error(f"Dataset overview failed: {str(e)}")
                eda_results['dataset_overview'] = {'error': str(e)}
        
            try:
                eda_results['descriptive_statistics'] = self._get_descriptive_statistics(df)
            except Exception as e:
                logger.error(f"Descriptive statistics failed: {str(e)}")
                eda_results['descriptive_statistics'] = {'error': str(e)}
        
            try:
                eda_results['missing_value_analysis'] = self._analyze_missing_values(df)
            except Exception as e:
                logger.error(f"Missing value analysis failed: {str(e)}")
                eda_results['missing_value_analysis'] = {'error': str(e)}
        
            try:
                eda_results['correlation_analysis'] = self._correlation_analysis(df)
            except Exception as e:
                logger.error(f"Correlation analysis failed: {str(e)}")
                eda_results['correlation_analysis'] = {'error': str(e)}
        
            try:
                eda_results['distribution_analysis'] = self._distribution_analysis(df)
            except Exception as e:
                logger.error(f"Distribution analysis failed: {str(e)}")
                eda_results['distribution_analysis'] = {'error': str(e)}
        
            try:
                eda_results['outlier_analysis'] = self._outlier_analysis(df)
            except Exception as e:
                logger.error(f"Outlier analysis failed: {str(e)}")
                eda_results['outlier_analysis'] = {'error': str(e)}
        
            try:
                eda_results['categorical_analysis'] = self._categorical_analysis(df)
            except Exception as e:
                logger.error(f"Categorical analysis failed: {str(e)}")
                eda_results['categorical_analysis'] = {'error': str(e)}
        
            try:
                eda_results['feature_relationships'] = self._analyze_feature_relationships(df, target_column)
            except Exception as e:
                logger.error(f"Feature relationships failed: {str(e)}")
                eda_results['feature_relationships'] = {'error': str(e)}
        
            try:
                eda_results['data_insights'] = self._generate_insights(df, target_column)
            except Exception as e:
                logger.error(f"Insights generation failed: {str(e)}")
                eda_results['data_insights'] = [f"Error generating insights: {str(e)}"]
        
            try:
                logger.info("Starting visualization generation...")
                visualizations = self._generate_visualizations(df, target_column)
                eda_results['visualizations'] = visualizations
                logger.info(f"Visualization generation complete: {len(visualizations)} charts created")
            except Exception as e:
                logger.error(f"Visualization creation failed: {str(e)}", exc_info=True)
                eda_results['visualizations'] = []
        
            try:
                eda_results['recommendations'] = self._generate_recommendations(df, target_column)
            except Exception as e:
                logger.error(f"Recommendations generation failed: {str(e)}")
                eda_results['recommendations'] = [f"Error generating recommendations: {str(e)}"]
        
            if custom_prompt:
                try:
                    eda_results['custom_analysis'] = self._custom_analysis(df, custom_prompt)
                except Exception as e:
                    logger.error(f"Custom analysis failed: {str(e)}")
                    eda_results['custom_analysis'] = {'error': str(e)}
        
            return eda_results
        
        except Exception as e:
            logger.error(f"Error in comprehensive EDA: {str(e)}")
            raise DataProcessingException(f"EDA failed: {str(e)}")

            
        except Exception as e:
            logger.error(f"Error in comprehensive EDA: {str(e)}")
            raise DataProcessingException(f"EDA failed: {str(e)}")
    
    def _get_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset overview"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _get_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive descriptive statistics"""
        numeric_stats = df.describe().to_dict()
        
        # Additional statistics for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_stats[col].update({
                'skewness': float(stats.skew(df[col].dropna())),
                'kurtosis': float(stats.kurtosis(df[col].dropna())),
                'variance': float(df[col].var()),
                'coefficient_of_variation': float(df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0
            })
        
        # Statistics for categorical columns
        categorical_stats = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            categorical_stats[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency_of_most_frequent': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                'least_frequent': df[col].value_counts().index[-1] if len(df[col].value_counts()) > 0 else None
            }
        
        return {
            'numeric_statistics': numeric_stats,
            'categorical_statistics': categorical_stats
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values patterns"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df) * 100).round(2)
        
        # Missing value patterns
        missing_patterns = df.isnull().value_counts().head(10).to_dict()
        
        return {
            'missing_counts': missing_data.to_dict(),
            'missing_percentages': missing_percentage.to_dict(),
            'columns_with_missing': missing_data[missing_data > 0].index.tolist(),
            'total_missing': missing_data.sum(),
            'missing_patterns': {str(k): v for k, v in missing_patterns.items()}
        }
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numeric columns for correlation analysis'}
        
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
        
        return {
            'correlation_matrix': correlation_matrix.round(3).to_dict(),
            'highly_correlated_pairs': high_corr_pairs,
            'max_correlation': float(correlation_matrix.abs().max().max()),
            'min_correlation': float(correlation_matrix.min().min())
        }
    
    def _distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
        distribution_analysis = {}
    
        for col in numeric_columns:
            try:
                data = df[col].dropna()
            
                if len(data) < 3:
                    # Not enough data for statistical tests
                    distribution_analysis[col] = {
                        'skewness': 0.0,
                        'kurtosis': 0.0,
                        'shapiro_wilk_p_value': 1.0,
                        'is_normal': False,
                        'anderson_darling_stat': 0.0,
                        'quartiles': {
                            'Q1': float(data.quantile(0.25)) if len(data) > 0 else 0.0,
                            'Q2': float(data.quantile(0.5)) if len(data) > 0 else 0.0,
                            'Q3': float(data.quantile(0.75)) if len(data) > 0 else 0.0
                        }
                    }
                    continue
            
            # Statistical tests for normality
                sample_size = min(5000, len(data))
                sample_data = data.sample(sample_size, random_state=42) if len(data) > sample_size else data
            
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                except Exception:
                    shapiro_p = 1.0
            
                try:
                    anderson_result = stats.anderson(data)
                    anderson_stat = float(anderson_result.statistic)
                except Exception:
                    anderson_stat = 0.0
            
                distribution_analysis[col] = {
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'shapiro_wilk_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05,
                    'anderson_darling_stat': anderson_stat,
                    'quartiles': {
                        'Q1': float(data.quantile(0.25)),
                        'Q2': float(data.quantile(0.5)),
                        'Q3': float(data.quantile(0.75))
                    }
                }
            except Exception as e:
                logger.warning(f"Distribution analysis failed for {col}: {str(e)}")
                distribution_analysis[col] = {
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'shapiro_wilk_p_value': 1.0,
                    'is_normal': False,
                    'anderson_darling_stat': 0.0,
                    'quartiles': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0}
                }
    
        return distribution_analysis

    
    def _outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze outliers"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_columns:
            data = df[col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            
            outlier_analysis[col] = {
                'iqr_outliers_count': len(iqr_outliers),
                'iqr_outliers_percentage': len(iqr_outliers) / len(data) * 100,
                'z_score_outliers_count': len(z_outliers),
                'z_score_outliers_percentage': len(z_outliers) / len(data) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': iqr_outliers.head(10).tolist()
            }
        
        return outlier_analysis
    
    def _categorical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical variables"""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        categorical_analysis = {}
        
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            
            categorical_analysis[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent_value': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.head(10).to_dict(),
                'cardinality_ratio': df[col].nunique() / len(df)
            }
        
        return categorical_analysis
    
    def _analyze_feature_relationships(self, df: pd.DataFrame, 
                                     target_column: str = None) -> Dict[str, Any]:
        """Analyze relationships between features"""
        if not target_column or target_column not in df.columns:
            return {'message': 'No target column specified or found'}
    
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        target_relationships = {}
    
        if target_column in numeric_columns:
            # Target is numeric - calculate correlations
            for col in numeric_columns:
                if col != target_column:
                    try:
                        correlation = df[col].corr(df[target_column])
                        if not pd.isna(correlation):
                            target_relationships[col] = {
                                'correlation_with_target': float(correlation),
                                'relationship_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
                            }
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation for {col}: {str(e)}")
                        continue
        else:
            # Target is categorical - use different measures
            for col in numeric_columns:
                try:
                    # ANOVA F-statistic for categorical target
                    groups = [df[df[target_column] == group][col].dropna().values 
                             for group in df[target_column].unique()]
                    # Filter out empty groups
                    groups = [g for g in groups if len(g) > 0]
                
                    if len(groups) > 1:
                        f_stat, p_val = stats.f_oneway(*groups)
                        target_relationships[col] = {
                            'anova_f_statistic': float(f_stat) if not pd.isna(f_stat) else 0.0,
                            'anova_p_value': float(p_val) if not pd.isna(p_val) else 1.0,
                            'significant': p_val < 0.05 if not pd.isna(p_val) else False
                        }
                except Exception as e:
                    logger.warning(f"Could not perform ANOVA for {col}: {str(e)}")
                    continue
    
        return target_relationships

    
    def _generate_insights(self, df: pd.DataFrame, 
                          target_column: str = None) -> List[str]:
        """Generate data insights automatically"""
        insights = []
        
        # Dataset size insights
        if len(df) < 1000:
            insights.append("Dataset is relatively small. Consider collecting more data for better model performance.")
        elif len(df) > 100000:
            insights.append("Large dataset detected. Consider using sampling techniques for faster analysis.")
        
        # Missing values insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            insights.append(f"High percentage of missing values ({missing_pct:.1f}%). Consider imputation strategies.")
        elif missing_pct < 1:
            insights.append("Dataset has very few missing values - excellent data quality.")
        
        # Correlation insights
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
            if high_corr.any().any():
                insights.append("High correlation detected between some features. Consider feature selection.")
        
        # Class imbalance insights (if target is categorical)
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
                value_counts = df[target_column].value_counts()
                imbalance_ratio = value_counts.max() / value_counts.min()
                if imbalance_ratio > 5:
                    insights.append("Significant class imbalance detected. Consider resampling techniques.")
        
        # Feature count insights
        if len(df.columns) > 100:
            insights.append("High-dimensional dataset. Consider dimensionality reduction techniques.")
        
        return insights
    def _generate_visualizations(self, df: pd.DataFrame, target_column: str = None) -> list:
        """Generate visualizations and return as Plotly JSON strings"""
    
        import json 
        visualizations = []
    
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
            logger.info(f"Generating visualizations: {len(num_cols)} numeric, {len(cat_cols)} categorical columns")
        
        # 1. Histogram for first numeric column
            if num_cols:
                try:
                    col = num_cols[0]
                    fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                    # CORRECTED: Use the built-in to_json() method
                    viz_json =  fig.to_json()
                    visualizations.append(viz_json)
                    logger.info(f"✓ Created histogram for {col}")
                except Exception as e:
                    logger.warning(f"Histogram failed: {str(e)}")
        
        # 2. Scatter plot for first two numeric columns
            if len(num_cols) >= 2:
                try:
                    fig = px.scatter(
                        df, 
                        x=num_cols[0], 
                        y=num_cols[1],
                        title=f"{num_cols[1]} vs {num_cols[0]}"
                    )
                    # CORRECTED: Ensured consistent use of to_json()
                    viz_json =   fig.to_json()
                    visualizations.append(viz_json)
                    logger.info(f"✓ Created scatter plot: {num_cols[0]} vs {num_cols[1]}")
                except Exception as e:
                    logger.warning(f"Scatter plot failed: {str(e)}")
        
        # 3. Correlation heatmap
            if len(num_cols) > 1:
                try:
                    # To handle potential non-numeric data in numeric-typed columns
                    corr_df = df[num_cols].select_dtypes(include=[np.number])
                    if not corr_df.empty:
                        corr = corr_df.corr()
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            title="Correlation Heatmap",
                            color_continuous_scale='RdBu_r',
                            aspect='auto'
                        )
                        # CORRECTED: Ensured consistent use of to_json()
                        viz_json =  fig.to_json()
                        visualizations.append(viz_json)
                        logger.info(f"✓ Created correlation heatmap")
                except Exception as e:
                    logger.warning(f"Heatmap failed: {str(e)}")
        
        # 4. Box plot (categorical vs numeric)
            if cat_cols and num_cols:
                try:
                    fig = px.box(
                        df,
                        x=cat_cols[0],
                        y=num_cols[0],
                        title=f"{num_cols[0]} by {cat_cols[0]}"
                    )
                    # CORRECTED: Ensured consistent use of to_json()
                    viz_json =  fig.to_json()
                    visualizations.append(viz_json)
                    logger.info(f"✓ Created box plot")
                except Exception as e:
                    logger.warning(f"Box plot failed: {str(e)}")
        
        # 5. Bar chart for categorical value counts
            if cat_cols:
                try:
                    value_counts = df[cat_cols[0]].value_counts().nlargest(10).reset_index()
                    value_counts.columns = [cat_cols[0], 'count']
                
                    fig = px.bar(
                        value_counts,
                        x=cat_cols[0],
                        y='count',
                        title=f"Top 10 values in {cat_cols[0]}"
                    )
                    # CORRECTED: Ensured consistent use of to_json()
                    viz_json = fig.to_json()
                    visualizations.append(viz_json)
                    logger.info(f"✓ Created bar chart")
                except Exception as e:
                    logger.warning(f"Bar chart failed: {str(e)}")
        
        # 6. Violin plot
            if cat_cols and num_cols:
                try:
                    fig = px.violin(
                        df,
                        x=cat_cols[0],
                        y=num_cols[0],
                        box=True,
                        points='outliers',
                        title=f"Violin plot: {num_cols[0]} by {cat_cols[0]}"
                    )
                    # CORRECTED: Ensured consistent use of to_json()
                    viz_json = fig.to_json()
                    visualizations.append(viz_json)
                    logger.info(f"✓ Created violin plot")
                except Exception as e:
                    logger.warning(f"Violin plot failed: {str(e)}")
        
        # 7. Additional histogram for second numeric column
            if len(num_cols) > 1:
                try:
                    col = num_cols[1]
                    fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                    # CORRECTED: Ensured consistent use of to_json()
                    viz_json = fig.to_json()
                    visualizations.append(viz_json)
                    logger.info(f"✓ Created second histogram for {col}")
                except Exception as e:
                    logger.warning(f"Second histogram failed: {str(e)}")
        
        # 8. Pie chart for categorical with low cardinality
            if cat_cols:
                try:
                    for cat_col in cat_cols[:2]:  # Try first 2 categorical columns
                        unique_count = df[cat_col].nunique()
                        if 1 < unique_count <= 10:
                            value_counts = df[cat_col].value_counts().reset_index()
                            value_counts.columns = [cat_col, 'count']
                        
                            fig = px.pie(
                                value_counts,
                                values='count',
                                names=cat_col,
                                title=f"Distribution of {cat_col}"
                            )
                            # CORRECTED: Ensured consistent use of to_json()
                            viz_json = fig.to_json()
                            visualizations.append(viz_json)
                            logger.info(f"✓ Created pie chart for {cat_col}")
                            break  # Only create one pie chart
                except Exception as e:
                    logger.warning(f"Pie chart failed: {str(e)}")
        
            logger.info(f"✅ Successfully created {len(visualizations)} visualizations")
        
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}", exc_info=True)
    
        return visualizations


    
    def _generate_recommendations(self, df: pd.DataFrame, 
                                target_column: str = None) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            recommendations.append("Implement robust missing value imputation strategies before modeling")
        
        # Feature engineering recommendations
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality_cols:
            recommendations.append("Consider target encoding or embedding techniques for high cardinality categorical features")
        
        # Modeling recommendations
        if target_column:
            if df[target_column].dtype == 'object':
                recommendations.append("Classification task detected. Consider algorithms like Random Forest, XGBoost, or Neural Networks")
            else:
                recommendations.append("Regression task detected. Consider algorithms like Random Forest, XGBoost, or Linear Regression")
        
        # Scaling recommendations
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            scales = numeric_df.std()
            if scales.max() / scales.min() > 100:
                recommendations.append("Features have very different scales. Consider standardization or normalization")
        
        # Sample size recommendations
        if len(df) < 1000:
            recommendations.append("Consider data augmentation techniques or collecting more data")
        
        return recommendations
    
    def _custom_analysis(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Perform custom analysis based on user prompt"""
        # This is a placeholder for custom analysis
        # In a real implementation, you might use LLMs to interpret the prompt
        return {
            'custom_prompt': prompt,
            'analysis_performed': 'Custom analysis would be implemented here based on the prompt',
            'results': 'Results would be generated based on the specific requirements'
        }
