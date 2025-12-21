import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import base64
import io
import json
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """Advanced visualization engine for data science workflows"""
    
    def __init__(self):
        # Color schemes
        self.color_schemes = {
            'default': px.colors.qualitative.Set3,
            'professional': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'gradient': px.colors.sequential.Viridis,
            'diverging': px.colors.diverging.RdBu
        }
        
        # Chart templates
        self.chart_templates = {
            'professional': {
                'layout': {
                    'font': {'family': 'Arial, sans-serif', 'size': 12},
                    'plot_bgcolor': 'white',
                    'paper_bgcolor': 'white',
                    'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60}
                }
            }
        }
        
        # Supported chart types
        self.chart_types = {
            'histogram': self.create_histogram,
            'scatter': self.create_scatter_plot,
            'box': self.create_box_plot,
            'violin': self.create_violin_plot,
            'heatmap': self.create_heatmap,
            'bar': self.create_bar_chart,
            'line': self.create_line_chart,
            'pie': self.create_pie_chart,
            'distribution': self.create_distribution_plot,
            'correlation': self.create_correlation_matrix,
            'pairplot': self.create_pairplot,
            'feature_importance': self.create_feature_importance_plot,
            'confusion_matrix': self.create_confusion_matrix,
            'roc_curve': self.create_roc_curve,
            'learning_curve': self.create_learning_curve,
            'pca': self.create_pca_plot,
            'tsne': self.create_tsne_plot
        }
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame, 
                                          target_column: Optional[str] = None,
                                          max_charts: int = 20) -> List[Dict[str, Any]]:
        """Create comprehensive set of visualizations for dataset"""
        try:
            visualizations = []
            
            # Data overview
            overview_charts = self._create_overview_charts(df)
            visualizations.extend(overview_charts)
            
            # Distribution analysis
            dist_charts = self._create_distribution_charts(df)
            visualizations.extend(dist_charts[:5])  # Limit to 5 distribution charts
            
            # Correlation analysis
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                corr_chart = self.create_correlation_matrix(df)
                if corr_chart:
                    visualizations.append(corr_chart)
            
            # Target variable analysis
            if target_column and target_column in df.columns:
                target_charts = self._create_target_analysis_charts(df, target_column)
                visualizations.extend(target_charts)
            
            # Missing values analysis
            if df.isnull().sum().sum() > 0:
                missing_chart = self.create_missing_values_heatmap(df)
                if missing_chart:
                    visualizations.append(missing_chart)
            
            # Outlier analysis
            outlier_charts = self._create_outlier_charts(df)
            visualizations.extend(outlier_charts[:3])  # Limit to 3 outlier charts
            
            # Advanced analysis
            advanced_charts = self._create_advanced_charts(df)
            visualizations.extend(advanced_charts[:2])  # Limit to 2 advanced charts
            
            # Limit total number of charts
            return visualizations[:max_charts]
            
        except Exception as e:
            logger.error(f"Comprehensive visualization creation failed: {str(e)}")
            return []
    
    def create_histogram(self, df: pd.DataFrame, column: str, 
                        title: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Create histogram for numeric column"""
        try:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                return None
            
            title = title or f'Distribution of {column}'
            
            fig = px.histogram(
                df, x=column, 
                title=title,
                nbins=kwargs.get('bins', 30),
                marginal=kwargs.get('marginal', 'box'),
                color_discrete_sequence=self.color_schemes['professional']
            )
            
            # Add statistics
            mean_val = df[column].mean()
            median_val = df[column].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                         annotation_text=f"Median: {median_val:.2f}")
            
            self._apply_professional_template(fig)
            
            return {
                'type': 'histogram',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_histogram_insights(df[column])
            }
            
        except Exception as e:
            logger.error(f"Histogram creation failed: {str(e)}")
            return None
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                           color_col: Optional[str] = None, size_col: Optional[str] = None,
                           title: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Create scatter plot"""
        try:
            if x_col not in df.columns or y_col not in df.columns:
                return None
            
            title = title or f'{y_col} vs {x_col}'
            
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col,
                size=size_col,
                title=title,
                trendline=kwargs.get('trendline', 'ols'),
                color_discrete_sequence=self.color_schemes['professional']
            )
            
            self._apply_professional_template(fig)
            
            # Calculate correlation if both are numeric
            correlation = None
            if (pd.api.types.is_numeric_dtype(df[x_col]) and 
                pd.api.types.is_numeric_dtype(df[y_col])):
                correlation = df[x_col].corr(df[y_col])
                fig.add_annotation(
                    text=f"Correlation: {correlation:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    bgcolor="white", bordercolor="black"
                )
            
            return {
                'type': 'scatter',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_scatter_insights(df, x_col, y_col, correlation)
            }
            
        except Exception as e:
            logger.error(f"Scatter plot creation failed: {str(e)}")
            return None
    
    def create_box_plot(self, df: pd.DataFrame, column: str, 
                       group_by: Optional[str] = None,
                       title: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Create box plot for outlier detection"""
        try:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                return None
            
            title = title or f'Box Plot of {column}'
            
            if group_by and group_by in df.columns:
                fig = px.box(df, x=group_by, y=column, title=title,
                           color_discrete_sequence=self.color_schemes['professional'])
            else:
                fig = px.box(df, y=column, title=title,
                           color_discrete_sequence=self.color_schemes['professional'])
            
            self._apply_professional_template(fig)
            
            return {
                'type': 'box',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_box_plot_insights(df[column])
            }
            
        except Exception as e:
            logger.error(f"Box plot creation failed: {str(e)}")
            return None
    
    def create_heatmap(self, data: Union[pd.DataFrame, np.ndarray], 
                      title: Optional[str] = None,
                      labels: Optional[Dict[str, str]] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Create heatmap visualization"""
        try:
            title = title or 'Heatmap'
            
            if isinstance(data, pd.DataFrame):
                fig = px.imshow(
                    data.values if hasattr(data, 'values') else data,
                    labels=labels or {'x': 'Features', 'y': 'Features', 'color': 'Value'},
                    x=data.columns.tolist() if hasattr(data, 'columns') else None,
                    y=data.index.tolist() if hasattr(data, 'index') else None,
                    title=title,
                    color_continuous_scale=kwargs.get('color_scale', 'RdBu_r'),
                    text_auto=kwargs.get('text_auto', True)
                )
            else:
                fig = px.imshow(
                    data,
                    title=title,
                    color_continuous_scale=kwargs.get('color_scale', 'RdBu_r'),
                    text_auto=kwargs.get('text_auto', True)
                )
            
            self._apply_professional_template(fig)
            
            return {
                'type': 'heatmap',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_heatmap_insights(data)
            }
            
        except Exception as e:
            logger.error(f"Heatmap creation failed: {str(e)}")
            return None
    
    def create_correlation_matrix(self, df: pd.DataFrame, 
                                 title: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Create correlation matrix heatmap"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return None
            
            correlation_matrix = numeric_df.corr()
            title = title or 'Feature Correlation Matrix'
            
            # Mask for upper triangle (optional)
            if kwargs.get('mask_upper', False):
                mask = np.triu(np.ones_like(correlation_matrix))
                correlation_matrix = correlation_matrix.mask(mask.astype(bool))
            
            fig = px.imshow(
                correlation_matrix,
                title=title,
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto=True,
                zmin=-1, zmax=1
            )
            
            self._apply_professional_template(fig)
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            return {
                'type': 'correlation',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_correlation_insights(correlation_matrix, high_corr_pairs)
            }
            
        except Exception as e:
            logger.error(f"Correlation matrix creation failed: {str(e)}")
            return None
    
    def create_feature_importance_plot(self, importance_dict: Dict[str, float],
                                     title: Optional[str] = None, top_n: int = 15) -> Optional[Dict[str, Any]]:
        """Create feature importance plot"""
        try:
            if not importance_dict:
                return None
            
            # Sort and limit features
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            title = title or f'Top {len(features)} Feature Importances'
            
            fig = px.bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                title=title,
                color=list(importances),
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='Importance',
                yaxis_title='Features'
            )
            
            self._apply_professional_template(fig)
            
            return {
                'type': 'feature_importance',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_feature_importance_insights(importance_dict)
            }
            
        except Exception as e:
            logger.error(f"Feature importance plot creation failed: {str(e)}")
            return None
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create confusion matrix heatmap"""
        try:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            title = title or 'Confusion Matrix'
            
            if labels is None:
                labels = [f'Class {i}' for i in range(len(cm))]
            
            fig = px.imshow(
                cm,
                title=title,
                labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
                x=labels,
                y=labels,
                color_continuous_scale='Blues',
                text_auto=True
            )
            
            self._apply_professional_template(fig)
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            
            return {
                'type': 'confusion_matrix',
                'title': title,
                'figure': fig.to_json(),
                'insights': [f"Overall accuracy: {accuracy:.3f}"]
            }
            
        except Exception as e:
            logger.error(f"Confusion matrix creation failed: {str(e)}")
            return None
    
    def create_missing_values_heatmap(self, df: pd.DataFrame,
                                    title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create heatmap showing missing value patterns"""
        try:
            if df.isnull().sum().sum() == 0:
                return None
            
            title = title or 'Missing Values Pattern'
            
            # Create missing values matrix
            missing_matrix = df.isnull().astype(int)
            
            fig = px.imshow(
                missing_matrix.T,  # Transpose for better visualization
                title=title,
                labels={'x': 'Records', 'y': 'Features', 'color': 'Missing'},
                color_continuous_scale=['white', 'red'],
                aspect='auto'
            )
            
            self._apply_professional_template(fig)
            
            # Calculate missing percentages
            missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
            missing_summary = missing_pct[missing_pct > 0].to_dict()
            
            return {
                'type': 'missing_values',
                'title': title,
                'figure': fig.to_json(),
                'insights': self._generate_missing_values_insights(missing_summary)
            }
            
        except Exception as e:
            logger.error(f"Missing values heatmap creation failed: {str(e)}")
            return None
    
    def create_pca_plot(self, df: pd.DataFrame, n_components: int = 2,
                       color_col: Optional[str] = None,
                       title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create PCA visualization"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2 or len(numeric_df) < 10:
                return None
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(numeric_df.fillna(0))
            
            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            
            if color_col and color_col in df.columns:
                pca_df[color_col] = df[color_col].values
            
            title = title or f'PCA Analysis ({n_components}D)'
            
            if n_components == 2:
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2',
                    color=color_col if color_col else None,
                    title=title,
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                           'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
                )
            else:
                fig = px.scatter_3d(
                    pca_df, x='PC1', y='PC2', z='PC3',
                    color=color_col if color_col else None,
                    title=title
                )
            
            self._apply_professional_template(fig)
            
            # Add explained variance info
            explained_var = pca.explained_variance_ratio_.sum()
            
            return {
                'type': 'pca',
                'title': title,
                'figure': fig.to_json(),
                'insights': [
                    f"First {n_components} components explain {explained_var:.1%} of variance",
                    f"Component 1 explains {pca.explained_variance_ratio_[0]:.1%} of variance"
                ]
            }
            
        except Exception as e:
            logger.error(f"PCA plot creation failed: {str(e)}")
            return None
    
    def _create_overview_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create overview charts for dataset"""
        charts = []
        
        # Data types distribution
        dtype_counts = df.dtypes.value_counts()
        if len(dtype_counts) > 1:
            fig = px.pie(
                values=dtype_counts.values,
                names=[str(dtype) for dtype in dtype_counts.index],
                title='Data Types Distribution'
            )
            self._apply_professional_template(fig)
            
            charts.append({
                'type': 'pie',
                'title': 'Data Types Distribution',
                'figure': fig.to_json(),
                'insights': ["Dataset contains mixed data types"]
            })
        
        # Missing values summary
        if df.isnull().sum().sum() > 0:
            missing_counts = df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0]
            
            if len(missing_counts) > 0:
                fig = px.bar(
                    x=missing_counts.index,
                    y=missing_counts.values,
                    title='Missing Values by Column'
                )
                fig.update_layout(xaxis_tickangle=-45)
                self._apply_professional_template(fig)
                
                charts.append({
                    'type': 'bar',
                    'title': 'Missing Values by Column',
                    'figure': fig.to_json(),
                    'insights': [f"Total missing values: {missing_counts.sum()}"]
                })
        
        return charts
    
    def _create_distribution_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create distribution charts for numeric columns"""
        charts = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns[:5]:  # Limit to first 5 numeric columns
            chart = self.create_histogram(df, col)
            if chart:
                charts.append(chart)
        
        return charts
    
    def _create_target_analysis_charts(self, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        """Create charts for target variable analysis"""
        charts = []
        
        # Target distribution
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
            # Categorical target
            value_counts = df[target_column].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {target_column}'
            )
            self._apply_professional_template(fig)
            
            charts.append({
                'type': 'bar',
                'title': f'Target Variable Distribution ({target_column})',
                'figure': fig.to_json(),
                'insights': self._generate_target_insights(df[target_column])
            })
        else:
            # Numeric target
            chart = self.create_histogram(df, target_column, 
                                        title=f'Distribution of {target_column}')
            if chart:
                charts.append(chart)
        
        return charts
    
    def _create_outlier_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create charts for outlier detection"""
        charts = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
            chart = self.create_box_plot(df, col)
            if chart:
                charts.append(chart)
        
        return charts
    
    def _create_advanced_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create advanced analysis charts"""
        charts = []
        
        # PCA plot if enough numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 3:
            pca_chart = self.create_pca_plot(df)
            if pca_chart:
                charts.append(pca_chart)
        
        return charts
    
    def _apply_professional_template(self, fig):
        """Apply professional styling to plotly figure"""
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=30, t=60, b=60)
        )
        
        # Update axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
    
    # Insight generation methods
    def _generate_histogram_insights(self, series: pd.Series) -> List[str]:
        """Generate insights for histogram"""
        insights = []
        
        skewness = stats.skew(series.dropna())
        if abs(skewness) > 1:
            insights.append(f"Distribution is {'right' if skewness > 0 else 'left'} skewed")
        else:
            insights.append("Distribution is approximately symmetric")
        
        if series.isnull().sum() > 0:
            insights.append(f"{series.isnull().sum()} missing values detected")
        
        return insights
    
    def _generate_scatter_insights(self, df: pd.DataFrame, x_col: str, y_col: str, correlation: float) -> List[str]:
        """Generate insights for scatter plot"""
        insights = []
        
        if correlation is not None:
            if abs(correlation) > 0.7:
                insights.append(f"Strong {'positive' if correlation > 0 else 'negative'} correlation")
            elif abs(correlation) > 0.3:
                insights.append(f"Moderate {'positive' if correlation > 0 else 'negative'} correlation")
            else:
                insights.append("Weak or no correlation")
        
        return insights
    
    def _generate_box_plot_insights(self, series: pd.Series) -> List[str]:
        """Generate insights for box plot"""
        insights = []
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        if len(outliers) > 0:
            insights.append(f"{len(outliers)} outliers detected ({len(outliers)/len(series)*100:.1f}%)")
        else:
            insights.append("No outliers detected")
        
        return insights
    
    def _generate_heatmap_insights(self, data) -> List[str]:
        """Generate insights for heatmap"""
        return ["Heatmap shows data patterns and relationships"]
    
    def _generate_correlation_insights(self, corr_matrix, high_corr_pairs) -> List[str]:
        """Generate insights for correlation matrix"""
        insights = []
        
        if high_corr_pairs:
            insights.append(f"{len(high_corr_pairs)} highly correlated feature pairs found")
            for pair in high_corr_pairs[:3]:  # Top 3 pairs
                insights.append(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            insights.append("No highly correlated features detected")
        
        return insights
    
    def _generate_feature_importance_insights(self, importance_dict) -> List[str]:
        """Generate insights for feature importance"""
        insights = []
        
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_feature = sorted_features[0]
        insights.append(f"Most important feature: {top_feature[0]} ({top_feature[1]:.3f})")
        
        # Check feature concentration
        top_5_importance = sum([imp for _, imp in sorted_features[:5]])
        if top_5_importance > 0.8:
            insights.append("Feature importance is concentrated in top 5 features")
        
        return insights
    
    def _generate_target_insights(self, target_series: pd.Series) -> List[str]:
        """Generate insights for target variable"""
        insights = []
        
        if target_series.dtype == 'object':
            value_counts = target_series.value_counts()
            insights.append(f"Target has {len(value_counts)} unique classes")
            
            # Check for class imbalance
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 5:
                insights.append(f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        else:
            insights.append(f"Target range: {target_series.min():.2f} to {target_series.max():.2f}")
        
        return insights
    
    def _generate_missing_values_insights(self, missing_summary: Dict[str, float]) -> List[str]:
        """Generate insights for missing values"""
        insights = []
        
        total_features_with_missing = len(missing_summary)
        insights.append(f"{total_features_with_missing} features have missing values")
        
        high_missing_features = [col for col, pct in missing_summary.items() if pct > 50]
        if high_missing_features:
            insights.append(f"{len(high_missing_features)} features have >50% missing values")
        
        return insights

    # Additional chart creation methods (placeholders)
    def create_violin_plot(self, df, column, **kwargs): pass
    def create_bar_chart(self, df, x, y, **kwargs): pass
    def create_line_chart(self, df, x, y, **kwargs): pass
    def create_pie_chart(self, df, values, names, **kwargs): pass
    def create_distribution_plot(self, df, column, **kwargs): pass
    def create_pairplot(self, df, **kwargs): pass
    def create_roc_curve(self, y_true, y_score, **kwargs): pass
    def create_learning_curve(self, train_scores, val_scores, **kwargs): pass
    def create_tsne_plot(self, df, **kwargs): pass

