import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any
import json

class ModelResultsComponent:
    """Component for displaying model training results"""
    
    def render(self, model_results: Dict[str, Any]):
        """Render model results dashboard"""
        
        if 'training_results' not in model_results:
            st.error("Invalid model results format")
            return
        
        results = model_results['training_results']
        
        # Model Performance Overview
        st.markdown("## 🏆 Model Performance Overview")
        self.show_performance_overview(results)
        
        # Model Comparison
        st.markdown("## 📊 Model Comparison")
        self.show_model_comparison(results)
        
        # Best Model Details
        st.markdown("## 🥇 Best Model Analysis")
        self.show_best_model_details(results)
        
        # Feature Importance
        st.markdown("## 🎯 Feature Importance Analysis")
        self.show_feature_importance(results)
        
        # Model Insights and Recommendations
        st.markdown("## 💡 Insights & Recommendations")
        self.show_recommendations(results)
    
    def show_performance_overview(self, results: Dict[str, Any]):
        """Display performance overview"""
        
        if 'model_results' not in results:
            return
        
        model_results = results['model_results']
        task_type = results.get('task_type', 'classification')
        
        # Performance metrics summary
        performance_data = []
        primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
        
        for algo_name, result in model_results.items():
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                performance_data.append({
                    'Algorithm': algo_name.replace('_', ' ').title(),
                    'Primary Metric': metrics.get(primary_metric, 0),
                    'All Metrics': metrics
                })
        
        # Sort by primary metric
        performance_data.sort(key=lambda x: x['Primary Metric'], reverse=True)
        
        # Display top performers
        col1, col2, col3 = st.columns(3)
        
        if len(performance_data) >= 1:
            with col1:
                best = performance_data[0]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);">
                    <h3>🥇 Best Model</h3>
                    <h2>{best['Algorithm']}</h2>
                    <p>{primary_metric.replace('_', ' ').title()}: {best['Primary Metric']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if len(performance_data) >= 2:
            with col2:
                second = performance_data[1]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, #C0C0C0 0%, #A9A9A9 100%);">
                    <h3>🥈 Runner-up</h3>
                    <h2>{second['Algorithm']}</h2>
                    <p>{primary_metric.replace('_', ' ').title()}: {second['Primary Metric']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if len(performance_data) >= 3:
            with col3:
                third = performance_data[2]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, #CD7F32 0%, #8B4513 100%);">
                    <h3>🥉 Third Place</h3>
                    <h2>{third['Algorithm']}</h2>
                    <p>{primary_metric.replace('_', ' ').title()}: {third['Primary Metric']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def show_model_comparison(self, results: Dict[str, Any]):
        """Display detailed model comparison"""
        
        if 'model_comparison' not in results:
            return
        
        comparison = results['model_comparison']
        task_type = results.get('task_type', 'classification')
        
        # Create comparison table
        if 'ranking' in comparison:
            ranking = comparison['ranking']
            
            table_data = []
            for rank, model_info in enumerate(ranking, 1):
                metrics = model_info['all_metrics']
                row = {
                    'Rank': rank,
                    'Algorithm': model_info['algorithm'].replace('_', ' ').title()
                }
                
                # Add relevant metrics based on task type
                if task_type == 'classification':
                    row.update({
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}",
                        'F1-Score': f"{metrics.get('f1_score', 0):.4f}"
                    })
                else:
                    row.update({
                        'R² Score': f"{metrics.get('r2_score', 0):.4f}",
                        'RMSE': f"{metrics.get('root_mean_squared_error', 0):.4f}",
                        'MAE': f"{metrics.get('mean_absolute_error', 0):.4f}"
                    })
                
                table_data.append(row)
            
            comparison_df = pd.DataFrame(table_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance comparison chart
            algorithms = [row['Algorithm'] for row in table_data]
            primary_metric = 'Accuracy' if task_type == 'classification' else 'R² Score'
            scores = [float(row[primary_metric]) for row in table_data]
            
            fig = px.bar(x=algorithms, y=scores,
                        title=f"Model Performance Comparison ({primary_metric})",
                        color=scores,
                        color_continuous_scale="viridis")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_best_model_details(self, results: Dict[str, Any]):
        """Show detailed analysis of the best model"""
        
        best_model = results.get('best_model')
        if not best_model or 'model_results' not in results:
            return
        
        model_info = results['model_results'][best_model]
        task_type = results.get('task_type', 'classification')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🎯 {best_model.replace('_', ' ').title()} Details")
            
            # Performance metrics
            metrics = model_info.get('performance_metrics', {})
            
            if task_type == 'classification':
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
                
                # Confusion Matrix
                if 'confusion_matrix' in metrics:
                    st.markdown("#### Confusion Matrix")
                    cm = np.array(metrics['confusion_matrix'])
                    
                    fig = px.imshow(cm, text_auto=True, aspect="auto",
                                   title="Confusion Matrix",
                                   labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                st.metric("RMSE", f"{metrics.get('root_mean_squared_error', 0):.4f}")
                st.metric("MAE", f"{metrics.get('mean_absolute_error', 0):.4f}")
                st.metric("MSE", f"{metrics.get('mean_squared_error', 0):.4f}")
        
        with col2:
            # Hyperparameters
            st.markdown("#### 🔧 Hyperparameters")
            hyperparams = model_info.get('hyperparameters', {})
            
            param_data = []
            for param, value in hyperparams.items():
                param_data.append({
                    'Parameter': param,
                    'Value': str(value)
                })
            
            if param_data:
                param_df = pd.DataFrame(param_data)
                st.dataframe(param_df, use_container_width=True)
    
    def show_feature_importance(self, results: Dict[str, Any]):
        """Show feature importance analysis"""
        
        best_model = results.get('best_model')
        if not best_model or 'model_results' not in results:
            return
        
        model_info = results['model_results'][best_model]
        
        if 'feature_importance' in model_info:
            importance = model_info['feature_importance']
            
            # Create feature importance DataFrame
            importance_data = [
                {'Feature': feature, 'Importance': imp} 
                for feature, imp in importance.items()
            ]
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            # Feature importance plot
            fig = px.bar(importance_df.tail(15), x='Importance', y='Feature',
                        title="Top 15 Most Important Features", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("#### 📊 Feature Importance Rankings")
            importance_table = importance_df.sort_values('Importance', ascending=False)
            importance_table['Rank'] = range(1, len(importance_table) + 1)
            importance_table = importance_table[['Rank', 'Feature', 'Importance']]
            st.dataframe(importance_table, use_container_width=True)
        
        # SHAP values (if available)
        if 'shap_values' in model_info and model_info['shap_values']:
            st.markdown("#### 🎯 SHAP Analysis Available")
            st.info("SHAP (SHapley Additive exPlanations) values calculated for model interpretability.")
    
    def show_recommendations(self, results: Dict[str, Any]):
        """Show recommendations based on results"""
        
        recommendations = results.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>Recommendation {i}:</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
        
        # Additional insights
        best_model = results.get('best_model', '')
        task_type = results.get('task_type', 'classification')
        
        # Performance-based recommendations
        if best_model and 'model_results' in results:
            model_info = results['model_results'][best_model]
            metrics = model_info.get('performance_metrics', {})
            
            primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
            score = metrics.get(primary_metric, 0)
            
            if task_type == 'classification':
                if score < 0.7:
                    st.markdown("""
                    <div class="recommendation-card">
                        <strong>Performance Improvement:</strong> Consider feature engineering, collecting more data, or trying ensemble methods to improve accuracy.
                    </div>
                    """, unsafe_allow_html=True)
                elif score > 0.95:
                    st.markdown("""
                    <div class="recommendation-card">
                        <strong>Overfitting Check:</strong> Very high accuracy detected. Verify with cross-validation and check for data leakage.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                if score < 0.5:
                    st.markdown("""
                    <div class="recommendation-card">
                        <strong>Model Performance:</strong> Low R² score indicates poor fit. Consider polynomial features, feature scaling, or different algorithms.
                    </div>
                    """, unsafe_allow_html=True)
                elif score > 0.95:
                    st.markdown("""
                    <div class="recommendation-card">
                        <strong>Excellent Performance:</strong> High R² score achieved. Validate results and consider deploying this model.
                    </div>
                    """, unsafe_allow_html=True)
