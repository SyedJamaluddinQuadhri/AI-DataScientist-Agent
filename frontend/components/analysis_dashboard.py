import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, Optional
from utils.api_client import APIClient

class AnalysisDashboard:
    """Dashboard for displaying EDA results"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        
    def render(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Render the analysis dashboard"""
        
        # EDA Configuration
        st.markdown("### 🎯 Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get dataset info for target column selection
            dataset_info = self.api_client.get_dataset_info(dataset_id)
            if dataset_info and 'columns' in dataset_info:
                target_column = st.selectbox(
                    "Target Column (Optional)",
                    ["None"] + dataset_info['columns'],
                    help="Select a target column for focused analysis"
                )
                if target_column == "None":
                    target_column = None
            else:
                target_column = None
        
        with col2:
            custom_prompt = st.text_area(
                "Custom Analysis Prompt (Optional)",
                placeholder="e.g., 'Focus on customer segmentation patterns'",
                help="Provide specific instructions for the analysis"
            )
        
        # Run EDA button
        if st.button("🔍 Run Comprehensive Analysis", type="primary"):
            with st.spinner("Performing comprehensive analysis... This may take a few moments."):
                try:
                    eda_results = self.api_client.perform_eda(
                        dataset_id, 
                        target_column=target_column,
                        custom_prompt=custom_prompt if custom_prompt else None
                    )
                    
                    # Display results
                    self.display_eda_results(eda_results)
                    
                    return eda_results
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    return None
        
        # If we have cached results, show them
        if hasattr(st.session_state, 'eda_results') and st.session_state.eda_results:
            st.markdown("---")
            st.markdown("### 📊 Previous Analysis Results")
            self.display_eda_results(st.session_state.eda_results)
            return st.session_state.eda_results
        
        return None
    
    def display_eda_results(self, eda_results: Dict[str, Any]):
        """Display comprehensive EDA results"""
        
        if 'eda_results' not in eda_results:
            st.error("Invalid EDA results format")
            return
        
        results = eda_results['eda_results']
        
        # Dataset Overview
        st.markdown("## 📋 Dataset Overview")
        self.show_dataset_overview(results.get('dataset_overview', {}))
        
        # Key Insights
        st.markdown("## 💡 Key Insights")
        self.show_insights(results.get('data_insights', []))
        
        # Statistical Summary
        st.markdown("## 📈 Statistical Summary")
        self.show_statistical_summary(results.get('descriptive_statistics', {}))
        
        # Correlation Analysis
        st.markdown("## 🔗 Correlation Analysis")
        self.show_correlation_analysis(results.get('correlation_analysis', {}))
        
        # Missing Values Analysis
        st.markdown("## ❓ Missing Values Analysis")
        self.show_missing_values_analysis(results.get('missing_value_analysis', {}))
        
        # Distribution Analysis
        st.markdown("## 📊 Distribution Analysis")
        self.show_distribution_analysis(results.get('distribution_analysis', {}))
        
        # Outlier Analysis
        st.markdown("## 🎯 Outlier Detection")
        self.show_outlier_analysis(results.get('outlier_analysis', {}))
        
        # Visualizations
        st.markdown("## 📊 Generated Visualizations")
        self.show_visualizations(results.get('visualizations', []))
        
        # Recommendations
        st.markdown("## 🎯 Recommendations")
        self.show_recommendations(results.get('recommendations', []))
    
    def show_dataset_overview(self, overview: Dict[str, Any]):
        """Display dataset overview"""
        if not overview:
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Shape", f"{overview.get('shape', [0, 0])[0]:,} × {overview.get('shape', [0, 0])[1]}")
        
        with col2:
            st.metric("Memory Usage", f"{overview.get('memory_usage_mb', 0):.1f} MB")
        
        with col3:
            st.metric("Numeric Columns", len(overview.get('numeric_columns', [])))
        
        with col4:
            st.metric("Categorical Columns", len(overview.get('categorical_columns', [])))
        
        # Column types breakdown
        col_types = {
            'Numeric': len(overview.get('numeric_columns', [])),
            'Categorical': len(overview.get('categorical_columns', [])),
            'DateTime': len(overview.get('datetime_columns', []))
        }
        
        fig = px.pie(values=list(col_types.values()), names=list(col_types.keys()),
                    title="Column Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_insights(self, insights: list):
        """Display key insights"""
        if not insights:
            st.info("No specific insights generated.")
            return
            
        for i, insight in enumerate(insights, 1):
            st.markdown(f"""
            <div class="insight-card">
                <strong>Insight {i}:</strong> {insight}
            </div>
            """, unsafe_allow_html=True)
    
    def show_statistical_summary(self, stats: Dict[str, Any]):
        """Display statistical summary"""
        if 'numeric_statistics' in stats:
            st.markdown("### Numeric Variables Summary")
            
            # Create summary DataFrame
            numeric_stats = stats['numeric_statistics']
            summary_data = []
            
            for col, col_stats in numeric_stats.items():
                summary_data.append({
                    'Column': col,
                    'Mean': f"{col_stats.get('mean', 0):.2f}",
                    'Std': f"{col_stats.get('std', 0):.2f}",
                    'Min': f"{col_stats.get('min', 0):.2f}",
                    'Max': f"{col_stats.get('max', 0):.2f}",
                    'Skewness': f"{col_stats.get('skewness', 0):.2f}",
                    'Kurtosis': f"{col_stats.get('kurtosis', 0):.2f}"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        if 'categorical_statistics' in stats:
            st.markdown("### Categorical Variables Summary")
            
            cat_stats = stats['categorical_statistics']
            cat_data = []
            
            for col, col_stats in cat_stats.items():
                cat_data.append({
                    'Column': col,
                    'Unique Values': col_stats.get('unique_count', 0),
                    'Most Frequent': str(col_stats.get('most_frequent', '')),
                    'Frequency': col_stats.get('frequency_of_most_frequent', 0),
                    'Cardinality Ratio': f"{col_stats.get('cardinality_ratio', 0):.3f}"
                })
            
            if cat_data:
                cat_df = pd.DataFrame(cat_data)
                st.dataframe(cat_df, use_container_width=True)
    
    def show_correlation_analysis(self, corr_analysis: Dict[str, Any]):
        """Display correlation analysis"""
        if not corr_analysis or 'correlation_matrix' not in corr_analysis:
            st.info("No correlation analysis available (no numeric columns found).")
            return
        
        # Correlation heatmap
        corr_matrix = corr_analysis['correlation_matrix']
        corr_df = pd.DataFrame(corr_matrix)
        
        fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix",
                       color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        
        # Highly correlated pairs
        if 'highly_correlated_pairs' in corr_analysis:
            high_corr = corr_analysis['highly_correlated_pairs']
            if high_corr:
                st.markdown("#### 🔗 Highly Correlated Feature Pairs")
                
                for pair in high_corr:
                    correlation = pair['correlation']
                    color = "success" if abs(correlation) < 0.9 else "warning"
                    st.markdown(f"""
                    <div class="insight-card" style="border-left-color: {'#28a745' if color == 'success' else '#ffc107'};">
                        <strong>{pair['feature1']}</strong> ↔ <strong>{pair['feature2']}</strong><br>
                        Correlation: {correlation:.3f}
                    </div>
                    """, unsafe_allow_html=True)
    
    def show_missing_values_analysis(self, missing_analysis: Dict[str, Any]):
        """Display missing values analysis"""
        if not missing_analysis:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Missing Values", missing_analysis.get('total_missing', 0))
        
        with col2:
            missing_cols = len(missing_analysis.get('columns_with_missing', []))
            st.metric("Columns with Missing Values", missing_cols)
        
        # Missing values by column
        if 'missing_percentages' in missing_analysis:
            missing_pct = missing_analysis['missing_percentages']
            missing_data = [(col, pct) for col, pct in missing_pct.items() if pct > 0]
            
            if missing_data:
                missing_df = pd.DataFrame(missing_data, columns=['Column', 'Missing %'])
                missing_df = missing_df.sort_values('Missing %', ascending=False)
                
                fig = px.bar(missing_df, x='Missing %', y='Column',
                           title="Missing Values by Column", orientation='h')
                st.plotly_chart(fig, use_container_width=True)
    
    def show_distribution_analysis(self, dist_analysis: Dict[str, Any]):
        """Display distribution analysis"""
        if not dist_analysis:
            return
        
        st.markdown("### Distribution Characteristics")
        
        dist_data = []
        for col, analysis in dist_analysis.items():
            dist_data.append({
                'Column': col,
                'Skewness': f"{analysis.get('skewness', 0):.3f}",
                'Kurtosis': f"{analysis.get('kurtosis', 0):.3f}",
                'Normal Distribution': "✅" if analysis.get('is_normal', False) else "❌",
                'Shapiro-Wilk p-value': f"{analysis.get('shapiro_wilk_p_value', 0):.4f}"
            })
        
        if dist_data:
            dist_df = pd.DataFrame(dist_data)
            st.dataframe(dist_df, use_container_width=True)
    
    def show_outlier_analysis(self, outlier_analysis: Dict[str, Any]):
        """Display outlier analysis"""
        if not outlier_analysis:
            return
        
        outlier_data = []
        for col, analysis in outlier_analysis.items():
            outlier_data.append({
                'Column': col,
                'Outlier Count': analysis.get('iqr_outliers_count', 0),
                'Outlier %': f"{analysis.get('iqr_outliers_percentage', 0):.2f}%",
                'Lower Bound': f"{analysis.get('lower_bound', 0):.2f}",
                'Upper Bound': f"{analysis.get('upper_bound', 0):.2f}"
            })
        
        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)
            
            # Outlier visualization
            outlier_counts = [row['Outlier Count'] for row in outlier_data]
            columns = [row['Column'] for row in outlier_data]
            
            fig = px.bar(x=columns, y=outlier_counts,
                        title="Outlier Count by Column")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_visualizations(self, visualizations: list):
        """Display generated visualizations"""
        if not visualizations:
            st.info("No visualizations generated.")
            return
        
        for viz in visualizations:
            try:
                st.markdown(f"#### {viz.get('title', 'Visualization')}")
                
                # Parse and display plotly figure
                if 'figure' in viz:
                    fig_json = json.loads(viz['figure'])
                    fig = go.Figure(fig_json)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")
    
    def show_recommendations(self, recommendations: list):
        """Display recommendations"""
        if not recommendations:
            st.info("No specific recommendations generated.")
            return
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>Recommendation {i}:</strong> {rec}
            </div>
            """, unsafe_allow_html=True)
