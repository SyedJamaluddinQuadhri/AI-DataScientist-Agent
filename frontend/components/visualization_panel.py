import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Optional
from utils.api_client import APIClient
from utils.plotting_utils import PlottingUtils

class VisualizationPanel:
    """Advanced interactive visualization panel leveraging PlottingUtils for all chart generation."""
    
    def __init__(self):
        self.api_client = APIClient()
        self.plotting_utils = PlottingUtils()
        
        
        # Available chart types
        self.chart_types = {
            'scatter': 'Scatter Plot',
            'line': 'Line Chart',
            'bar': 'Bar Chart',
            'histogram': 'Histogram',
            'box': 'Box Plot',
            'violin': 'Violin Plot',
            'heatmap': 'Heatmap',
            'pie': 'Pie Chart',
            'area': 'Area Chart',
            'sunburst': 'Sunburst Chart',
            'treemap': 'Treemap',
            'parallel_coordinates': 'Parallel Coordinates',
            'radar': 'Radar Chart'
        }
        
        # Chart configurations
        self.chart_configs = {
            'color_schemes': ['plotly', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'themes': ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']
        }
    
    def render(self, dataset_id: str, df: Optional[pd.DataFrame] = None) -> None:
        """Render the visualization panel"""
        st.markdown("## 📊 Custom Visualization Creator")
        
        if df is None:
            try:
                # Load dataset info
                dataset_info = self.api_client.get_dataset_info(dataset_id)
                if not dataset_info:
                    st.error("Dataset not found")
                    return
                st.info("Dataset loaded. Create custom visualizations below.")
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                return
        
        # Visualization builder interface
        self._render_chart_builder(dataset_id, df)
        
        # Auto-generated visualizations section
        self._render_auto_visualizations(df)
        
        # Saved visualizations
        self._render_saved_visualizations()
    
    def _render_auto_visualizations(self, df: Optional[pd.DataFrame]):
        """Automatically generate comprehensive visualizations using PlottingUtils"""
        if df is None or df.empty:
            return
        
        st.markdown("---")
        st.markdown("## 📈 Auto-Generated Visualizations")
        st.markdown("*Comprehensive analysis using all available plotting methods*")
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        figures = []
        
        # 1. Histogram for numeric columns
        if num_cols:
            for col in num_cols[:2]:
                try:
                    fig = self.plotting_utils.histogram(df, col)
                    figures.append(('histogram', f"Distribution of {col}", fig))
                except Exception as e:
                    st.warning(f"Could not create histogram for {col}: {str(e)}")
        
        # 2. Scatter plots
        if len(num_cols) >= 2:
            try:
                fig = self.plotting_utils.scatter(df, x=num_cols[0], y=num_cols[1])
                figures.append(('scatter', f"{num_cols[1]} vs {num_cols[0]}", fig))
            except Exception as e:
                st.warning(f"Could not create scatter plot: {str(e)}")
        
        # 3. Correlation heatmap
        if len(num_cols) > 1:
            try:
                fig = self.plotting_utils.correlation_heatmap(df[num_cols])
                figures.append(('heatmap', "Correlation Heatmap", fig))
            except Exception as e:
                st.warning(f"Could not create correlation heatmap: {str(e)}")
        
        # 4. Box plots
        if cat_cols and num_cols:
            try:
                fig = self.plotting_utils.box(df, x=cat_cols[0], y=num_cols[0])
                figures.append(('box', f"{num_cols[0]} by {cat_cols[0]}", fig))
            except Exception as e:
                st.warning(f"Could not create box plot: {str(e)}")
        
        # 5. Violin plots
        if cat_cols and num_cols:
            try:
                fig = self.plotting_utils.violin(df, x=cat_cols[0], y=num_cols[0])
                figures.append(('violin', f"{num_cols[0]} distribution by {cat_cols[0]}", fig))
            except Exception as e:
                st.warning(f"Could not create violin plot: {str(e)}")
        
        # 6. Bar charts for categorical
        if cat_cols:
            try:
                val_counts = df[cat_cols[0]].value_counts().nlargest(10).reset_index()
                val_counts.columns = [cat_cols[0], 'count']
                fig = self.plotting_utils.bar(val_counts, x=cat_cols[0], y='count')
                figures.append(('bar', f"Top categories in {cat_cols[0]}", fig))
            except Exception as e:
                st.warning(f"Could not create bar chart: {str(e)}")
        
        # 7. Pie chart for small categoricals
        if cat_cols and df[cat_cols[0]].nunique() <= 8:
            try:
                pie_data = df[cat_cols[0]].value_counts().reset_index()
                pie_data.columns = [cat_cols[0], 'count']
                fig = self.plotting_utils.pie(pie_data, names=cat_cols[0], values='count')
                figures.append(('pie', f"Distribution of {cat_cols[0]}", fig))
            except Exception as e:
                st.warning(f"Could not create pie chart: {str(e)}")
        
        # 8. Sunburst for hierarchical data
        if len(cat_cols) >= 2:
            try:
                fig = self.plotting_utils.sunburst(df, path=cat_cols[:3])
                figures.append(('sunburst', "Hierarchical Breakdown", fig))
            except Exception as e:
                st.warning(f"Could not create sunburst: {str(e)}")
        
        # 9. Treemap
        if len(cat_cols) >= 2:
            try:
                fig = self.plotting_utils.treemap(df, path=cat_cols[:3])
                figures.append(('treemap', "Treemap Visualization", fig))
            except Exception as e:
                st.warning(f"Could not create treemap: {str(e)}")
        
        # 10. Parallel coordinates
        if len(num_cols) >= 3:
            try:
                color_col = cat_cols[0] if cat_cols else None
                fig = self.plotting_utils.parallel_coordinates(df, dimensions=num_cols[:5], color=color_col)
                figures.append(('parallel', "Parallel Coordinates", fig))
            except Exception as e:
                st.warning(f"Could not create parallel coordinates: {str(e)}")
        
        # Display all figures in a 2-column layout
        if not figures:
            st.info("No auto-visualizations could be generated for this dataset.")
        else:
            cols = st.columns(2)
            for idx, (chart_type, title, fig) in enumerate(figures):
                with cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_chart_builder(self, dataset_id: str, df: Optional[pd.DataFrame]):
        """Render the custom chart builder interface"""
        st.markdown("### 🎨 Custom Chart Builder")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Chart Configuration")
            
            # Chart type selection
            chart_type = st.selectbox(
                "Select Chart Type",
                options=list(self.chart_types.keys()),
                format_func=lambda x: self.chart_types[x],
                key="chart_type_select"
            )
            
            # Get available columns
            available_columns = ['feature1', 'feature2', 'feature3', 'target'] if df is None else df.columns.tolist()
            
            # Dynamic configuration based on chart type
            config = self._get_chart_config_ui(chart_type, available_columns)
            
            # Styling options
            st.markdown("#### Styling Options")
            color_scheme = st.selectbox("Color Scheme", self.chart_configs['color_schemes'])
            theme = st.selectbox("Theme", self.chart_configs['themes'])
            
            # Advanced options
            with st.expander("Advanced Options"):
                show_grid = st.checkbox("Show Grid", True)
                show_legend = st.checkbox("Show Legend", True)
                custom_title = st.text_input("Custom Title", "")
                width = st.slider("Width", 400, 1200, 800)
                height = st.slider("Height", 300, 800, 500)
            
            # Generate chart button
            if st.button("🚀 Generate Chart", type="primary"):
                chart_config = {
                    'chart_type': chart_type,
                    'config': config,
                    'styling': {
                        'color_scheme': color_scheme,
                        'theme': theme,
                        'show_grid': show_grid,
                        'show_legend': show_legend,
                        'title': custom_title,
                        'width': width,
                        'height': height
                    }
                }
                
                with st.spinner("Generating visualization..."):
                    chart = self._create_custom_chart(dataset_id, chart_config, df)
                    if chart:
                        st.session_state['current_chart'] = chart
        
        with col2:
            st.markdown("#### Chart Preview")
            
            # Display current chart
            if 'current_chart' in st.session_state:
                chart = st.session_state['current_chart']
                
                if 'figure' in chart:
                    try:
                        fig_dict = json.loads(chart['figure'])
                        fig = go.Figure(fig_dict)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Chart actions
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button("💾 Save Chart"):
                                self._save_chart(chart)
                        with col_b:
                            if st.button("📥 Download"):
                                self._download_chart(chart)
                        with col_c:
                            if st.button("🔄 Regenerate"):
                                st.rerun()
                        
                        # Chart insights
                        if 'insights' in chart and chart['insights']:
                            st.markdown("#### 💡 Chart Insights")
                            for insight in chart['insights']:
                                st.info(insight)
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h3>👈 Configure your chart</h3>
                        <p>Select chart type and options, then click 'Generate Chart'</p>
                    </div>
                """, unsafe_allow_html=True)
    
    def _get_chart_config_ui(self, chart_type: str, columns: List[str]) -> Dict[str, Any]:
        """Render dynamic UI for chart configuration based on type"""
        config = {}
        
        num_cols = [c for c in columns if c not in ['object', 'category', 'bool']]
        cat_cols = columns  # Simplified
        
        if chart_type == 'scatter':
            config['x_axis'] = st.selectbox("X-Axis", num_cols, key="scatter_x")
            config['y_axis'] = st.selectbox("Y-Axis", num_cols, key="scatter_y")
            config['color'] = st.selectbox("Color By (optional)", [None] + cat_cols, key="scatter_color")
            config['size'] = st.selectbox("Size By (optional)", [None] + num_cols, key="scatter_size")
            config['trendline'] = st.checkbox("Add Trendline", key="scatter_trendline")
            
        elif chart_type == 'line':
            config['x_axis'] = st.selectbox("X-Axis", columns, key="line_x")
            config['y_axis'] = st.selectbox("Y-Axis", num_cols, key="line_y")
            config['color'] = st.selectbox("Color By (optional)", [None] + cat_cols, key="line_color")
            
        elif chart_type == 'bar':
            config['x_axis'] = st.selectbox("X-Axis", columns, key="bar_x")
            config['y_axis'] = st.selectbox("Y-Axis", columns, key="bar_y")
            config['color'] = st.selectbox("Color By (optional)", [None] + cat_cols, key="bar_color")
            config['orientation'] = st.radio("Orientation", ['vertical', 'horizontal'], key="bar_orient")
            
        elif chart_type == 'histogram':
            config['column'] = st.selectbox("Column", num_cols, key="hist_col")
            config['nbins'] = st.slider("Number of Bins", 10, 100, 30, key="hist_bins")
            
        elif chart_type in ['box', 'violin']:
            config['x_axis'] = st.selectbox("Category (X)", cat_cols, key=f"{chart_type}_x")
            config['y_axis'] = st.selectbox("Values (Y)", num_cols, key=f"{chart_type}_y")
            config['color'] = st.selectbox("Color By (optional)", [None] + cat_cols, key=f"{chart_type}_color")
            
        elif chart_type == 'heatmap':
            config['columns'] = st.multiselect("Select Columns", num_cols, default=num_cols[:5], key="heatmap_cols")
            config['type'] = st.radio("Heatmap Type", ['correlation', 'values'], key="heatmap_type")
            
        elif chart_type == 'pie':
            config['names'] = st.selectbox("Names", cat_cols, key="pie_names")
            config['values'] = st.selectbox("Values", num_cols, key="pie_values")
            
        elif chart_type == 'area':
            config['x_axis'] = st.selectbox("X-Axis", columns, key="area_x")
            config['y_axis'] = st.selectbox("Y-Axis", num_cols, key="area_y")
            
        elif chart_type in ['sunburst', 'treemap']:
            config['path'] = st.multiselect("Hierarchical Path", cat_cols, default=cat_cols[:2], key=f"{chart_type}_path")
            config['values'] = st.selectbox("Values (optional)", [None] + num_cols, key=f"{chart_type}_values")
            
        elif chart_type == 'parallel_coordinates':
            config['dimensions'] = st.multiselect("Dimensions", num_cols, default=num_cols[:4], key="parallel_dims")
            config['color'] = st.selectbox("Color By (optional)", [None] + cat_cols, key="parallel_color")
            
        elif chart_type == 'radar':
            config['dimensions'] = st.multiselect("Dimensions", num_cols, default=num_cols[:5], key="radar_dims")
            
        return config
    
    def _create_custom_chart(self, dataset_id: str, chart_config: Dict, df: Optional[pd.DataFrame]) -> Optional[Dict]:
        """Create custom chart using PlottingUtils methods"""
        try:
            if df is None:
                st.error("No data available")
                return None
            
            chart_type = chart_config['chart_type']
            config = chart_config['config']
            styling = chart_config['styling']
            
            # Call appropriate PlottingUtils method based on chart type
            if chart_type == 'scatter':
                fig = self.plotting_utils.scatter(df, x=config['x_axis'], y=config['y_axis'])
                
            elif chart_type == 'line':
                fig = self.plotting_utils.line(df, x=config['x_axis'], y=config['y_axis'])
                
            elif chart_type == 'bar':
                fig = self.plotting_utils.bar(df, x=config['x_axis'], y=config['y_axis'])
                
            elif chart_type == 'histogram':
                fig = self.plotting_utils.histogram(df, config['column'])
                
            elif chart_type == 'box':
                fig = self.plotting_utils.box(df, x=config['x_axis'], y=config['y_axis'])
                
            elif chart_type == 'violin':
                fig = self.plotting_utils.violin(df, x=config['x_axis'], y=config['y_axis'])
                
            elif chart_type == 'heatmap':
                fig = self.plotting_utils.correlation_heatmap(df[config['columns']])
                
            elif chart_type == 'pie':
                fig = self.plotting_utils.pie(df, names=config['names'], values=config['values'])
                
            elif chart_type == 'area':
                fig = self.plotting_utils.area(df, x=config['x_axis'], y=config['y_axis'])
                
            elif chart_type == 'sunburst':
                fig = self.plotting_utils.sunburst(df, path=config['path'])
                
            elif chart_type == 'treemap':
                fig = self.plotting_utils.treemap(df, path=config['path'])
                
            elif chart_type == 'parallel_coordinates':
                fig = self.plotting_utils.parallel_coordinates(df, dimensions=config['dimensions'], color=config.get('color'))
                
            elif chart_type == 'radar':
                fig = self.plotting_utils.radar(df, numeric_cols=config['dimensions'])
                
            else:
                st.error(f"Unsupported chart type: {chart_type}")
                return None
            
            # Apply custom styling
            if styling['title']:
                fig.update_layout(title=styling['title'])
            fig.update_layout(
                template=styling['theme'],
                showlegend=styling['show_legend'],
                width=styling['width'],
                height=styling['height']
            )
            
            return {
                'figure': fig.to_json(),
                'type': chart_type,
                'config': config,
                'insights': self._generate_chart_insights(chart_type, df, config)
            }
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def _generate_chart_insights(self, chart_type: str, df: pd.DataFrame, config: Dict) -> List[str]:
        """Generate insights based on chart type and data"""
        insights = []
        
        if chart_type == 'scatter':
            corr = df[config['x_axis']].corr(df[config['y_axis']])
            insights.append(f"Correlation coefficient: {corr:.3f}")
            
        elif chart_type == 'histogram':
            mean_val = df[config['column']].mean()
            median_val = df[config['column']].median()
            insights.append(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}")
            
        elif chart_type in ['box', 'violin']:
            groups = df.groupby(config['x_axis'])[config['y_axis']].mean()
            insights.append(f"Highest average: {groups.idxmax()} ({groups.max():.2f})")
            
        return insights
    
    def _render_saved_visualizations(self):
        """Render saved visualizations section"""
        st.markdown("---")
        st.markdown("### 💾 Saved Visualizations")
        
        if 'saved_charts' not in st.session_state:
            st.session_state['saved_charts'] = []
        
        if not st.session_state['saved_charts']:
            st.info("No saved visualizations yet. Create and save charts to see them here.")
        else:
            for idx, chart in enumerate(st.session_state['saved_charts']):
                with st.expander(f"Chart {idx + 1}: {chart['type'].title()}"):
                    fig_dict = json.loads(chart['figure'])
                    fig = go.Figure(fig_dict)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _save_chart(self, chart: Dict):
        """Save chart to session state"""
        if 'saved_charts' not in st.session_state:
            st.session_state['saved_charts'] = []
        st.session_state['saved_charts'].append(chart)
        st.success("Chart saved successfully!")
    
    def _download_chart(self, chart: Dict):
        """Download chart as JSON"""
        st.download_button(
            label="Download Chart JSON",
            data=chart['figure'],
            file_name=f"{chart['type']}_chart.json",
            mime="application/json"
        )
