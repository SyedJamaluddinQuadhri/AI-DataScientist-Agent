import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union
import base64
import io

class PlottingUtils:
    """Utility functions for enhanced plotting and visualization"""
    
    def __init__(self):
        # Extended color palettes
        self.color_palettes = {
            'categorical': {
                'set1': px.colors.qualitative.Set1,
                'set2': px.colors.qualitative.Set2,
                'set3': px.colors.qualitative.Set3,
                'pastel': px.colors.qualitative.Pastel,
                'dark24': px.colors.qualitative.Dark24,
                'light24': px.colors.qualitative.Light24
            },
            'sequential': {
                'blues': px.colors.sequential.Blues,
                'viridis': px.colors.sequential.Viridis,
                'plasma': px.colors.sequential.Plasma,
                'inferno': px.colors.sequential.Inferno,
                'magma': px.colors.sequential.Magma,
                'turbo': px.colors.sequential.Turbo
            },
            'diverging': {
                'rdbu': px.colors.diverging.RdBu,
                'rdylbu': px.colors.diverging.RdYlBu,
                'spectral': px.colors.diverging.Spectral,
                'coolwarm': px.colors.diverging.RdYlGn
            }
        }
        
        # Chart templates
        self.templates = {
            'clean': {
                'layout': {
                    'plot_bgcolor': 'white',
                    'paper_bgcolor': 'white',
                    'font': {'family': 'Arial, sans-serif'},
                    'xaxis': {'showgrid': True, 'gridcolor': '#E5E5E5'},
                    'yaxis': {'showgrid': True, 'gridcolor': '#E5E5E5'}
                }
            },
            'dark': {
                'layout': {
                    'plot_bgcolor': '#2F2F2F',
                    'paper_bgcolor': '#1E1E1E',
                    'font': {'family': 'Arial, sans-serif', 'color': 'white'},
                    'xaxis': {'showgrid': True, 'gridcolor': '#404040'},
                    'yaxis': {'showgrid': True, 'gridcolor': '#404040'}
                }
            },
            'minimal': {
                'layout': {
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'font': {'family': 'Arial, sans-serif'},
                    'xaxis': {'showgrid': False, 'zeroline': False},
                    'yaxis': {'showgrid': False, 'zeroline': False}
                }
            }
        }
    
    def create_subplot_figure(self, rows: int, cols: int, 
                             subplot_titles: Optional[List[str]] = None,
                             specs: Optional[List[List[Dict]]] = None,
                             **kwargs) -> go.Figure:
        """Create a subplot figure with enhanced configuration"""
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=specs,
            **kwargs
        )
        
        # Apply default styling
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def add_statistical_annotations(self, fig: go.Figure, data: pd.Series, 
                                   x_pos: float = 0.02, y_pos: float = 0.98) -> go.Figure:
        """Add statistical annotations to a figure"""
        stats_text = f"""
        Count: {len(data):,}<br>
        Mean: {data.mean():.2f}<br>
        Std: {data.std():.2f}<br>
        Min: {data.min():.2f}<br>
        Max: {data.max():.2f}
        """
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=x_pos, y=y_pos,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
    
    def create_correlation_network(self, corr_matrix: pd.DataFrame, 
                                  threshold: float = 0.5) -> go.Figure:
        """Create network graph showing correlations above threshold"""
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add edges for correlations above threshold
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= threshold:
                        G.add_edge(col1, col2, weight=abs(corr_val), correlation=corr_val)
        
        if len(G.edges()) == 0:
            return None
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_colors.append(edge[2]['correlation'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Correlation Network (|r| ≥ {threshold})',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Nodes: Features, Edges: Correlations",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_parallel_coordinates(self, df: pd.DataFrame, 
                                   columns: List[str],
                                   color_column: Optional[str] = None,
                                   title: str = "Parallel Coordinates") -> go.Figure:
        """Create parallel coordinates plot"""
        
        # Prepare dimensions
        dimensions = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dimensions.append(dict(
                    label=col,
                    values=df[col]
                ))
            else:
                # Convert categorical to numeric
                unique_vals = df[col].unique()
                val_map = {val: i for i, val in enumerate(unique_vals)}
                dimensions.append(dict(
                    label=col,
                    values=[val_map[val] for val in df[col]],
                    tickvals=list(range(len(unique_vals))),
                    ticktext=list(unique_vals)
                ))
        
        # Create figure
        if color_column and color_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[color_column]):
                color = df[color_column]
            else:
                # Convert categorical to numeric for coloring
                unique_vals = df[color_column].unique()
                val_map = {val: i for i, val in enumerate(unique_vals)}
                color = [val_map[val] for val in df[color_column]]
        else:
            color = None
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=color, colorscale='Viridis') if color is not None else dict(color='blue'),
            dimensions=dimensions
        ))
        
        fig.update_layout(title=title)
        return fig
    
    def create_sunburst_chart(self, df: pd.DataFrame, 
                             hierarchy_columns: List[str],
                             value_column: Optional[str] = None,
                             title: str = "Sunburst Chart") -> go.Figure:
        """Create sunburst chart for hierarchical data"""
        
        # Prepare data for sunburst
        if value_column and value_column in df.columns:
            values = df[value_column]
        else:
            # Count occurrences
            grouped = df.groupby(hierarchy_columns).size().reset_index(name='count')
            values = grouped['count']
            df = grouped
        
        # Create hierarchical labels and parents
        labels = []
        parents = []
        chart_values = []
        
        # Add root
        labels.append("Total")
        parents.append("")
        chart_values.append(values.sum())
        
        # Process each level of hierarchy
        for level in range(len(hierarchy_columns)):
            if level == 0:
                # First level - parent is root
                unique_vals = df[hierarchy_columns[0]].unique()
                for val in unique_vals:
                    labels.append(str(val))
                    parents.append("Total")
                    val_sum = df[df[hierarchy_columns[0]] == val][value_column if value_column else 'count'].sum()
                    chart_values.append(val_sum)
            else:
                # Subsequent levels
                for i in range(level):
                    parent_cols = hierarchy_columns[:i+1]
                    child_col = hierarchy_columns[i+1]
                    
                    grouped = df.groupby(parent_cols + [child_col])[value_column if value_column else 'count'].sum()
                    
                    for idx, val in grouped.items():
                        parent_label = " - ".join([str(x) for x in idx[:-1]])
                        child_label = " - ".join([str(x) for x in idx])
                        
                        if child_label not in labels:
                            labels.append(child_label)
                            parents.append(parent_label)
                            chart_values.append(val)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=chart_values,
            branchvalues="total"
        ))
        
        fig.update_layout(title=title)
        return fig
    
    def create_animated_chart(self, df: pd.DataFrame, 
                             x_col: str, y_col: str, 
                             animation_frame: str,
                             color_col: Optional[str] = None,
                             size_col: Optional[str] = None,
                             chart_type: str = 'scatter') -> go.Figure:
        """Create animated chart"""
        
        if chart_type == 'scatter':
            fig = px.scatter(
                df, x=x_col, y=y_col,
                animation_frame=animation_frame,
                color=color_col,
                size=size_col,
                hover_name=color_col,
                range_x=[df[x_col].min(), df[x_col].max()],
                range_y=[df[y_col].min(), df[y_col].max()]
            )
        elif chart_type == 'bar':
            fig = px.bar(
                df, x=x_col, y=y_col,
                animation_frame=animation_frame,
                color=color_col
            )
        elif chart_type == 'line':
            fig = px.line(
                df, x=x_col, y=y_col,
                animation_frame=animation_frame,
                color=color_col
            )
        else:
            raise ValueError(f"Unsupported animated chart type: {chart_type}")
        
        # Enhance animation
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
        
        return fig
    
    def add_trendline_equations(self, fig: go.Figure, df: pd.DataFrame, 
                               x_col: str, y_col: str) -> go.Figure:
        """Add trendline equation annotations to scatter plot"""
        from sklearn.linear_model import LinearRegression
        
        # Fit linear regression
        X = df[x_col].dropna().values.reshape(-1, 1)
        y = df[y_col].dropna().values
        
        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, y)
            
            equation_text = f"y = {slope:.3f}x + {intercept:.3f}<br>R² = {r_squared:.3f}"
            
            fig.add_annotation(
                text=equation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        return fig
    
    def create_metric_gauge(self, value: float, 
                           title: str,
                           min_val: float = 0,
                           max_val: float = 100,
                           threshold_good: float = 70,
                           threshold_warning: float = 40) -> go.Figure:
        """Create gauge chart for metrics"""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': threshold_good},
            gauge = {
                'axis': {'range': [None, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, threshold_warning], 'color': "lightgray"},
                    {'range': [threshold_warning, threshold_good], 'color': "yellow"},
                    {'range': [threshold_good, max_val], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold_good
                }
            }
        ))
        
        return fig
    
    def create_waterfall_chart(self, categories: List[str], 
                              values: List[float],
                              title: str = "Waterfall Chart") -> go.Figure:
        """Create waterfall chart"""
        
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * (len(values) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"+{v}" if v > 0 else str(v) for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(title=title, showlegend=False)
        return fig
    
    def apply_custom_theme(self, fig: go.Figure, theme_name: str) -> go.Figure:
        """Apply custom theme to figure"""
        if theme_name in self.templates:
            fig.update_layout(**self.templates[theme_name]['layout'])
        
        return fig
    
    def save_plot_as_image(self, fig: go.Figure, 
                          filename: str,
                          format: str = 'png',
                          width: int = 800,
                          height: int = 600,
                          scale: float = 2) -> str:
        """Save plotly figure as image and return base64 string"""
        
        img_bytes = fig.to_image(
            format=format,
            width=width,
            height=height,
            scale=scale
        )
        
        # Convert to base64 for embedding
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return img_base64
    
    def create_comparison_chart(self, data_dict: Dict[str, pd.Series],
                               chart_type: str = 'box',
                               title: str = "Comparison Chart") -> go.Figure:
        """Create comparison chart for multiple series"""
        
        if chart_type == 'box':
            fig = go.Figure()
            for name, series in data_dict.items():
                fig.add_trace(go.Box(
                    y=series.values,
                    name=name,
                    boxmean=True
                ))
        
        elif chart_type == 'violin':
            fig = go.Figure()
            for name, series in data_dict.items():
                fig.add_trace(go.Violin(
                    y=series.values,
                    name=name,
                    box_visible=True,
                    meanline_visible=True
                ))
        
        elif chart_type == 'histogram':
            fig = go.Figure()
            for name, series in data_dict.items():
                fig.add_trace(go.Histogram(
                    x=series.values,
                    name=name,
                    opacity=0.7
                ))
            fig.update_layout(barmode='overlay')
        
        else:
            raise ValueError(f"Unsupported comparison chart type: {chart_type}")
        
        fig.update_layout(title=title)
        return fig
    
    def get_optimal_bin_count(self, data: pd.Series) -> int:
        """Calculate optimal number of bins for histogram using Sturges' rule"""
        n = len(data.dropna())
        if n > 0:
            return max(int(np.ceil(np.log2(n) + 1)), 10)
        return 30
    
    def detect_outliers_iqr(self, data: pd.Series) -> Tuple[pd.Series, float, float]:
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return outliers, lower_bound, upper_bound
