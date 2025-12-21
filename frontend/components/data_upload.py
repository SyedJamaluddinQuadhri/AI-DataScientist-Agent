import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from utils.api_client import APIClient

class DataUploadComponent:
    """Component for handling data upload and preview"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        
    def render(self) -> Optional[Dict[str, Any]]:
        """Render the data upload component"""
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'arff', 'tsv'],
            help="Supported formats: CSV, Excel, JSON, Parquet, ARFF, TSV"
        )
        
        if uploaded_file is not None:
            # Show file details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)
            
            # Upload and process
            if st.button("📤 Upload and Process Dataset", type="primary"):
                with st.spinner("Uploading and analyzing dataset..."):
                    try:
                        # Upload file
                        result = self.api_client.upload_dataset(uploaded_file)
                        
                        # Show upload success
                        st.success("✅ Dataset uploaded successfully!")
                        
                        # Display basic info
                        self.show_dataset_overview(result)
                        
                        # Display preview
                        self.show_dataset_preview(result)
                        
                        # Display quality report
                        self.show_quality_report(result)
                        
                        return result
                        
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
                        return None
        
        return None
    
    def show_dataset_overview(self, dataset_info: Dict[str, Any]):
        """Show dataset overview"""
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{dataset_info['rows']:,}</h3>
                <p>Rows</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{dataset_info['columns']}</h3>
                <p>Columns</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{dataset_info['file_size']/1024/1024:.1f}MB</h3>
                <p>File Size</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{dataset_info['file_type'].upper()}</h3>
                <p>Format</p>
            </div>
            """, unsafe_allow_html=True)
    
    def show_dataset_preview(self, dataset_info: Dict[str, Any]):
        """Show dataset preview"""
        st.markdown("### 👀 Data Preview")
        
        if 'preview' in dataset_info:
            df_preview = pd.DataFrame(dataset_info['preview'])
            st.dataframe(df_preview, use_container_width=True)
        
        # Column information
        if 'column_info' in dataset_info:
            st.markdown("### 📊 Column Information")
            
            col_data = []
            for col_info in dataset_info['column_info']:
                col_data.append({
                    'Column': col_info['name'],
                    'Data Type': col_info['dtype'],
                    'Sample Values': ', '.join([str(v) for v in col_info['sample_values'][:3]])
                })
            
            col_df = pd.DataFrame(col_data)
            st.dataframe(col_df, use_container_width=True)
    
    def show_quality_report(self, dataset_info: Dict[str, Any]):
        """Show data quality report"""
        if 'quality_report' not in dataset_info:
            return
            
        quality_report = dataset_info['quality_report']
        
        st.markdown("### 🔍 Data Quality Report")
        
        # Missing values summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values")
            missing_pct = (quality_report['missing_values_count'] / 
                          (quality_report['total_rows'] * quality_report['total_columns']) * 100)
            
            if missing_pct < 1:
                st.success(f"Excellent! Only {missing_pct:.1f}% missing values")
            elif missing_pct < 10:
                st.warning(f"Moderate: {missing_pct:.1f}% missing values")
            else:
                st.error(f"High: {missing_pct:.1f}% missing values")
        
        with col2:
            st.markdown("#### Duplicate Records")
            dup_pct = (quality_report['duplicate_rows'] / quality_report['total_rows'] * 100)
            
            if dup_pct == 0:
                st.success("No duplicate records found")
            elif dup_pct < 5:
                st.warning(f"{dup_pct:.1f}% duplicate records")
            else:
                st.error(f"{dup_pct:.1f}% duplicate records")
        
        # Data type distribution
        if 'data_types' in quality_report:
            st.markdown("#### Data Type Distribution")
            
            type_counts = {}
            for dtype in quality_report['data_types'].values():
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            type_df = pd.DataFrame(list(type_counts.items()), 
                                 columns=['Data Type', 'Count'])
            
            import plotly.express as px
            fig = px.pie(type_df, values='Count', names='Data Type', 
                        title="Column Data Types")
            st.plotly_chart(fig, use_container_width=True)
        
        # Issues and recommendations
        issues = []
        recommendations = []
        
        if quality_report.get('constant_columns'):
            issues.append(f"Found {len(quality_report['constant_columns'])} constant columns")
            recommendations.append("Remove constant columns before modeling")
        
        if quality_report.get('high_cardinality_columns'):
            issues.append(f"Found {len(quality_report['high_cardinality_columns'])} high cardinality columns")
            recommendations.append("Consider target encoding for high cardinality categorical features")
        
        if quality_report['missing_values_count'] > 0:
            recommendations.append("Implement missing value imputation strategy")
        
        if quality_report['duplicate_rows'] > 0:
            recommendations.append("Remove or handle duplicate records")
        
        if issues:
            st.markdown("#### ⚠️ Identified Issues")
            for issue in issues:
                st.warning(issue)
        
        if recommendations:
            st.markdown("#### 💡 Recommendations")
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
