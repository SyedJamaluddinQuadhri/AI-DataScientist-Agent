import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import Dict, Any, Optional
import time

from components.data_upload import DataUploadComponent
from components.analysis_dashboard import AnalysisDashboard
from components.visualization_panel import VisualizationPanel
from components.model_results import ModelResultsComponent
from utils.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="AI Data Scientist Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86de;
        border-bottom: 2px solid #2e86de;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #000000;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataScientistAgent:
    """Main application class for the AI Data Scientist Agent"""
    
    def __init__(self):
        self.visualization_panel = VisualizationPanel()
        self.api_client = APIClient()
        self.initialize_session_state()
        
        

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'dataset_id' not in st.session_state:
            st.session_state.dataset_id = None
        if 'dataset_info' not in st.session_state:
            st.session_state.dataset_info = None
        if 'eda_results' not in st.session_state:
            st.session_state.eda_results = None
        if 'model_results' not in st.session_state:
            st.session_state.model_results = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'upload'
        
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🤖 AI Data Scientist Agent</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Advanced AI-powered data science workflow automation. 
                Upload your data and let AI handle the complete analysis pipeline.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        self.create_sidebar()
        
        # Main content based on current step
        if st.session_state.current_step == 'upload':
            self.show_upload_section()
        
        elif st.session_state.current_step == 'analysis':
            self.show_analysis_section()
        elif st.session_state.current_step == 'modeling':
            self.show_modeling_section()
        elif st.session_state.current_step == 'results':
            self.show_results_section()
        
    def create_sidebar(self):
        """Create sidebar navigation"""
        st.sidebar.markdown("## Navigation")
        
        # Progress indicator
        steps = ['Upload', 'Analysis', 'Modeling', 'Results']
        current_index = steps.index(st.session_state.current_step.title())
        
        for i, step in enumerate(steps):
            if i <= current_index:
                st.sidebar.markdown(f"✅ **{step}**")
            else:
                st.sidebar.markdown(f"⏳ {step}")
        
        st.sidebar.markdown("---")
        
        # Dataset info
        if st.session_state.dataset_info:
            st.sidebar.markdown("### Dataset Info")
            info = st.session_state.dataset_info
            st.sidebar.metric("Rows", f"{info['rows']:,}")
            st.sidebar.metric("Columns", info['columns'])
            st.sidebar.metric("File Size", f"{info['file_size']/1024/1024:.1f} MB")
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.markdown("### Quick Actions")
        if st.sidebar.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.session_state.dataset_id:
            if st.sidebar.button("📥 Download Results"):
                self.download_results()
    
    def show_upload_section(self):
        """Show data upload section"""
        st.markdown('<div class="section-header">📁 Data Upload & Preprocessing</div>', 
                   unsafe_allow_html=True)
        
        upload_component = DataUploadComponent(self.api_client)
        result = upload_component.render()
        
        if result:
            st.session_state.dataset_id = result['dataset_id']
            st.session_state.dataset_info = result
            st.session_state.current_step = 'analysis'
            st.success("✅ Dataset uploaded successfully! Proceeding to analysis...")
            time.sleep(1)
            st.rerun()
    
    def show_analysis_section(self):
        """Show analysis section"""
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.dataset_id:
            st.error("No dataset found. Please upload a dataset first.")
            return
        
        dashboard = AnalysisDashboard(self.api_client)
        eda_results = dashboard.render(st.session_state.dataset_id)
        
        if eda_results:
            st.session_state.eda_results = eda_results
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⬅️ Back to Upload", type="secondary"):
                    st.session_state.current_step = 'upload'
                    st.rerun()
            with col2:
                if st.button("➡️ Proceed to Modeling", type="primary"):
                    st.session_state.current_step = 'modeling'
                    st.rerun()
    
    def show_modeling_section(self):
        """Show modeling section"""
        st.markdown('<div class="section-header">🤖 Machine Learning Modeling</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.dataset_id:
            st.error("No dataset found. Please upload a dataset first.")
            return
        
        # Get target column suggestions
        suggestions = self.api_client.get_target_suggestions(st.session_state.dataset_id)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Model Configuration")
            
            # Target column selection
            if st.session_state.dataset_info:
                columns = st.session_state.dataset_info.get('column_info', [])
                column_names = [col['name'] for col in columns]
                
                target_column = st.selectbox(
                    "Select Target Column",
                    column_names,
                    help="Choose the column you want to predict"
                )
                
                # Task type
                task_type = st.selectbox(
                    "Task Type",
                    ["auto", "classification", "regression"],
                    help="Auto-detect will determine the task based on target column"
                )
                
                # Advanced settings
                with st.expander("Advanced Settings"):
                    test_size = st.slider("Test Split Size", 0.1, 0.5, 0.2, 0.05)
                    
                    algorithms = st.multiselect(
                        "Select Algorithms (leave empty for auto-selection)",
                        ["random_forest", "xgboost", "lightgbm", "gradient_boosting", 
                         "logistic_regression", "linear_regression", "svm", "knn"]
                    )
                    
                    hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", True)
                    max_trials = st.slider("Max Optimization Trials", 10, 200, 50, 10)
        
        with col2:
            st.markdown("### Target Column Suggestions")
            if suggestions and 'target_suggestions' in suggestions:
                for suggestion in suggestions['target_suggestions'][:3]:
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>{suggestion['column']}</strong><br>
                        Task: {suggestion['task_type']}<br>
                        Confidence: {suggestion['confidence']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Train models button
        if st.button("🚀 Start Model Training", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    model_results = self.api_client.train_models(
                        st.session_state.dataset_id,
                        target_column=target_column,
                        task_type=task_type,
                        test_size=test_size,
                        algorithms=algorithms if algorithms else None,
                        hyperparameter_tuning=hyperparameter_tuning,
                        max_trials=max_trials
                    )
                    
                    st.session_state.model_results = model_results
                    st.session_state.current_step = 'results'
                    st.success("✅ Model training completed!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")
        
        # Navigation
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Analysis", type="secondary"):
                st.session_state.current_step = 'analysis'
                st.rerun()
    
    def show_results_section(self):
        """Show results and model comparison section"""
        st.markdown('<div class="section-header">📈 Results & Model Performance</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.model_results:
            st.error("No model results found. Please train models first.")
            return
        
        results_component = ModelResultsComponent()
        results_component.render(st.session_state.model_results)
        st.markdown("---")
        from components.report_download import ReportDownloadComponent
        report_component = ReportDownloadComponent(self.api_client)
        report_component.render(st.session_state.dataset_id)
        # Navigation
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Modeling", type="secondary"):
                st.session_state.current_step = 'modeling'
                st.rerun()
        with col2:
            if st.button("🔄 Train New Models", type="primary"):
                st.session_state.current_step = 'modeling'
                st.rerun()
    
    def download_results(self):
        """Download comprehensive results"""
        if not st.session_state.dataset_id:
            st.error("No dataset to download results for.")
            return
        
        # Compile all results
        results = {
            "dataset_info": st.session_state.dataset_info,
            "eda_results": st.session_state.eda_results,
            "model_results": st.session_state.model_results
        }
        
        
        # In your results section or analysis dashboard
        if st.button("📥 Download PDF Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_content = api_client.generate_report(
                        dataset_id=st.session_state.dataset_id,
                        format_type='pdf',
                        include_eda=True,
                        include_modeling=True
                    )
            
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_content,
                        file_name=f"analysis_report_{st.session_state.dataset_id}.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                    st.success("✅ PDF report ready for download!")
            
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    st.info("💡 Make sure wkhtmltopdf is installed on the server")

        # Also add HTML download as alternative
        if st.button("📥 Download HTML Report"):
            with st.spinner("Generating HTML report..."):
                try:
                    html_content = api_client.generate_report(
                        dataset_id=st.session_state.dataset_id,
                        format_type='html',
                        include_eda=True,
                        include_modeling=True
                    )
            
                    st.download_button(
                        label="🌐 Download HTML Report",
                        data=html_content,
                        file_name=f"analysis_report_{st.session_state.dataset_id}.html",
                        mime="text/html",
                        key="download_html"
                    )
                    st.success("✅ HTML report ready for download!")
            
                except Exception as e:
                    st.error(f"HTML generation failed: {str(e)}")
    # In your streamlit_app.py or wherever you show results

    def show_download_reports_section(dataset_id: str):
        """Show report download section"""
        st.markdown("### 📥 Download Analysis Reports")
        st.write("Generate and download comprehensive reports of your analysis.")
        
        col1, col2, col3 = st.columns(3)
        
        # PDF Download
        with col1:
            pdf_clicked = st.button("📄 PDF Report", use_container_width=True, type="primary")
        
        # HTML Download
        with col2:
            html_clicked = st.button("🌐 HTML Report", use_container_width=True)
        
        # JSON Download
        with col3:
            json_clicked = st.button("📊 JSON Data", use_container_width=True)
        
        # Handle PDF download
        if pdf_clicked:
            try:
                with st.spinner("⏳ Generating PDF report... This may take 30-60 seconds."):
                    report_content = st.session_state.api_client.generate_report(
                        dataset_id=dataset_id,
                        format_type='pdf'
                    )
                    
                    # Check if content is valid
                    if len(report_content) == 0:
                        st.error("Generated PDF is empty. Please try again.")
                    else:
                        st.success("✅ PDF report generated successfully!")
                        
                        # Immediate download button
                        st.download_button(
                            label="⬇️ Download PDF Report Now",
                            data=report_content,
                            file_name=f"analysis_report_{dataset_id}.pdf",
                            mime="application/pdf",
                            key="pdf_download_btn"
                        )
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {str(e)}")
                if 'wkhtmltopdf' in str(e).lower():
                    st.info("💡 PDF generation requires wkhtmltopdf. Try HTML report instead.")
        
        # Handle HTML download
        if html_clicked:
            try:
                with st.spinner("⏳ Generating HTML report..."):
                    report_content = st.session_state.api_client.generate_report(
                        dataset_id=dataset_id,
                        format_type='html'
                    )
                    
                    if len(report_content) == 0:
                        st.error("Generated HTML is empty. Please try again.")
                    else:
                        st.success("✅ HTML report generated successfully!")
                        
                        # Immediate download button
                        st.download_button(
                            label="⬇️ Download HTML Report Now",
                            data=report_content,
                            file_name=f"analysis_report_{dataset_id}.html",
                            mime="text/html",
                            key="html_download_btn"
                        )
            except Exception as e:
                st.error(f"❌ Failed to generate HTML: {str(e)}")
        
        # Handle JSON download
        if json_clicked:
            try:
                with st.spinner("⏳ Generating JSON data..."):
                    report_content = st.session_state.api_client.generate_report(
                        dataset_id=dataset_id,
                        format_type='json'
                    )
                    
                    if len(report_content) == 0:
                        st.error("Generated JSON is empty. Please try again.")
                    else:
                        st.success("✅ JSON data generated successfully!")
                        
                        # Immediate download button
                        st.download_button(
                            label="⬇️ Download JSON Data Now",
                            data=report_content,
                            file_name=f"analysis_data_{dataset_id}.json",
                            mime="application/json",
                            key="json_download_btn"
                        )
            except Exception as e:
                st.error(f"❌ Failed to generate JSON: {str(e)}")

    # Add this to your results page
    if st.session_state.get('dataset_id'):
        show_download_reports_section(st.session_state.dataset_id)


# Run the application
if __name__ == "__main__":
    app = DataScientistAgent()
    app.run()
