import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from pathlib import Path
import base64
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import pdfkit
except ImportError:
    pdfkit = None
import platform
import os
import markdown

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Advanced report generation engine for comprehensive data science reports"""
    
    def __init__(self):
        self.pdfkit_config = self._get_pdfkit_config()
        
        # Professional CSS styling for PDF reports
        self.css_styles = """
        <style>
        @page {
            size: A4;
            margin: 1.5cm;
        }
        * {
            box-sizing: border-box;
        }
        body { 
            font-family: 'Segoe UI', 'Arial', sans-serif; 
            margin: 0;
            padding: 20px;
            background-color: #ffffff;
            color: #2c3e50;
            line-height: 1.6;
            font-size: 11pt;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 40px 30px; 
            border-radius: 12px; 
            text-align: center;
            margin-bottom: 30px;
            page-break-after: avoid;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 36px;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header h2 {
            margin: 0 0 10px 0;
            font-size: 22px;
            font-weight: 400;
            opacity: 0.95;
        }
        .header p {
            margin: 5px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .section { 
            margin: 25px 0; 
            padding: 25px; 
            border-left: 5px solid #667eea; 
            background: #f8f9fa;
            border-radius: 8px;
            page-break-inside: avoid;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .section h2 {
            color: #667eea;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 24px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            font-weight: 600;
        }
        .section h3 {
            color: #764ba2;
            font-size: 18px;
            margin-top: 25px;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .section h4 {
            color: #5a67d8;
            font-size: 15px;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
            justify-content: space-around;
        }
        .metric-box { 
            flex: 1;
            min-width: 140px;
            max-width: 200px;
            padding: 25px 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px; 
            text-align: center;
            page-break-inside: avoid;
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }
        .metric-box h3 {
            margin: 0;
            font-size: 32px;
            font-weight: 700;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        .metric-box p {
            margin: 10px 0 0 0;
            font-size: 13px;
            opacity: 0.95;
            font-weight: 500;
        }
        .insight { 
            background: linear-gradient(to right, #e3f2fd 0%, #f0f7ff 100%);
            padding: 15px 20px; 
            border-radius: 8px; 
            margin: 12px 0; 
            border-left: 5px solid #2196F3;
            page-break-inside: avoid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .insight strong {
            color: #1976D2;
        }
        .recommendation { 
            background: linear-gradient(to right, #fff3cd 0%, #fffaeb 100%);
            padding: 15px 20px; 
            border-radius: 8px; 
            margin: 12px 0; 
            border-left: 5px solid #ffc107;
            page-break-inside: avoid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .warning {
            background: linear-gradient(to right, #ffe0b2 0%, #fff3e0 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 12px 0;
            border-left: 5px solid #ff9800;
            page-break-inside: avoid;
        }
        .success {
            background: linear-gradient(to right, #c8e6c9 0%, #e8f5e9 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 12px 0;
            border-left: 5px solid #4caf50;
            page-break-inside: avoid;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            background: white;
            page-break-inside: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-radius: 6px;
            overflow: hidden;
        }
        th, td { 
            padding: 14px 12px; 
            text-align: left; 
            border-bottom: 1px solid #e0e0e0;
        }
        th { 
            background: linear-gradient(to bottom, #667eea 0%, #5a67d8 100%);
            color: white;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e9ecef;
            transition: background-color 0.2s;
        }
        td {
            font-size: 12px;
        }
        .chart-container {
            margin: 25px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            page-break-inside: avoid;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .chart-title {
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .chart-container img {
            width: 100%;
            height: auto;
            border-radius: 6px;
        }
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        li {
            margin: 8px 0;
            line-height: 1.6;
        }
        .page-break {
            page-break-after: always;
        }
        .highlight {
            background-color: #fff176;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin: 0 5px;
        }
        .badge-success {
            background-color: #4caf50;
            color: white;
        }
        .badge-warning {
            background-color: #ff9800;
            color: white;
        }
        .badge-info {
            background-color: #2196F3;
            color: white;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 25px 20px;
            color: #666;
            border-top: 3px solid #667eea;
            page-break-before: avoid;
            background: #f8f9fa;
            border-radius: 8px;
        }
        footer p {
            margin: 5px 0;
        }
        </style>
        """
    
    def _get_pdfkit_config(self):
        """Get pdfkit configuration based on OS"""
        try:
            import pdfkit
            
            if platform.system() == 'Windows':
                possible_paths = [
                    r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                    r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
                    os.path.join(os.getenv('PROGRAMFILES', 'C:\\Program Files'), 'wkhtmltopdf', 'bin', 'wkhtmltopdf.exe'),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Found wkhtmltopdf at: {path}")
                        return pdfkit.configuration(wkhtmltopdf=path)
                
                logger.warning("wkhtmltopdf not found in common paths.")
                return None
            else:
                return None
        except ImportError:
            logger.warning("pdfkit not available")
            return None
    
    def generate_comprehensive_report(self, dataset_info: Dict[str, Any], 
                                    eda_results: Dict[str, Any], 
                                    model_results: Dict[str, Any],
                                    format_type: str = 'html') -> Dict[str, Any]:
        """Generate comprehensive data science report"""
        try:
            report_data = {
                'metadata': {
                    'report_type': 'comprehensive',
                    'generated_at': datetime.now().isoformat(),
                    'dataset_id': dataset_info.get('dataset_id', 'unknown'),
                    'dataset_name': dataset_info.get('original_filename', 'Unknown Dataset')
                },
                'dataset_info': dataset_info,
                'eda_results': eda_results,
                'model_results': model_results
            }
            
            if format_type == 'html':
                content = self._generate_html_report(report_data)
            elif format_type == 'pdf':
                content = self._generate_pdf_report(report_data)
            elif format_type == 'json':
                content = json.dumps(report_data, indent=2)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
            
            return {
                'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'format': format_type,
                'content': content,
                'metadata': report_data['metadata'],
                'size_bytes': len(content.encode('utf-8')) if isinstance(content, str) else len(content)
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report with all sections"""
        dataset_info = report_data.get('dataset_info', {})
        eda_results = report_data.get('eda_results', {})
        model_results = report_data.get('model_results', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Data Scientist Comprehensive Report</title>
            {self.css_styles}
        </head>
        <body>
            <!-- HEADER -->
            <div class="header">
                <h1>🤖 AI Data Scientist Comprehensive Report</h1>
                <h2>{dataset_info.get('original_filename', 'Dataset Analysis')}</h2>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                <p>Report ID: {dataset_info.get('dataset_id', 'N/A')}</p>
            </div>
            
            <!-- TABLE OF CONTENTS -->
            {self._generate_table_of_contents(eda_results, model_results)}
            
            <!-- EXECUTIVE SUMMARY -->
            <div class="section">
                <h2>📋 Executive Summary</h2>
                {self._generate_executive_summary(dataset_info, eda_results, model_results)}
            </div>
            
            <!-- DATASET OVERVIEW -->
            <div class="section">
                <h2>📊 Dataset Overview</h2>
                {self._generate_dataset_overview(dataset_info, eda_results)}
            </div>
            
            <div class="page-break"></div>
            
            <!-- DATA QUALITY ASSESSMENT -->
            <div class="section">
                <h2>✅ Data Quality Assessment</h2>
                {self._generate_data_quality_section(eda_results)}
            </div>
            
            <!-- EXPLORATORY DATA ANALYSIS -->
            <div class="section">
                <h2>🔍 Exploratory Data Analysis</h2>
                {self._generate_eda_section(eda_results)}
            </div>
            
            <div class="page-break"></div>
            
            <!-- STATISTICAL ANALYSIS -->
            <div class="section">
                <h2>📈 Statistical Analysis</h2>
                {self._generate_statistical_analysis(eda_results)}
            </div>
            
            <!-- DATA VISUALIZATIONS -->
            <div class="section">
                <h2>📊 Data Visualizations</h2>
                {self._generate_visualizations_section(eda_results)}
            </div>
            
            <div class="page-break"></div>
            
            <!-- MODEL RESULTS -->
            {self._generate_model_results_section(model_results)}
            
            <!-- KEY INSIGHTS -->
            <div class="section">
                <h2>💡 Key Insights & Findings</h2>
                {self._format_insights_section(eda_results)}
            </div>
            
            <!-- RECOMMENDATIONS -->
            <div class="section">
                <h2>🎯 Actionable Recommendations</h2>
                {self._format_recommendations_section(eda_results, model_results)}
            </div>
            
            <!-- FOOTER -->
            <footer>
                <p><strong>Generated by AI Data Scientist Agent</strong></p>
                <p>Advanced AI-Powered Data Science Platform</p>
                <p style="font-size: 10px; margin-top: 10px;">© {datetime.now().year} | Report ID: {dataset_info.get('dataset_id', 'N/A')}</p>
            </footer>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_table_of_contents(self, eda_results, model_results) -> str:
        """Generate table of contents"""
        has_models = model_results and model_results.get('training_results')
        
        html = """
        <div class="section" style="background: white;">
            <h2>📑 Table of Contents</h2>
            <ol style="line-height: 2;">
                <li>Executive Summary</li>
                <li>Dataset Overview</li>
                <li>Data Quality Assessment</li>
                <li>Exploratory Data Analysis</li>
                <li>Statistical Analysis</li>
                <li>Data Visualizations</li>
        """
        
        if has_models:
            html += "<li>Machine Learning Model Results</li>"
        
        html += """
                <li>Key Insights & Findings</li>
                <li>Actionable Recommendations</li>
            </ol>
        </div>
        """
        
        return html
    
    def _generate_executive_summary(self, dataset_info, eda_results, model_results) -> str:
        """Generate comprehensive executive summary"""
        has_models = model_results and model_results.get('training_results')
        best_model = None
        best_score = None
        
        if has_models:
            training_results = model_results.get('training_results', {})
            best_model = training_results.get('best_model')
            if best_model:
                model_data = training_results.get('model_results', {}).get(best_model, {})
                metrics = model_data.get('performance_metrics', {})
                best_score = metrics.get('accuracy') or metrics.get('r2_score')
        
        # Calculate data quality score
        missing_analysis = eda_results.get('missing_value_analysis', {})
        total_missing = missing_analysis.get('total_missing', 0)
        total_cells = dataset_info.get('rows', 1) * dataset_info.get('columns', 1)
        data_quality_score = max(0, 100 - (total_missing / total_cells * 100)) if total_cells > 0 else 0
        
        html = f"""
        <p><strong>This comprehensive report analyzes a dataset containing {dataset_info.get('rows', 0):,} records 
        across {dataset_info.get('columns', 0)} features, providing in-depth statistical analysis, data quality 
        assessment, and machine learning insights.</strong></p>
        
        <div class="metric-container">
            <div class="metric-box">
                <h3>{dataset_info.get('rows', 0):,}</h3>
                <p>Total Records</p>
            </div>
            <div class="metric-box">
                <h3>{dataset_info.get('columns', 0)}</h3>
                <p>Features</p>
            </div>
            <div class="metric-box">
                <h3>{dataset_info.get('file_size', 0) / 1024 / 1024:.2f} MB</h3>
                <p>File Size</p>
            </div>
            <div class="metric-box">
                <h3>{data_quality_score:.1f}%</h3>
                <p>Data Quality</p>
            </div>
        """
        
        if has_models and best_model and best_score:
            html += f"""
            <div class="metric-box">
                <h3>{best_score:.1%}</h3>
                <p>Best Model Score</p>
            </div>
            """
        
        html += "</div>"
        
        # Key highlights
        insights = eda_results.get('data_insights', [])
        if insights:
            html += f"""
            <h3>Key Highlights</h3>
            <div class="success">
                <strong>✓</strong> {len(insights)} key insights discovered from data analysis
            </div>
            """
        
        if has_models:
            html += f"""
            <div class="success">
                <strong>✓</strong> Best performing model: <span class="highlight">{best_model.replace('_', ' ').title()}</span>
            </div>
            """
        
        if total_missing > 0:
            html += f"""
            <div class="warning">
                <strong>⚠</strong> {total_missing:,} missing values detected requiring attention
            </div>
            """
        
        return html
    
    def _generate_dataset_overview(self, dataset_info, eda_results) -> str:
        """Generate detailed dataset overview"""
        dataset_overview = eda_results.get('dataset_overview', {})
        
        html = f"""
        <h3>Basic Information</h3>
        <table>
            <tr><th>Property</th><th>Value</th><th>Details</th></tr>
            <tr>
                <td><strong>Dataset Name</strong></td>
                <td>{dataset_info.get('original_filename', 'Unknown')}</td>
                <td>Original uploaded file</td>
            </tr>
            <tr>
                <td><strong>Total Rows</strong></td>
                <td>{dataset_info.get('rows', 0):,}</td>
                <td>Number of data records</td>
            </tr>
            <tr>
                <td><strong>Total Columns</strong></td>
                <td>{dataset_info.get('columns', 0)}</td>
                <td>Number of features</td>
            </tr>
            <tr>
                <td><strong>File Size</strong></td>
                <td>{dataset_info.get('file_size', 0) / 1024 / 1024:.2f} MB</td>
                <td>Storage space used</td>
            </tr>
            <tr>
                <td><strong>File Format</strong></td>
                <td>{dataset_info.get('file_type', 'Unknown').upper()}</td>
                <td>Data file type</td>
            </tr>
            <tr>
                <td><strong>Memory Usage</strong></td>
                <td>{dataset_overview.get('memory_usage', 'N/A')}</td>
                <td>RAM required for processing</td>
            </tr>
        </table>
        
        <h3>Column Information</h3>
        {self._generate_column_info_table(eda_results)}
        """
        
        return html
    
    def _generate_column_info_table(self, eda_results) -> str:
        """Generate detailed column information table"""
        columns_info = eda_results.get('columns_info', [])
        
        if not columns_info:
            return "<p>No column information available.</p>"
        
        html = """
        <table>
            <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Null Count</th>
                <th>Null %</th>
                <th>Unique Values</th>
            </tr>
        """
        
        for col_info in columns_info[:20]:  # Show first 20 columns
            null_pct = col_info.get('null_percentage', 0)
            null_badge = 'badge-success' if null_pct == 0 else 'badge-warning' if null_pct < 10 else 'badge-error'
            
            html += f"""
            <tr>
                <td><strong>{col_info.get('name', 'Unknown')}</strong></td>
                <td>{col_info.get('dtype', 'Unknown')}</td>
                <td>{col_info.get('null_count', 0)}</td>
                <td><span class="badge {null_badge}">{null_pct:.1f}%</span></td>
                <td>{col_info.get('unique_count', 0):,}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_data_quality_section(self, eda_results) -> str:
        """Generate data quality assessment section"""
        missing_analysis = eda_results.get('missing_value_analysis', {})
        
        html = "<h3>Missing Values Analysis</h3>"
        
        total_missing = missing_analysis.get('total_missing', 0)
        
        if total_missing == 0:
            html += "<div class='success'>✅ <strong>Excellent!</strong> No missing values detected in the dataset.</div>"
        else:
            html += f"""
            <div class='warning'>
                <strong>⚠ Warning:</strong> Found {total_missing:,} missing values in the dataset.
            </div>
            
            <h4>Missing Values by Column</h4>
            <table>
                <tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Severity</th></tr>
            """
            
            missing_by_column = missing_analysis.get('missing_by_column', {})
            missing_percentage = missing_analysis.get('missing_percentage', {})
            
            # Sort by missing percentage
            sorted_missing = sorted(
                [(col, count, missing_percentage.get(col, 0)) 
                 for col, count in missing_by_column.items() if count > 0],
                key=lambda x: x[2],
                reverse=True
            )
            
            for col, count, pct in sorted_missing[:15]:
                severity = "🔴 Critical" if pct > 50 else "🟡 High" if pct > 20 else "🟢 Low"
                html += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td>{count:,}</td>
                    <td>{pct:.1f}%</td>
                    <td>{severity}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Duplicate analysis
        duplicates = eda_results.get('duplicate_rows', 0)
        if duplicates > 0:
            html += f"""
            <h4>Duplicate Records</h4>
            <div class='warning'>
                Found <strong>{duplicates:,}</strong> duplicate rows in the dataset.
                Consider removing duplicates for accurate analysis.
            </div>
            """
        else:
            html += """
            <h4>Duplicate Records</h4>
            <div class='success'>✅ No duplicate rows detected.</div>
            """
        
        return html
    
    def _generate_eda_section(self, eda_results) -> str:
        """Generate comprehensive EDA section"""
        html = ""
        
        # Descriptive Statistics
        html += "<h3>📊 Descriptive Statistics</h3>"
        desc_stats = eda_results.get('descriptive_statistics', {})
        if desc_stats:
            html += self._create_statistics_table(desc_stats)
        else:
            html += "<p>No descriptive statistics available.</p>"
        
        # Categorical Analysis
        html += "<h3>🏷️ Categorical Features Analysis</h3>"
        cat_analysis = eda_results.get('categorical_analysis', {})
        html += self._create_categorical_analysis(cat_analysis)
        
        return html
    
    def _create_statistics_table(self, desc_stats) -> str:
        """Create detailed statistics table"""
        if not desc_stats:
            return "<p>No statistical summary available.</p>"
        
        html = """
        <table>
            <tr>
                <th>Feature</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>25%</th>
                <th>Median</th>
                <th>75%</th>
                <th>Max</th>
            </tr>
        """
        
        for feature, stats in list(desc_stats.items())[:15]:
            if isinstance(stats, dict):
                html += f"""
                <tr>
                    <td><strong>{feature}</strong></td>
                    <td>{stats.get('count', 'N/A')}</td>
                    <td>{self._format_number(stats.get('mean'))}</td>
                    <td>{self._format_number(stats.get('std'))}</td>
                    <td>{self._format_number(stats.get('min'))}</td>
                    <td>{self._format_number(stats.get('25%'))}</td>
                    <td>{self._format_number(stats.get('50%'))}</td>
                    <td>{self._format_number(stats.get('75%'))}</td>
                    <td>{self._format_number(stats.get('max'))}</td>
                </tr>
                """
        
        html += "</table>"
        return html
    
    def _format_number(self, value) -> str:
        """Format number for display"""
        if value is None or (isinstance(value, str) and value == 'N/A'):
            return 'N/A'
        try:
            num = float(value)
            if abs(num) < 0.01 or abs(num) > 10000:
                return f"{num:.2e}"
            else:
                return f"{num:.2f}"
        except:
            return str(value)
    
    def _create_categorical_analysis(self, cat_analysis) -> str:
        """Create categorical features analysis"""
        if not cat_analysis:
            return "<p>No categorical features found in the dataset.</p>"
        
        html = """
        <table>
            <tr>
                <th>Feature</th>
                <th>Unique Values</th>
                <th>Most Common</th>
                <th>Frequency</th>
                <th>Cardinality</th>
            </tr>
        """
        
        for feature, data in list(cat_analysis.items())[:10]:
            if isinstance(data, dict):
                unique_count = data.get('unique_count', 0)
                most_common = data.get('most_common_value', 'N/A')
                frequency = data.get('most_common_frequency', 0)
                cardinality = "High" if unique_count > 50 else "Medium" if unique_count > 10 else "Low"
                
                html += f"""
                <tr>
                    <td><strong>{feature}</strong></td>
                    <td>{unique_count}</td>
                    <td>{most_common}</td>
                    <td>{frequency}</td>
                    <td><span class="badge badge-info">{cardinality}</span></td>
                </tr>
                """
        
        html += "</table>"
        return html

    
    def _generate_statistical_analysis(self, eda_results) -> str:
        """Generate detailed statistical analysis section"""
        html = ""
        
        # Correlation Analysis
        html += "<h3>🔗 Correlation Analysis</h3>"
        corr_analysis = eda_results.get('correlation_analysis', {})
        html += self._create_correlation_section(corr_analysis)
        
        # Distribution Analysis
        html += "<h3>📈 Distribution Analysis</h3>"
        dist_analysis = eda_results.get('distribution_analysis', {})
        html += self._create_distribution_section(dist_analysis)
        
        # Outlier Analysis
        html += "<h3>⚠️ Outlier Detection</h3>"
        outlier_analysis = eda_results.get('outlier_analysis', {})
        html += self._create_outlier_section(outlier_analysis)
        
        return html
    
    def _create_correlation_section(self, corr_analysis) -> str:
        """Create detailed correlation analysis section"""
        if not corr_analysis:
            return "<p>No correlation analysis available.</p>"
        
        high_corr = corr_analysis.get('high_correlations', [])
        
        if high_corr:
            html = """
            <p>The following feature pairs show strong correlation (|correlation| > 0.7):</p>
            <table>
                <tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th><th>Strength</th></tr>
            """
            
            for item in high_corr[:15]:
                if isinstance(item, dict):
                    corr_value = item.get('correlation', 0)
                    strength = "Very Strong" if abs(corr_value) > 0.9 else "Strong"
                    color = "badge-warning" if abs(corr_value) > 0.9 else "badge-info"
                    
                    html += f"""
                    <tr>
                        <td><strong>{item.get('feature1', 'N/A')}</strong></td>
                        <td><strong>{item.get('feature2', 'N/A')}</strong></td>
                        <td>{corr_value:.3f}</td>
                        <td><span class="badge {color}">{strength}</span></td>
                    </tr>
                    """
            
            html += "</table>"
            html += """
            <div class="recommendation">
                <strong>💡 Recommendation:</strong> Consider removing one feature from highly correlated pairs 
                to reduce multicollinearity in machine learning models.
            </div>
            """
        else:
            html = """
            <div class='success'>
                ✅ No highly correlated features detected. Features appear to be independent.
            </div>
            """
        
        return html
    
    def _create_distribution_section(self, dist_analysis) -> str:
        """Create distribution analysis section"""
        if not dist_analysis:
            return "<p>No distribution analysis available.</p>"
        
        html = """
        <p>Statistical properties of numeric features:</p>
        <table>
            <tr>
                <th>Feature</th>
                <th>Skewness</th>
                <th>Kurtosis</th>
                <th>Normality Test</th>
                <th>Distribution Type</th>
            </tr>
        """
        
        for feature, stats in list(dist_analysis.items())[:15]:
            if isinstance(stats, dict):
                skewness = stats.get('skewness', 0)
                kurtosis = stats.get('kurtosis', 0)
                is_normal = stats.get('is_normal', False)
                
                # Determine distribution type
                if abs(skewness) < 0.5:
                    dist_type = "Symmetric"
                elif skewness > 0:
                    dist_type = "Right-skewed"
                else:
                    dist_type = "Left-skewed"
                
                normality_badge = "badge-success" if is_normal else "badge-warning"
                normality_text = "✓ Normal" if is_normal else "✗ Non-normal"
                
                html += f"""
                <tr>
                    <td><strong>{feature}</strong></td>
                    <td>{skewness:.2f}</td>
                    <td>{kurtosis:.2f}</td>
                    <td><span class="badge {normality_badge}">{normality_text}</span></td>
                    <td>{dist_type}</td>
                </tr>
                """
        
        html += "</table>"
        
        html += """
        <div class="insight">
            <strong>ℹ️ Note:</strong> Skewness values between -0.5 and 0.5 indicate symmetric distributions. 
            Kurtosis values near 0 indicate normal-like tails.
        </div>
        """
        
        return html
    
    def _create_outlier_section(self, outlier_analysis) -> str:
        """Create outlier detection section"""
        if not outlier_analysis:
            return "<p>No outlier analysis available.</p>"
        
        html = """
        <table>
            <tr>
                <th>Feature</th>
                <th>Outliers Count</th>
                <th>Outlier %</th>
                <th>Lower Bound</th>
                <th>Upper Bound</th>
                <th>Severity</th>
            </tr>
        """
        
        has_outliers = False
        for feature, data in list(outlier_analysis.items())[:15]:
            if isinstance(data, dict):
                count = data.get('outlier_count', 0)
                pct = data.get('outlier_percentage', 0)
                
                if count > 0:
                    has_outliers = True
                    severity = "🔴 High" if pct > 10 else "🟡 Medium" if pct > 5 else "🟢 Low"
                    
                    html += f"""
                    <tr>
                        <td><strong>{feature}</strong></td>
                        <td>{count:,}</td>
                        <td>{pct:.1f}%</td>
                        <td>{self._format_number(data.get('lower_bound'))}</td>
                        <td>{self._format_number(data.get('upper_bound'))}</td>
                        <td>{severity}</td>
                    </tr>
                    """
        
        html += "</table>"
        
        if not has_outliers:
            html += """
            <div class='success'>
                ✅ No significant outliers detected using IQR method.
            </div>
            """
        else:
            html += """
            <div class="recommendation">
                <strong>💡 Recommendation:</strong> Investigate outliers to determine if they are errors or genuine extreme values. 
                Consider using robust scaling methods for features with many outliers.
            </div>
            """
        
        return html
    
    def _generate_visualizations_section(self, eda_results) -> str:
        """Generate visualizations section with embedded charts"""
        visualizations = eda_results.get('visualizations', [])
        
        if not visualizations:
            return "<p>No visualizations available in this report.</p>"
        
        html = f"""
        <p>This section contains {len(visualizations)} interactive visualizations generated during exploratory data analysis.</p>
        """
        
        chart_titles = [
            "Feature Distribution Analysis",
            "Feature Relationships",
            "Correlation Heatmap",
            "Category Distribution",
            "Feature Value Counts",
            "Distribution Comparison",
            "Hierarchical Analysis",
            "Comparative Analysis",
            "Trend Analysis"
        ]
        
        for idx, viz_json in enumerate(visualizations[:9], 1):
            try:
                # Parse the plotly figure
                fig_dict = json.loads(viz_json)
                fig = go.Figure(fig_dict)
                
                # Convert to PNG image
                img_bytes = fig.to_image(format="png", width=800, height=500)
                img_base64 = base64.b64encode(img_bytes).decode()
                
                title = chart_titles[idx - 1] if idx <= len(chart_titles) else f"Visualization {idx}"
                
                html += f"""
                <div class="chart-container">
                    <div class="chart-title">Chart {idx}: {title}</div>
                    <img src="data:image/png;base64,{img_base64}" alt="{title}">
                </div>
                """
                
                if idx % 3 == 0 and idx < len(visualizations):
                    html += '<div class="page-break"></div>'
                    
            except Exception as e:
                logger.warning(f"Could not embed visualization {idx}: {str(e)}")
                html += f"""
                <div class="warning">
                    <strong>⚠ Chart {idx}:</strong> Visualization could not be rendered in the report.
                </div>
                """
        
        return html
    
    def _generate_model_results_section(self, model_results) -> str:
        """Generate comprehensive model results section"""
        if not model_results or not model_results.get('training_results'):
            return ""
        
        training_results = model_results['training_results']
        
        html = """
        <div class="page-break"></div>
        <div class="section">
            <h2>🤖 Machine Learning Model Results</h2>
        """
        
        # Task Configuration
        task_type = training_results.get('task_type', 'unknown')
        target_column = training_results.get('target_column', 'unknown')
        dataset_info = training_results.get('dataset_info', {})
        
        html += f"""
        <h3>Model Training Configuration</h3>
        <table>
            <tr><th>Configuration</th><th>Value</th><th>Description</th></tr>
            <tr>
                <td><strong>Task Type</strong></td>
                <td><span class="badge badge-info">{task_type.title()}</span></td>
                <td>Type of machine learning problem</td>
            </tr>
            <tr>
                <td><strong>Target Variable</strong></td>
                <td><strong>{target_column}</strong></td>
                <td>Predicted variable</td>
            </tr>
            <tr>
                <td><strong>Training Samples</strong></td>
                <td>{dataset_info.get('train_size', 0):,}</td>
                <td>80% of total data</td>
            </tr>
            <tr>
                <td><strong>Test Samples</strong></td>
                <td>{dataset_info.get('test_size', 0):,}</td>
                <td>20% of total data</td>
            </tr>
            <tr>
                <td><strong>Features Used</strong></td>
                <td>{dataset_info.get('n_features', 0)}</td>
                <td>Number of input features</td>
            </tr>
            <tr>
                <td><strong>Cross-Validation Folds</strong></td>
                <td>{dataset_info.get('cv_folds_used', 5)}</td>
                <td>For hyperparameter tuning</td>
            </tr>
        </table>
        """
        
        # Model Performance Comparison
        html += "<h3>Model Performance Comparison</h3>"
        html += self._create_model_comparison_table(training_results)
        
        # Best Model Details
        best_model = training_results.get('best_model')
        if best_model:
            html += f"""
            <h3>🏆 Best Performing Model: {best_model.replace('_', ' ').title()}</h3>
            """
            html += self._create_best_model_details(training_results, best_model)
        
        # Feature Importance
        html += self._create_feature_importance_section(training_results, best_model)
        
        html += "</div>"
        return html
    
    def _create_model_comparison_table(self, training_results) -> str:
        """Create comprehensive model comparison table"""
        model_results = training_results.get('model_results', {})
        task_type = training_results.get('task_type', 'classification')
        best_model = training_results.get('best_model')
        
        if not model_results:
            return "<p>No model results available.</p>"
        
        if task_type == 'classification':
            html = """
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Status</th>
                </tr>
            """
            
            for model_name, data in model_results.items():
                metrics = data.get('performance_metrics', {})
                is_best = model_name == best_model
                status_badge = "🏆 Best" if is_best else "✓ Trained"
                row_style = 'style="background-color: #e8f5e9;"' if is_best else ''
                
                html += f"""
                <tr {row_style}>
                    <td><strong>{model_name.replace('_', ' ').title()}</strong></td>
                    <td><strong>{metrics.get('accuracy', 0):.4f}</strong></td>
                    <td>{metrics.get('precision', 0):.4f}</td>
                    <td>{metrics.get('recall', 0):.4f}</td>
                    <td>{metrics.get('f1_score', 0):.4f}</td>
                    <td><span class="badge badge-success">{status_badge}</span></td>
                </tr>
                """
        else:  # Regression
            html = """
            <table>
                <tr>
                    <th>Model</th>
                    <th>R² Score</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>MAPE</th>
                    <th>Status</th>
                </tr>
            """
            
            for model_name, data in model_results.items():
                metrics = data.get('performance_metrics', {})
                is_best = model_name == best_model
                status_badge = "🏆 Best" if is_best else "✓ Trained"
                row_style = 'style="background-color: #e8f5e9;"' if is_best else ''
                
                html += f"""
                <tr {row_style}>
                    <td><strong>{model_name.replace('_', ' ').title()}</strong></td>
                    <td><strong>{metrics.get('r2_score', 0):.4f}</strong></td>
                    <td>{metrics.get('mae', 0):.4f}</td>
                    <td>{metrics.get('rmse', 0):.4f}</td>
                    <td>{metrics.get('mape', 0):.2f}%</td>
                    <td><span class="badge badge-success">{status_badge}</span></td>
                </tr>
                """
        
        html += "</table>"
        return html
    
    def _create_best_model_details(self, training_results, best_model) -> str:
        """Create detailed section for best model"""
        model_data = training_results.get('model_results', {}).get(best_model, {})
        metrics = model_data.get('performance_metrics', {})
        
        html = """
        <div class='success'>
            <h4>Performance Metrics</h4>
            <ul>
        """
        
        for metric, value in metrics.items():
            formatted_name = metric.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                html += f"<li><strong>{formatted_name}:</strong> {value:.4f}</li>"
        
        html += """
            </ul>
        </div>
        """
        
        # Hyperparameters
        hyperparams = model_data.get('hyperparameters', {})
        if hyperparams:
            html += """
            <h4>Hyperparameters</h4>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
            """
            
            for param, value in list(hyperparams.items())[:10]:
                html += f"<tr><td>{param}</td><td>{value}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _create_feature_importance_section(self, training_results, best_model) -> str:
        """Create feature importance analysis section"""
        if not best_model:
            return ""
        
        model_data = training_results.get('model_results', {}).get(best_model, {})
        feature_importance = model_data.get('feature_importance', {})
        
        if not feature_importance:
            return ""
        
        html = """
        <h3>📊 Feature Importance Analysis</h3>
        <p>The following features have the most significant impact on model predictions:</p>
        <table>
            <tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Impact</th></tr>
        """
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for rank, (feature, importance) in enumerate(sorted_features[:15], 1):
            impact = "Very High" if abs(importance) > 0.2 else "High" if abs(importance) > 0.1 else "Moderate" if abs(importance) > 0.05 else "Low"
            badge_color = "badge-success" if abs(importance) > 0.2 else "badge-info" if abs(importance) > 0.1 else "badge-warning"
            
            html += f"""
            <tr>
                <td><strong>#{rank}</strong></td>
                <td><strong>{feature}</strong></td>
                <td>{importance:.4f}</td>
                <td><span class="badge {badge_color}">{impact}</span></td>
            </tr>
            """
        
        html += "</table>"
        
        html += """
        <div class="insight">
            <strong>💡 Insight:</strong> Focus on the top-ranked features for the most significant model improvements.
            Features with higher importance scores have more influence on predictions.
        </div>
        """
        
        return html
    
    def _format_insights_section(self, eda_results: Dict) -> str:
        """Format comprehensive insights section"""
        insights = eda_results.get('data_insights', [])
        
        if not insights:
            return "<p>No automated insights were generated for this dataset.</p>"
        
        html = f"<p>Analysis has identified <strong>{len(insights)}</strong> key insights:</p>"
        
        for i, insight in enumerate(insights[:20], 1):
            html += f"""
            <div class="insight">
                <strong>💡 Insight {i}:</strong> {insight}
            </div>
            """
        
        return html
    
    def _format_recommendations_section(self, eda_results: Dict, model_results: Dict) -> str:
        """Format comprehensive recommendations section"""
        recommendations = eda_results.get('recommendations', [])
        
        # Add model-specific recommendations
        if model_results and model_results.get('training_results'):
            model_recs = model_results['training_results'].get('recommendations', [])
            if model_recs:
                recommendations.extend(model_recs)
        
        if not recommendations:
            return "<p>No specific recommendations available for this dataset.</p>"
        
        html = f"<p>Based on the analysis, here are <strong>{len(recommendations)}</strong> actionable recommendations:</p><ol>"
        
        for rec in recommendations[:20]:
            html += f'<li class="recommendation">{rec}</li>'
        
        html += "</ol>"
        
        html += """
        <div class="success">
            <strong>✓ Next Steps:</strong> Implement these recommendations to improve data quality and model performance.
        </div>
        """
        
        return html
    
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> bytes:
        """Generate PDF from HTML with enhanced options"""
        try:
            html_content = self._generate_html_report(report_data)
            
            try:
                import pdfkit
            except ImportError:
                logger.warning("pdfkit not available, returning HTML as bytes")
                return html_content.encode('utf-8')
            
            options = {
                'page-size': 'A4',
                'margin-top': '15mm',
                'margin-right': '15mm',
                'margin-bottom': '15mm',
                'margin-left': '15mm',
                'encoding': 'UTF-8',
                'no-outline': None,
                'enable-local-file-access': None,
                'print-media-type': None,
                'dpi': 300,
                'image-quality': 100,
                'quiet': ''
            }
            
            try:
                if self.pdfkit_config:
                    pdf_content = pdfkit.from_string(
                        html_content,
                        False,
                        options=options,
                        configuration=self.pdfkit_config
                    )
                else:
                    pdf_content = pdfkit.from_string(html_content, False, options=options)
                
                logger.info(f"PDF generated successfully: {len(pdf_content)} bytes")
                return pdf_content
                
            except OSError as e:
                if 'wkhtmltopdf' in str(e).lower():
                    logger.error("wkhtmltopdf not installed")
                    raise Exception(
                        "PDF generation requires wkhtmltopdf. "
                        "Download from https://wkhtmltopdf.org/downloads.html"
                    )
                raise
                
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            raise Exception(f"PDF generation failed: {str(e)}")
