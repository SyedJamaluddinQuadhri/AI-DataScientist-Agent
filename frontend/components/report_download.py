import streamlit as st
from utils.api_client import APIClient
import time

class ReportDownloadComponent:
    """Component for downloading reports in different formats"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def render(self, dataset_id: str):
        """Render report download section"""
        st.markdown("### 📥 Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        # PDF Download
        with col1:
            if st.button("📄 Download PDF Report", use_container_width=True, type="primary"):
                self._download_report(dataset_id, 'pdf')
        
        # HTML Download
        with col2:
            if st.button("🌐 Download HTML Report", use_container_width=True):
                self._download_report(dataset_id, 'html')
        
        # JSON Download
        with col3:
            if st.button("📊 Download JSON Report", use_container_width=True):
                self._download_report(dataset_id, 'json')
    
    def _download_report(self, dataset_id: str, format_type: str):
        """Download report in specified format"""
        try:
            with st.spinner(f"Generating {format_type.upper()} report... This may take a moment."):
                # Call API to generate report
                report_content = self.api_client.generate_report(
                    dataset_id=dataset_id,
                    format_type=format_type
                )
                
                # Determine file extension and mime type
                if format_type == 'pdf':
                    file_extension = 'pdf'
                    mime_type = 'application/pdf'
                elif format_type == 'html':
                    file_extension = 'html'
                    mime_type = 'text/html'
                else:
                    file_extension = 'json'
                    mime_type = 'application/json'
                
                filename = f"ai_ds_report_{dataset_id}_{int(time.time())}.{file_extension}"
                
                # Create download button
                st.download_button(
                    label=f"⬇️ Click to Download {format_type.upper()}",
                    data=report_content,
                    file_name=filename,
                    mime=mime_type,
                    key=f"download_{format_type}_{time.time()}"
                )
                
                st.success(f"✅ {format_type.upper()} report generated successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to generate {format_type.upper()} report: {str(e)}")
            
            if 'wkhtmltopdf' in str(e).lower() and format_type == 'pdf':
                st.warning("""
                **PDF generation requires wkhtmltopdf to be installed.**
                
                To fix this:
                1. Download from: https://wkhtmltopdf.org/downloads.html
                2. Install on your server
                3. Restart the backend server
                
                Alternatively, download HTML or JSON report.
                """)
