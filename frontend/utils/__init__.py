"""
Frontend utilities package
Contains utility functions and helper classes for the Streamlit frontend
"""

from .api_client import APIClient
from .plotting_utils import PlottingUtils

# Create utility instances for easy access
def get_api_client(base_url: str = "http://localhost:8000") -> APIClient:
    """Get API client instance with specified base URL"""
    return APIClient(base_url)

def get_plotting_utils() -> PlottingUtils:
    """Get plotting utilities instance"""
    return PlottingUtils()

# Default instances
api_client = APIClient()
plotting_utils = PlottingUtils()

__all__ = [
    # Utility classes
    'APIClient',
    'PlottingUtils',
    
    # Factory functions
    'get_api_client',
    'get_plotting_utils',
    
    # Default instances
    'api_client',
    'plotting_utils'
]
