import requests
import streamlit as st
from typing import Dict, Any, Optional, List
import json
import logging
logger = logging.getLogger(__name__)

import requests
import logging

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_report(self, dataset_id: str, format_type: str = 'pdf') -> bytes:
        """Generate and download report in specified format"""
        try:
            url = f"{self.base_url}/api/v1/analysis/{dataset_id}/generate-report"
            params = {
                'format_type': format_type,
                'include_eda': True,
                'include_modeling': True
            }
            
            logger.info(f"Requesting report: {url} with params {params}")
            
            response = self.session.post(
                url,
                params=params,
                timeout=300  # 5 minute timeout
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Report generation failed: {error_detail}")
                raise Exception(f"Report generation failed (Status {response.status_code}): {error_detail}")
            
            return response.content
            
        except requests.exceptions.Timeout:
            raise Exception("Report generation timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to backend server. Please check if it's running.")
        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            raise

    
    def upload_dataset(self, uploaded_file) -> Dict[str, Any]:
        """Upload dataset to backend"""
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        response = self.session.post(f"{self.base_url}/api/v1/data/upload", files=files)
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.text}")
        
        return response.json()
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset information"""
        response = self.session.get(f"{self.base_url}/api/v1/data/{dataset_id}/info")
        
        if response.status_code != 200:
            raise Exception(f"Failed to get dataset info: {response.text}")
        
        return response.json()
    
    def get_target_suggestions(self, dataset_id: str) -> Dict[str, Any]:
        """Get target column suggestions"""
        response = self.session.get(f"{self.base_url}/api/v1/data/{dataset_id}/suggestions")
        
        if response.status_code != 200:
            raise Exception(f"Failed to get suggestions: {response.text}")
        
        return response.json()
    
    def perform_eda(self, dataset_id: str, target_column: Optional[str] = None, 
                   custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        params = {}
        if target_column:
            params['target_column'] = target_column
        if custom_prompt:
            params['custom_prompt'] = custom_prompt
        
        response = self.session.post(
            f"{self.base_url}/api/v1/analysis/{dataset_id}/eda",
            params=params
        )
        
        if response.status_code != 200:
            raise Exception(f"EDA failed: {response.text}")
        
        return response.json()
    
    def train_models(self, dataset_id: str, target_column: str, task_type: str = "auto",
                    test_size: float = 0.2, algorithms: Optional[List[str]] = None,
                    hyperparameter_tuning: bool = True, max_trials: int = 50) -> Dict[str, Any]:
        """Train machine learning models"""
        params = {
            'target_column': target_column,
            'task_type': task_type,
            'test_size': test_size,
            'hyperparameter_tuning': hyperparameter_tuning,
            'max_trials': max_trials
        }
        
        if algorithms:
            params['algorithms'] = algorithms
        
        response = self.session.post(
            f"{self.base_url}/api/v1/modeling/{dataset_id}/train",
            params=params
        )
        
        if response.status_code != 200:
            raise Exception(f"Model training failed: {response.text}")
        
        return response.json()
    
    def get_model_suggestions(self, dataset_id: str, target_column: str) -> Dict[str, Any]:
        """Get model algorithm suggestions"""
        params = {'target_column': target_column}
        
        response = self.session.get(
            f"{self.base_url}/api/v1/modeling/{dataset_id}/model-suggestions",
            params=params
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get model suggestions: {response.text}")
        
        return response.json()
    


