import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Accept': 'application/json'
  }
});

export const dataAPI = {
  uploadDataset: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  getDatasetSummary: async (datasetId) => {
    const response = await api.get(`/data/${datasetId}/summary`);
    return response.data;
  }
};

export const analysisAPI = {
  getAnalysis: async (datasetId) => {
    const response = await api.post(`/analysis/${datasetId}/eda`);
    return response.data;
  },
  generateReport: async (datasetId, modelId) => {
    // Generate and download PDF
    const response = await api.post(`/analysis/${datasetId}/generate-report?format_type=pdf&model_id=${modelId}`, {}, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `ai_ds_report_${datasetId}.pdf`);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
    return true;
  }
};

export const modelAPI = {
  trainModel: async (datasetId, params) => {
    const response = await api.post(`/modeling/${datasetId}/train?target_column=${params.target_column}&task_type=${params.task_type}&test_size=${params.test_size}&hyperparameter_tuning=${params.hyperparameter_tuning}`);
    return response.data;
  },
  getSuggestions: async (datasetId) => {
    const response = await api.get(`/modeling/${datasetId}/suggestions`);
    return response.data;
  }
};
