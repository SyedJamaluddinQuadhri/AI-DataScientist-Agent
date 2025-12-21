from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

class DataProcessingException(Exception):
    """Custom exception for data processing errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ModelTrainingException(Exception):
    """Custom exception for model training errors"""
    def __init__(self, message: str, model_type: str = None):
        self.message = message
        self.model_type = model_type
        super().__init__(self.message)

def setup_exception_handlers(app):
    """Setup custom exception handlers"""
    
    @app.exception_handler(DataProcessingException)
    async def data_processing_exception_handler(request: Request, exc: DataProcessingException):
        logger.error(f"Data processing error: {exc.message}, Details: {exc.details}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Data Processing Error",
                "message": exc.message,
                "details": exc.details
            }
        )
    
    @app.exception_handler(ModelTrainingException)
    async def model_training_exception_handler(request: Request, exc: ModelTrainingException):
        logger.error(f"Model training error: {exc.message}, Model: {exc.model_type}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Training Error",
                "message": exc.message,
                "model_type": exc.model_type
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )
