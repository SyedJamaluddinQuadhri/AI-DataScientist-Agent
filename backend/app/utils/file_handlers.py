import os
import shutil
import uuid
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
import logging
import hashlib
import tempfile
from werkzeug.utils import secure_filename

from app.core.config import settings
from app.core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)

class FileHandler:
    """Advanced file handling utilities with security and validation"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Supported MIME types
        self.supported_mime_types = {
            'text/csv': 'csv',
            'application/vnd.ms-excel': 'xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/json': 'json',
            'application/octet-stream': 'parquet',  # Parquet files
            'text/tab-separated-values': 'tsv',
            'text/plain': 'txt'
        }
        
        # Maximum file sizes by type (in bytes)
        self.max_file_sizes = {
            'csv': 500 * 1024 * 1024,  # 500MB
            'xlsx': 100 * 1024 * 1024,  # 100MB
            'json': 200 * 1024 * 1024,  # 200MB
            'parquet': 1024 * 1024 * 1024,  # 1GB
            'default': settings.MAX_UPLOAD_SIZE
        }
    
    def validate_file(self, file: BinaryIO, filename: str) -> Dict[str, Any]:
        """Comprehensive file validation"""
        try:
            # Reset file pointer
            file.seek(0)
            
            # Get file info
            file_size = len(file.read())
            file.seek(0)
            
            # Validate filename
            secure_name = secure_filename(filename)
            if not secure_name:
                raise DataProcessingException("Invalid filename")
            
            # Get file extension
            file_extension = Path(filename).suffix.lower().lstrip('.')
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                raise DataProcessingException(
                    f"File type not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
                )
            
            # Validate file size
            max_size = self.max_file_sizes.get(file_extension, self.max_file_sizes['default'])
            if file_size > max_size:
                raise DataProcessingException(
                    f"File too large. Maximum size for {file_extension}: {max_size/1024/1024:.1f}MB"
                )
            
            # Validate MIME type
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type and mime_type not in self.supported_mime_types:
                logger.warning(f"Unexpected MIME type: {mime_type} for file: {filename}")
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file)
            
            return {
                'filename': secure_name,
                'file_extension': file_extension,
                'file_size': file_size,
                'mime_type': mime_type,
                'file_hash': file_hash,
                'validation_status': 'passed'
            }
            
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            raise DataProcessingException(f"File validation failed: {str(e)}")
    
    def save_file(self, file: BinaryIO, filename: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Securely save uploaded file"""
        try:
            # Validate file first
            validation_result = self.validate_file(file, filename)
            
            # Generate dataset ID if not provided
            if not dataset_id:
                dataset_id = str(uuid.uuid4())
            
            # Create secure filename
            file_extension = validation_result['file_extension']
            secure_name = f"{dataset_id}.{file_extension}"
            file_path = self.upload_dir / secure_name
            
            # Save file
            file.seek(0)
            with open(file_path, 'wb') as buffer:
                shutil.copyfileobj(file, buffer)
            
            # Verify file integrity
            if not self._verify_file_integrity(file_path, validation_result['file_hash']):
                file_path.unlink()  # Remove corrupted file
                raise DataProcessingException("File integrity check failed")
            
            # Set secure permissions
            os.chmod(file_path, 0o644)
            
            logger.info(f"File saved successfully: {secure_name}")
            
            return {
                'dataset_id': dataset_id,
                'filename': secure_name,
                'file_path': str(file_path),
                'file_size': validation_result['file_size'],
                'file_hash': validation_result['file_hash'],
                'save_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"File save failed: {str(e)}")
            raise DataProcessingException(f"Failed to save file: {str(e)}")
    
    def get_file_path(self, dataset_id: str) -> Optional[Path]:
        """Get file path for dataset ID"""
        for ext in settings.ALLOWED_EXTENSIONS:
            file_path = self.upload_dir / f"{dataset_id}.{ext}"
            if file_path.exists():
                return file_path
        return None
    
    def delete_file(self, dataset_id: str) -> Dict[str, Any]:
        """Safely delete file and related data"""
        try:
            deleted_files = []
            
            # Delete all files related to this dataset
            for file_path in self.upload_dir.glob(f"{dataset_id}*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(str(file_path))
            
            # Delete model files if they exist
            model_dir = Path("models") / dataset_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
                deleted_files.append(str(model_dir))
            
            logger.info(f"Deleted files for dataset {dataset_id}: {deleted_files}")
            
            return {
                'dataset_id': dataset_id,
                'deleted_files': deleted_files,
                'deletion_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"File deletion failed: {str(e)}")
            raise DataProcessingException(f"Failed to delete files: {str(e)}")
    
    def create_temp_file(self, content: bytes, suffix: str = ".tmp") -> str:
        """Create temporary file for processing"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(content)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Temp file creation failed: {str(e)}")
            raise DataProcessingException(f"Failed to create temp file: {str(e)}")
    
    def cleanup_temp_files(self, temp_dir: Optional[str] = None):
        """Clean up temporary files"""
        try:
            if temp_dir:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            else:
                # Clean up system temp directory for our files
                temp_dir = tempfile.gettempdir()
                for temp_file in Path(temp_dir).glob("tmp*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                    except:
                        pass  # Ignore errors for temp cleanup
            
            logger.info("Temporary files cleaned up")
            
        except Exception as e:
            logger.warning(f"Temp file cleanup warning: {str(e)}")
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information"""
        try:
            stat = file_path.stat()
            
            return {
                'filename': file_path.name,
                'file_size': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'file_extension': file_path.suffix.lower().lstrip('.'),
                'mime_type': mimetypes.guess_type(str(file_path))[0],
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK)
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info: {str(e)}")
            return {}
    
    def _calculate_file_hash(self, file: BinaryIO) -> str:
        """Calculate SHA-256 hash of file content"""
        hasher = hashlib.sha256()
        file.seek(0)
        
        while chunk := file.read(8192):
            hasher.update(chunk)
        
        file.seek(0)
        return hasher.hexdigest()
    
    def _verify_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity using hash"""
        try:
            with open(file_path, 'rb') as f:
                calculated_hash = self._calculate_file_hash(f)
            return calculated_hash == expected_hash
        except Exception as e:
            logger.error(f"File integrity check failed: {str(e)}")
            return False
    
    def archive_old_files(self, days_old: int = 30):
        """Archive files older than specified days"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            archived_files = []
            archive_dir = self.upload_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    archive_path = archive_dir / file_path.name
                    shutil.move(str(file_path), str(archive_path))
                    archived_files.append(str(file_path))
            
            logger.info(f"Archived {len(archived_files)} old files")
            return archived_files
            
        except Exception as e:
            logger.error(f"File archiving failed: {str(e)}")
            return []
