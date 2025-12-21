from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
import hmac
import time
import logging
from functools import wraps
import redis
from ipaddress import ip_address, ip_network
import re

from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token security
security = HTTPBearer()

class SecurityManager:
    """Advanced security management for the application"""
    
    def __init__(self):
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.secret_key = settings.SECRET_KEY
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = {}
        
        # Session management
        self.active_sessions = {}
        
        # Security policies
        self.password_policy = {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True,
            'max_age_days': 90
        }
        
        # Rate limiting policies
        self.rate_limits = {
            'api_default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            'upload': {'requests': 10, 'window': 3600},        # 10 uploads per hour
            'analysis': {'requests': 20, 'window': 3600},      # 20 analyses per hour
            'modeling': {'requests': 5, 'window': 3600}        # 5 model trainings per hour
        }
        
        # Trusted IP networks
        self.trusted_networks = [
            ip_network('127.0.0.0/8'),    # Localhost
            ip_network('10.0.0.0/8'),     # Private network
            ip_network('172.16.0.0/12'),  # Private network
            ip_network('192.168.0.0/16')  # Private network
        ]
    
    def create_access_token(self, data: Dict[Any, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})
        
        # Add session ID for token tracking
        session_id = secrets.token_urlsafe(32)
        to_encode.update({"session_id": session_id})
        
        # Store session info
        self.active_sessions[session_id] = {
            'created_at': datetime.utcnow(),
            'expires_at': expire,
            'user_data': data
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[Any, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if session is still active
            session_id = payload.get("session_id")
            if session_id and session_id not in self.active_sessions:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session has been revoked"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def revoke_session(self, session_id: str):
        """Revoke a specific session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} revoked")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if session_data['expires_at'] < current_time
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password against security policy"""
        policy = self.password_policy
        issues = []
        
        if len(password) < policy['min_length']:
            issues.append(f"Password must be at least {policy['min_length']} characters long")
        
        if policy['require_uppercase'] and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if policy['require_lowercase'] and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if policy['require_digits'] and not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        if policy['require_special'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        
        # Check for common weak patterns
        common_patterns = ['123456', 'password', 'qwerty', 'abc123']
        if any(pattern in password.lower() for pattern in common_patterns):
            issues.append("Password contains common weak patterns")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'strength_score': max(0, 100 - len(issues) * 20)
        }
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def create_api_key(self, user_id: str, description: str = "") -> Dict[str, str]:
        """Create API key for user"""
        api_key = f"adsai_{secrets.token_urlsafe(32)}"
        api_secret = secrets.token_urlsafe(64)
        
        # In production, store this in database
        key_data = {
            'user_id': user_id,
            'description': description,
            'created_at': datetime.utcnow().isoformat(),
            'last_used': None,
            'is_active': True
        }
        
        return {
            'api_key': api_key,
            'api_secret': api_secret,
            'key_data': key_data
        }
    
    def validate_api_key(self, api_key: str, api_secret: str) -> bool:
        """Validate API key and secret"""
        # In production, verify against database
        # This is a placeholder implementation
        return api_key.startswith('adsai_') and len(api_secret) == 86
    
    def check_rate_limit(self, identifier: str, endpoint_type: str = 'api_default') -> Dict[str, Any]:
        """Check rate limiting for requests"""
        current_time = time.time()
        policy = self.rate_limits.get(endpoint_type, self.rate_limits['api_default'])
        
        window_start = current_time - policy['window']
        
        # Clean old entries
        if identifier in self.rate_limit_storage:
            self.rate_limit_storage[identifier] = [
                timestamp for timestamp in self.rate_limit_storage[identifier]
                if timestamp > window_start
            ]
        else:
            self.rate_limit_storage[identifier] = []
        
        # Check current request count
        request_count = len(self.rate_limit_storage[identifier])
        
        if request_count >= policy['requests']:
            return {
                'allowed': False,
                'requests_made': request_count,
                'requests_allowed': policy['requests'],
                'window_seconds': policy['window'],
                'retry_after': int(self.rate_limit_storage[identifier][0] + policy['window'] - current_time)
            }
        
        # Add current request
        self.rate_limit_storage[identifier].append(current_time)
        
        return {
            'allowed': True,
            'requests_made': request_count + 1,
            'requests_allowed': policy['requests'],
            'window_seconds': policy['window'],
            'retry_after': 0
        }
    
    def is_ip_trusted(self, ip_addr: str) -> bool:
        """Check if IP address is in trusted networks"""
        try:
            ip = ip_address(ip_addr)
            return any(ip in network for network in self.trusted_networks)
        except ValueError:
            logger.warning(f"Invalid IP address: {ip_addr}")
            return False
    
    def create_csrf_token(self, session_id: str) -> str:
        """Create CSRF token for form protection"""
        timestamp = str(int(time.time()))
        message = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def verify_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token"""
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check token age
            if time.time() - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp_str}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, TypeError):
            return False
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(input_data, str):
            return str(input_data)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        sanitized = input_data
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = 'INFO'):
        """Log security-related events"""
        security_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        if severity in ['WARNING', 'ERROR', 'CRITICAL']:
            logger.warning(f"Security Event: {security_log}")
        else:
            logger.info(f"Security Event: {security_log}")
        
        # In production, send to security monitoring system
        return security_log

# Global security manager instance
security_manager = SecurityManager()

# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    return payload

def require_api_key(api_key: str, api_secret: str):
    """Require valid API key for access"""
    if not security_manager.validate_api_key(api_key, api_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

def rate_limit(endpoint_type: str = 'api_default'):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract IP from request (simplified)
            client_ip = "127.0.0.1"  # In production, extract from request
            
            rate_check = security_manager.check_rate_limit(client_ip, endpoint_type)
            
            if not rate_check['allowed']:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(rate_check['retry_after'])}
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
