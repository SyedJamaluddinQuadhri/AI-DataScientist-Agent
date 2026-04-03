import sys
print("Starting imports")
try:
    from app.core.config import settings
    print("1 - config loaded")
    from app.core.exceptions import setup_exception_handlers
    print("2 - exceptions loaded")
    from app.models.database import Base
    print("3 - database Base loaded")
    from app.api.endpoints import data
    print("4 - endpoint data loaded")
    from app.api.endpoints import analysis
    print("5 - endpoint analysis loaded")
    from app.api.endpoints import modeling
    print("6 - endpoint modeling loaded")
    print("All imports successful")
except Exception as e:
    print(f"Exception: {e}")
