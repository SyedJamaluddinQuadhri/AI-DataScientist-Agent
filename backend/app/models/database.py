from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    rows_count = Column(Integer)
    columns_count = Column(Integer)
    file_type = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    schema_info = Column(JSON)
    quality_report = Column(JSON)

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, nullable=False, index=True)
    analysis_type = Column(String, nullable=False)
    parameters = Column(JSON)
    results = Column(JSON)
    insights = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    execution_time = Column(Float)

class Model(Base):
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, nullable=False, index=True)
    algorithm = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    feature_columns = Column(JSON)
    hyperparameters = Column(JSON)
    performance_metrics = Column(JSON)
    model_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    training_time = Column(Float)

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, nullable=False, index=True)
    report_type = Column(String, nullable=False)
    content = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String)
