"""
Database configuration and connection management for the Virtual Glasses Try-On system.
"""

from pathlib import Path
from typing import Optional
import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, Table, Column, String, LargeBinary, DateTime, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    "host": "152.53.12.68",
    "port": 4000,
    "user": "student_diff",
    "password": "diff_pass",
    "database": "postgres",
    "schema": "diffusion"
}

Base = declarative_base()

class DatabaseManager:
    """Manages database connections and operations for the Virtual Glasses Try-On system."""
    
    def __init__(self, config: dict = DB_CONFIG):
        """Initialize database manager with connection configuration."""
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self.metadata = None
        
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            connection_string = (
                f"postgresql://{self.config['user']}:{self.config['password']}@"
                f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            self.engine = create_engine(
                connection_string,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20
            )
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.metadata = MetaData(schema=self.config['schema'])
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            
            logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def get_session(self):
        """Get a new database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.SessionLocal()
    
    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params=params)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def create_selfies_table(self):
        """Create the selfies table for storing face images and metadata."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS diffusion.selfies (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename VARCHAR(255) NOT NULL,
            file_path VARCHAR(500),
            image_data BYTEA,
            width INTEGER,
            height INTEGER,
            face_detected BOOLEAN DEFAULT FALSE,
            face_landmarks TEXT,
            age_group VARCHAR(50),
            gender VARCHAR(20),
            ethnicity VARCHAR(50),
            quality_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_dataset VARCHAR(100) DEFAULT 'SCUT-FBP5500'
        );
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(create_table_sql))
                conn.commit()
            logger.info("Selfies table created successfully")
        except Exception as e:
            logger.error(f"Failed to create selfies table: {e}")
            raise
    
    def insert_selfie(self, filename: str, image_data: bytes, metadata: dict) -> str:
        """Insert a new selfie record and return the ID."""
        insert_sql = """
        INSERT INTO diffusion.selfies 
        (filename, file_path, image_data, width, height, face_detected, age_group, gender, quality_score)
        VALUES (%(filename)s, %(file_path)s, %(image_data)s, %(width)s, %(height)s, 
                %(face_detected)s, %(age_group)s, %(gender)s, %(quality_score)s)
        RETURNING id;
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(insert_sql), {
                    'filename': filename,
                    'file_path': metadata.get('file_path'),
                    'image_data': image_data,
                    'width': metadata.get('width'),
                    'height': metadata.get('height'),
                    'face_detected': metadata.get('face_detected', False),
                    'age_group': metadata.get('age_group'),
                    'gender': metadata.get('gender'),
                    'quality_score': metadata.get('quality_score', 0.0)
                })
                conn.commit()
                return result.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to insert selfie: {e}")
            raise
    
    def get_selfies_batch(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Get a batch of selfie records."""
        query = """
        SELECT id, filename, width, height, face_detected, age_group, gender, quality_score, created_at
        FROM diffusion.selfies
        ORDER BY created_at
        LIMIT %(limit)s OFFSET %(offset)s;
        """
        return self.execute_query(query, {'limit': limit, 'offset': offset})
    
    def get_glasses_batch(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Get a batch of glasses records."""
        query = """
        SELECT id, title, main_image, published, highlight
        FROM diffusion.frames
        ORDER BY product_sorting
        LIMIT %(limit)s OFFSET %(offset)s;
        """
        return self.execute_query(query, {'limit': limit, 'offset': offset})
    
    def get_selfie_image_data(self, selfie_id: str) -> Optional[bytes]:
        """Get image data for a specific selfie."""
        query = """
        SELECT image_data FROM diffusion.selfies WHERE id = %(selfie_id)s;
        """
        try:
            result = self.execute_query(query, {'selfie_id': selfie_id})
            if len(result) > 0 and result.iloc[0]['image_data'] is not None:
                return result.iloc[0]['image_data']
            else:
                logger.warning(f"No image data found for selfie {selfie_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to get selfie image data: {e}")
            return None
    
    def reset_connection(self):
        """Reset database connection to clear any transaction issues."""
        try:
            if self.engine:
                self.engine.dispose()
            
            # Reconnect
            return self.connect()
        except Exception as e:
            logger.error(f"Failed to reset database connection: {e}")
            return False
    
    def update_selfie_metadata(self, selfie_id: str, metadata: dict):
        """Update selfie metadata after processing."""
        update_sql = """
        UPDATE diffusion.selfies 
        SET face_detected = %(face_detected)s,
            face_landmarks = %(face_landmarks)s,
            quality_score = %(quality_score)s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %(selfie_id)s;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(update_sql), {
                    'selfie_id': selfie_id,
                    'face_detected': metadata.get('face_detected'),
                    'face_landmarks': metadata.get('face_landmarks'),
                    'quality_score': metadata.get('quality_score')
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update selfie metadata: {e}")
            raise

# Global database manager instance
db_manager = DatabaseManager()