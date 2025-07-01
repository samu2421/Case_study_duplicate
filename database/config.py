"""
Database configuration and connection management for the Virtual Glasses Try-On system.
COMPLETE FIXED VERSION - This resolves all SQL parameter and method issues.
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
        """Create or update the selfies table to match expected schema."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS diffusion.selfies (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            file_path TEXT,
            file_url TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size_kb INTEGER,
            image_width INTEGER,
            image_height INTEGER,
            face_detected BOOLEAN DEFAULT FALSE,
            preprocessing_status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- Additional columns for the Virtual Try-On system
            image_data BYTEA,
            quality_score FLOAT DEFAULT 0.0,
            age_group VARCHAR(50),
            gender VARCHAR(20),
            ethnicity VARCHAR(50),
            face_landmarks TEXT,
            source_dataset VARCHAR(100) DEFAULT 'SCUT-FBP5500'
        );
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(create_table_sql))
                conn.commit()
            logger.info("Selfies table created/updated successfully")
        except Exception as e:
            logger.error(f"Failed to create selfies table: {e}")
            raise
    
    def add_missing_columns(self):
        """Add missing columns to existing tables."""
        try:
            # Check and add missing columns to selfies table
            missing_columns = [
                ("image_data", "BYTEA"),
                ("quality_score", "FLOAT DEFAULT 0.0"),
                ("age_group", "VARCHAR(50)"),
                ("gender", "VARCHAR(20)"),
                ("ethnicity", "VARCHAR(50)"),
                ("face_landmarks", "TEXT"),
                ("source_dataset", "VARCHAR(100) DEFAULT 'SCUT-FBP5500'")
            ]
            
            for column_name, column_def in missing_columns:
                try:
                    alter_sql = f"ALTER TABLE {self.config['schema']}.selfies ADD COLUMN IF NOT EXISTS {column_name} {column_def};"
                    with self.engine.connect() as conn:
                        conn.execute(sa.text(alter_sql))
                        conn.commit()
                    logger.debug(f"Added column: {column_name}")
                except Exception as e:
                    logger.debug(f"Column {column_name} might already exist: {e}")
            
            logger.info("Missing columns added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add missing columns: {e}")
    
    def insert_selfie(self, filename: str, image_data: bytes, metadata: dict) -> str:
        """Insert a new selfie record and return the ID."""
        insert_sql = """
        INSERT INTO diffusion.selfies 
        (filename, file_path, image_data, image_width, image_height, face_detected, 
         age_group, gender, quality_score, source_dataset)
        VALUES (:filename, :file_path, :image_data, :width, :height, 
                :face_detected, :age_group, :gender, :quality_score, :source_dataset)
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
                    'quality_score': metadata.get('quality_score', 0.0),
                    'source_dataset': metadata.get('source_dataset', 'SCUT-FBP5500')
                })
                conn.commit()
                return result.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to insert selfie: {e}")
            raise
    
    def get_selfies_batch(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Get a batch of selfie records."""
        query = """
        SELECT id, filename, image_width, image_height, face_detected, 
               age_group, gender, quality_score, created_at
        FROM diffusion.selfies
        ORDER BY created_at
        LIMIT :limit OFFSET :offset;
        """
        return self.execute_query(query, {'limit': limit, 'offset': offset})
    
    def get_glasses_batch(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Get a batch of glasses records."""
        query = """
        SELECT id, title, main_image, published, highlight
        FROM diffusion.frames
        ORDER BY product_sorting
        LIMIT :limit OFFSET :offset;
        """
        return self.execute_query(query, {'limit': limit, 'offset': offset})
    
    def get_selfie_image_data(self, selfie_id: str) -> Optional[bytes]:
        """Get image data for a specific selfie."""
        query = """
        SELECT image_data FROM diffusion.selfies WHERE id = :selfie_id;
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
        SET face_detected = :face_detected,
            face_landmarks = :face_landmarks,
            quality_score = :quality_score,
            preprocessing_status = :preprocessing_status,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = :selfie_id;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(update_sql), {
                    'selfie_id': selfie_id,
                    'face_detected': metadata.get('face_detected'),
                    'face_landmarks': metadata.get('face_landmarks'),
                    'quality_score': metadata.get('quality_score'),
                    'preprocessing_status': metadata.get('preprocessing_status', 'completed')
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update selfie metadata: {e}")
            raise
    
    def setup_database_schema(self):
        """Setup the complete database schema - REQUIRED METHOD."""
        try:
            logger.info("Setting up database schema...")
            
            # Ensure connection
            if not self.engine:
                self.connect()
            
            # Create/update selfies table
            self.create_selfies_table()
            
            # Add missing columns to existing tables
            self.add_missing_columns()
            
            # Create processed_glasses table
            self._create_processed_glasses_table()
            
            logger.info("Database schema setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Database schema setup failed: {e}")
            return False
    
    def _create_processed_glasses_table(self):
        """Create the processed_glasses table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS diffusion.processed_glasses (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            original_id UUID REFERENCES diffusion.frames(id),
            title VARCHAR(255),
            glasses_type VARCHAR(50),
            transparency_ratio FLOAT DEFAULT 0.0,
            has_transparency BOOLEAN DEFAULT FALSE,
            processed_width INTEGER,
            processed_height INTEGER,
            image_data BYTEA,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(original_id)
        );
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(create_table_sql))
                conn.commit()
            logger.info("Processed glasses table created successfully")
        except Exception as e:
            logger.debug(f"Processed glasses table might already exist: {e}")
    
    def insert_processed_glasses(self, original_id: str, title: str, glasses_type: str, 
                                transparency_ratio: float, has_transparency: bool,
                                processed_width: int, processed_height: int, image_data: bytes) -> bool:
        """Insert processed glasses data - FIXED VERSION."""
        # CRITICAL FIX: Use :parameter syntax instead of %(parameter)s
        insert_sql = """
        INSERT INTO diffusion.processed_glasses 
        (original_id, title, glasses_type, transparency_ratio, has_transparency,
         processed_width, processed_height, image_data)
        VALUES (:original_id, :title, :glasses_type, :transparency_ratio,
                :has_transparency, :width, :height, :image_data)
        ON CONFLICT (original_id) DO UPDATE SET
            glasses_type = EXCLUDED.glasses_type,
            transparency_ratio = EXCLUDED.transparency_ratio,
            has_transparency = EXCLUDED.has_transparency,
            processed_width = EXCLUDED.processed_width,
            processed_height = EXCLUDED.processed_height,
            image_data = EXCLUDED.image_data;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(insert_sql), {
                    'original_id': original_id,
                    'title': title,
                    'glasses_type': glasses_type,
                    'transparency_ratio': transparency_ratio,
                    'has_transparency': has_transparency,
                    'width': processed_width,
                    'height': processed_height,
                    'image_data': image_data
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert processed glasses: {e}")
            return False
    
    def update_selfie_with_image_data(self, selfie_id: int, image_data: bytes, 
                                     width: int, height: int) -> bool:
        """Update selfie with image data - FIXED VERSION."""
        # CRITICAL FIX: Use :parameter syntax instead of %(parameter)s  
        update_sql = """
        UPDATE diffusion.selfies 
        SET image_data = :image_data,
            image_width = :width,
            image_height = :height,
            face_detected = true,
            quality_score = 0.7,
            preprocessing_status = 'completed'
        WHERE id = :selfie_id;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(update_sql), {
                    'selfie_id': selfie_id,
                    'image_data': image_data,
                    'width': width,
                    'height': height
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update selfie {selfie_id}: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()