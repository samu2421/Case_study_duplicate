# config/database_config.py
"""
Database configuration and management for the Virtual Glasses Try-On project
"""
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import logging

# Load environment variables
load_dotenv()

class DatabaseManager:
    """Manages database connections and operations for the project"""
    
    def __init__(self):
        """Initialize database connection"""
        self.user = os.getenv("POSTGRES_USER", "student_diff")
        self.password = os.getenv("POSTGRES_PASSWORD", "diff_pass") 
        self.host = os.getenv("POSTGRES_HOST", "152.53.12.68")
        self.port = os.getenv("POSTGRES_PORT", "4000")
        self.database = os.getenv("POSTGRES_DB", "postgres")
        self.schema = "diffusion"
        
        # Create connection URL
        self.postgres_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Create engine and session
        self.engine = create_engine(self.postgres_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                self.logger.info("✅ Database connection successful")
                return True
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def create_selfies_table(self) -> bool:
        """Create selfies table in the database"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.selfies (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            file_url TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size_kb INTEGER,
            image_width INTEGER,
            image_height INTEGER,
            face_detected BOOLEAN DEFAULT FALSE,
            preprocessing_status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            with self.engine.connect() as connection:
                connection.execute(text(create_table_sql))
                connection.commit()
                self.logger.info("✅ Selfies table created successfully")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create selfies table: {e}")
            return False
    
    def get_glasses_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve glasses data from the frames table"""
        query = f"SELECT * FROM {self.schema}.frames"
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"✅ Retrieved {len(df)} glasses records")
            return df
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve glasses data: {e}")
            return pd.DataFrame()
    
    def get_selfies_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve selfies data from the selfies table"""
        query = f"SELECT * FROM {self.schema}.selfies"
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"✅ Retrieved {len(df)} selfies records")
            return df
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve selfies data: {e}")
            return pd.DataFrame()
    
    def insert_selfie_record(self, filename: str, file_path: str, 
                           file_url: Optional[str] = None, 
                           file_size_kb: Optional[int] = None,
                           image_width: Optional[int] = None,
                           image_height: Optional[int] = None) -> bool:
        """Insert a new selfie record into the database"""
        
        insert_sql = f"""
        INSERT INTO {self.schema}.selfies 
        (filename, file_path, file_url, file_size_kb, image_width, image_height)
        VALUES (:filename, :file_path, :file_url, :file_size_kb, :image_width, :image_height)
        ON CONFLICT (filename) DO UPDATE SET
            file_path = EXCLUDED.file_path,
            file_url = EXCLUDED.file_url,
            file_size_kb = EXCLUDED.file_size_kb,
            image_width = EXCLUDED.image_width,
            image_height = EXCLUDED.image_height,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            with self.engine.connect() as connection:
                connection.execute(text(insert_sql), {
                    'filename': filename,
                    'file_path': file_path,
                    'file_url': file_url,
                    'file_size_kb': file_size_kb,
                    'image_width': image_width,
                    'image_height': image_height
                })
                connection.commit()
                self.logger.info(f"✅ Inserted selfie record: {filename}")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to insert selfie record: {e}")
            return False
    
    def update_preprocessing_status(self, filename: str, status: str, 
                                  face_detected: Optional[bool] = None) -> bool:
        """Update preprocessing status for a selfie"""
        
        update_sql = f"""
        UPDATE {self.schema}.selfies 
        SET preprocessing_status = :status,
            face_detected = COALESCE(:face_detected, face_detected),
            updated_at = CURRENT_TIMESTAMP
        WHERE filename = :filename
        """
        
        try:
            with self.engine.connect() as connection:
                connection.execute(text(update_sql), {
                    'filename': filename,
                    'status': status,
                    'face_detected': face_detected
                })
                connection.commit()
                self.logger.info(f"✅ Updated preprocessing status for: {filename}")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update preprocessing status: {e}")
            return False
    
    def get_tables_info(self) -> Dict[str, Any]:
        """Get information about all tables in the schema"""
        tables_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.schema}'
        AND table_type = 'BASE TABLE';
        """
        
        try:
            tables_df = pd.read_sql(tables_query, self.engine)
            tables_info = {'tables': tables_df['table_name'].tolist()}
            
            # Get row counts for each table
            for table in tables_info['tables']:
                count_query = f"SELECT COUNT(*) as count FROM {self.schema}.{table}"
                count_df = pd.read_sql(count_query, self.engine)
                tables_info[f'{table}_count'] = count_df['count'].iloc[0]
            
            return tables_info
        except Exception as e:
            self.logger.error(f"❌ Failed to get tables info: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Test connection
    if db_manager.test_connection():
        # Create selfies table
        db_manager.create_selfies_table()
        
        # Get tables info
        tables_info = db_manager.get_tables_info()
        print("Database Tables Info:")
        for key, value in tables_info.items():
            print(f"  {key}: {value}")
        
        # Test retrieving glasses data
        glasses_df = db_manager.get_glasses_data(limit=5)
        print(f"\nSample glasses data shape: {glasses_df.shape}")
        if not glasses_df.empty:
            print("Glasses columns:", list(glasses_df.columns))
        
        # Test retrieving selfies data
        selfies_df = db_manager.get_selfies_data(limit=5)
        print(f"\nSample selfies data shape: {selfies_df.shape}")
        if not selfies_df.empty:
            print("Selfies columns:", list(selfies_df.columns))