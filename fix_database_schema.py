"""
Script to fix database schema to match the expected structure.
This will add missing columns to existing tables.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from database.config import db_manager
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_selfies_table():
    """Add missing columns to selfies table."""
    try:
        logger.info("Fixing selfies table schema...")
        
        # Add missing columns to existing selfies table
        alter_commands = [
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0;",
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS age_group VARCHAR(50);",
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS gender VARCHAR(20);",
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS ethnicity VARCHAR(50);",
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS face_landmarks TEXT;",
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS source_dataset VARCHAR(100) DEFAULT 'SCUT-FBP5500';",
            "ALTER TABLE diffusion.selfies ADD COLUMN IF NOT EXISTS image_data BYTEA;"
        ]
        
        with db_manager.engine.connect() as conn:
            for cmd in alter_commands:
                logger.info(f"Executing: {cmd}")
                conn.execute(sa.text(cmd))
            conn.commit()
        
        logger.info("✅ Selfies table updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix selfies table: {e}")
        return False

def create_processed_glasses_table():
    """Create the missing processed_glasses table."""
    try:
        logger.info("Creating processed_glasses table...")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {db_manager.config['schema']}.processed_glasses (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            original_id UUID REFERENCES {db_manager.config['schema']}.frames(id),
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
        
        with db_manager.engine.connect() as conn:
            conn.execute(sa.text(create_table_sql))
            conn.commit()
        
        logger.info("✅ Processed glasses table created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create processed_glasses table: {e}")
        return False

def add_indexes():
    """Add useful indexes for better performance."""
    try:
        logger.info("Adding database indexes...")
        
        index_commands = [
            "CREATE INDEX IF NOT EXISTS idx_selfies_face_detected ON diffusion.selfies(face_detected);",
            "CREATE INDEX IF NOT EXISTS idx_selfies_quality_score ON diffusion.selfies(quality_score);",
            "CREATE INDEX IF NOT EXISTS idx_processed_glasses_type ON diffusion.processed_glasses(glasses_type);",
            "CREATE INDEX IF NOT EXISTS idx_processed_glasses_transparency ON diffusion.processed_glasses(has_transparency);"
        ]
        
        with db_manager.engine.connect() as conn:
            for cmd in index_commands:
                logger.info(f"Executing: {cmd}")
                conn.execute(sa.text(cmd))
            conn.commit()
        
        logger.info("✅ Indexes added successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to add indexes: {e}")
        return False

def verify_schema():
    """Verify the schema is correct after fixes."""
    try:
        logger.info("Verifying database schema...")
        
        # Check selfies table has required columns
        selfies_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'diffusion' 
        AND table_name = 'selfies'
        ORDER BY column_name;
        """
        
        selfies_columns = db_manager.execute_query(selfies_query)
        logger.info(f"Selfies table columns: {list(selfies_columns['column_name'])}")
        
        # Check processed_glasses table exists
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'diffusion'
        ORDER BY table_name;
        """
        
        tables = db_manager.execute_query(tables_query)
        logger.info(f"Available tables: {list(tables['table_name'])}")
        
        # Check if required columns exist
        required_selfie_columns = ['quality_score', 'age_group', 'gender', 'face_landmarks']
        existing_columns = list(selfies_columns['column_name'])
        
        missing_columns = [col for col in required_selfie_columns if col not in existing_columns]
        if missing_columns:
            logger.error(f"❌ Missing selfie columns: {missing_columns}")
            return False
        
        if 'processed_glasses' not in list(tables['table_name']):
            logger.error("❌ processed_glasses table not found")
            return False
        
        logger.info("✅ Database schema verification passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema verification failed: {e}")
        return False

def main():
    """Main function to fix database schema."""
    try:
        logger.info("🔧 Starting database schema fix...")
        
        # Connect to database
        if not db_manager.connect():
            logger.error("❌ Failed to connect to database")
            return 1
        
        # Fix schema step by step
        steps = [
            ("Fix selfies table", fix_selfies_table),
            ("Create processed_glasses table", create_processed_glasses_table),
            ("Add database indexes", add_indexes),
            ("Verify schema", verify_schema)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            if not step_func():
                logger.error(f"❌ Failed: {step_name}")
                return 1
        
        logger.info("\n🎉 Database schema fixed successfully!")
        logger.info("You can now run the pipeline again:")
        logger.info("python cli.py demo")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Schema fix failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())