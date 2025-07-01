"""
Database migration script to fix schema issues.
Updates existing tables to match the expected structure.
"""

import sys
from pathlib import Path
import logging
import sqlalchemy as sa

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.config import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_table_exists(table_name: str, schema: str = "diffusion") -> bool:
    """Check if a table exists."""
    try:
        query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{schema}' 
            AND table_name = '{table_name}'
        );
        """
        result = db_manager.execute_query(query)
        return result.iloc[0, 0] if len(result) > 0 else False
    except Exception as e:
        logger.error(f"Error checking table {table_name}: {e}")
        return False

def check_column_exists(table_name: str, column_name: str, schema: str = "diffusion") -> bool:
    """Check if a column exists in a table."""
    try:
        query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = '{schema}' 
            AND table_name = '{table_name}' 
            AND column_name = '{column_name}'
        );
        """
        result = db_manager.execute_query(query)
        return result.iloc[0, 0] if len(result) > 0 else False
    except Exception as e:
        logger.error(f"Error checking column {column_name} in {table_name}: {e}")
        return False

def migrate_selfies_table():
    """Update selfies table structure."""
    try:
        logger.info("Migrating selfies table...")
        
        # Add missing columns to selfies table
        missing_columns = [
            ("quality_score", "FLOAT DEFAULT 0.0"),
            ("gender", "VARCHAR(20)"),
            ("age_group", "VARCHAR(50)"),
            ("ethnicity", "VARCHAR(50)"),
            ("face_landmarks", "TEXT"),
            ("source_dataset", "VARCHAR(100) DEFAULT 'SCUT-FBP5500'"),
            ("image_data", "BYTEA")
        ]
        
        for column_name, column_def in missing_columns:
            if not check_column_exists("selfies", column_name):
                alter_sql = f"ALTER TABLE {db_manager.config['schema']}.selfies ADD COLUMN {column_name} {column_def};"
                logger.info(f"Adding column: {column_name}")
                with db_manager.engine.connect() as conn:
                    conn.execute(sa.text(alter_sql))
                    conn.commit()
            else:
                logger.info(f"Column {column_name} already exists")
        
        logger.info("✅ Selfies table migration completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Selfies table migration failed: {e}")
        return False

def create_processed_glasses_table():
    """Create the processed_glasses table."""
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
        
        # Create indexes
        index_sqls = [
            f"CREATE INDEX IF NOT EXISTS idx_processed_glasses_glasses_type ON {db_manager.config['schema']}.processed_glasses(glasses_type);",
            f"CREATE INDEX IF NOT EXISTS idx_processed_glasses_has_transparency ON {db_manager.config['schema']}.processed_glasses(has_transparency);",
            f"CREATE INDEX IF NOT EXISTS idx_processed_glasses_transparency_ratio ON {db_manager.config['schema']}.processed_glasses(transparency_ratio);"
        ]
        
        for index_sql in index_sqls:
            try:
                with db_manager.engine.connect() as conn:
                    conn.execute(sa.text(index_sql))
                    conn.commit()
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        logger.info("✅ Processed glasses table created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Processed glasses table creation failed: {e}")
        return False

def update_indexes():
    """Create missing indexes for better performance."""
    try:
        logger.info("Creating missing indexes...")
        
        index_sqls = [
            f"CREATE INDEX IF NOT EXISTS idx_selfies_face_detected ON {db_manager.config['schema']}.selfies(face_detected);",
            f"CREATE INDEX IF NOT EXISTS idx_selfies_quality_score ON {db_manager.config['schema']}.selfies(quality_score);",
            f"CREATE INDEX IF NOT EXISTS idx_selfies_source_dataset ON {db_manager.config['schema']}.selfies(source_dataset);"
        ]
        
        for index_sql in index_sqls:
            try:
                with db_manager.engine.connect() as conn:
                    conn.execute(sa.text(index_sql))
                    conn.commit()
                logger.info(f"Index created: {index_sql.split('idx_')[1].split(' ON')[0]}")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        logger.info("✅ Indexes updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Index update failed: {e}")
        return False

def verify_migration():
    """Verify that migration was successful."""
    try:
        logger.info("Verifying migration...")
        
        # Check selfies table structure
        required_selfie_columns = ["quality_score", "gender", "age_group", "image_data"]
        for column in required_selfie_columns:
            if not check_column_exists("selfies", column):
                logger.error(f"❌ Missing column in selfies table: {column}")
                return False
        
        # Check processed_glasses table exists
        if not check_table_exists("processed_glasses"):
            logger.error("❌ processed_glasses table does not exist")
            return False
        
        # Test queries
        test_queries = [
            f"SELECT COUNT(*) FROM {db_manager.config['schema']}.selfies;",
            f"SELECT COUNT(*) FROM {db_manager.config['schema']}.frames;",
            f"SELECT COUNT(*) FROM {db_manager.config['schema']}.processed_glasses;"
        ]
        
        for query in test_queries:
            try:
                result = db_manager.execute_query(query)
                table_name = query.split('FROM ')[1].split('.')[1].replace(';', '')
                count = result.iloc[0, 0]
                logger.info(f"✅ {table_name} table: {count} records")
            except Exception as e:
                logger.error(f"❌ Query failed: {query} - {e}")
                return False
        
        logger.info("✅ Migration verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration verification failed: {e}")
        return False

def main():
    """Run the complete migration."""
    try:
        logger.info("🚀 Starting database migration...")
        
        # Connect to database
        if not db_manager.connect():
            logger.error("❌ Failed to connect to database")
            return 1
        
        # Run migrations
        migrations = [
            ("Selfies table migration", migrate_selfies_table),
            ("Processed glasses table creation", create_processed_glasses_table),
            ("Index updates", update_indexes),
            ("Migration verification", verify_migration)
        ]
        
        for name, migration_func in migrations:
            logger.info(f"\n--- {name} ---")
            if not migration_func():
                logger.error(f"❌ {name} failed")
                return 1
        
        logger.info("\n🎉 Database migration completed successfully!")
        logger.info("You can now run the pipeline again.")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())