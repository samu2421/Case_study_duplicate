"""
Database migration script to fix schema issues and ensure compatibility.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.config import db_manager
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_fix_schema():
    """Check and fix database schema issues."""
    try:
        logger.info("🔧 Checking and fixing database schema...")
        
        # Connect to database
        if not db_manager.connect():
            logger.error("❌ Failed to connect to database")
            return False
        
        # Setup complete database schema
        success = db_manager.setup_database_schema()
        
        if success:
            logger.info("✅ Database schema check and fix completed")
            return True
        else:
            logger.error("❌ Database schema fix failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Schema check/fix failed: {e}")
        return False

def verify_schema():
    """Verify that the schema is correct."""
    try:
        logger.info("🔍 Verifying database schema...")
        
        # Check tables exist
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'diffusion'
        ORDER BY table_name;
        """
        
        tables = db_manager.execute_query(tables_query)
        table_names = list(tables['table_name'])
        
        logger.info(f"📊 Found tables: {table_names}")
        
        # Check required tables
        required_tables = ['frames', 'selfies']
        missing_tables = [t for t in required_tables if t not in table_names]
        
        if missing_tables:
            logger.error(f"❌ Missing required tables: {missing_tables}")
            return False
        
        # Check selfies table columns
        selfies_columns_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'diffusion' 
        AND table_name = 'selfies'
        ORDER BY column_name;
        """
        
        selfies_columns = db_manager.execute_query(selfies_columns_query)
        column_names = list(selfies_columns['column_name'])
        
        logger.info(f"📊 Selfies table columns: {column_names}")
        
        # Check for required columns
        required_columns = ['id', 'filename', 'face_detected', 'quality_score']
        missing_columns = [c for c in required_columns if c not in column_names]
        
        if missing_columns:
            logger.warning(f"⚠️ Missing recommended columns: {missing_columns}")
        
        logger.info("✅ Database schema verification completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema verification failed: {e}")
        return False

def test_basic_operations():
    """Test basic database operations."""
    try:
        logger.info("🧪 Testing basic database operations...")
        
        # Test basic query
        test_query = "SELECT 1 as test;"
        result = db_manager.execute_query(test_query)
        
        if len(result) > 0 and result.iloc[0]['test'] == 1:
            logger.info("✅ Basic query test passed")
        else:
            logger.error("❌ Basic query test failed")
            return False
        
        # Test tables access
        frames_count_query = "SELECT COUNT(*) as count FROM diffusion.frames;"
        frames_result = db_manager.execute_query(frames_count_query)
        frames_count = frames_result.iloc[0]['count']
        
        selfies_count_query = "SELECT COUNT(*) as count FROM diffusion.selfies;"
        selfies_result = db_manager.execute_query(selfies_count_query)
        selfies_count = selfies_result.iloc[0]['count']
        
        logger.info(f"📊 Database contents:")
        logger.info(f"   Frames: {frames_count}")
        logger.info(f"   Selfies: {selfies_count}")
        
        logger.info("✅ Basic operations test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic operations test failed: {e}")
        return False

def main():
    """Main migration function."""
    try:
        logger.info("🚀 STARTING DATABASE MIGRATION")
        logger.info("=" * 50)
        
        # Step 1: Check and fix schema
        logger.info("\n--- STEP 1: SCHEMA CHECK AND FIX ---")
        if not check_and_fix_schema():
            return 1
        
        # Step 2: Verify schema
        logger.info("\n--- STEP 2: SCHEMA VERIFICATION ---")
        if not verify_schema():
            return 1
        
        # Step 3: Test basic operations
        logger.info("\n--- STEP 3: BASIC OPERATIONS TEST ---")
        if not test_basic_operations():
            return 1
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 DATABASE MIGRATION COMPLETED SUCCESSFULLY!")
        logger.info("🚀 Ready to run: python cli.py pipeline")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())