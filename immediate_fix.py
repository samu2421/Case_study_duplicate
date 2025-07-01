#!/usr/bin/env python3
"""
Immediate fix script for Virtual Glasses Try-On system.
Solves SQL parameter issues and creates sample data for demo.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from database.config import db_manager
from image_processing.utils.image_utils import image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection and basic operations."""
    try:
        logger.info("🔌 Testing database connection...")
        
        if not db_manager.connect():
            logger.error("❌ Database connection failed")
            return False
        
        # Test basic query
        result = db_manager.execute_query("SELECT 1 as test;")
        if len(result) > 0 and result.iloc[0]['test'] == 1:
            logger.info("✅ Database connection and queries working")
            return True
        else:
            logger.error("❌ Database query test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def setup_database_schema():
    """Setup database schema using the fixed methods."""
    try:
        logger.info("🛠️ Setting up database schema...")
        
        if not db_manager.setup_database_schema():
            logger.error("❌ Schema setup failed")
            return False
        
        logger.info("✅ Database schema setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema setup failed: {e}")
        return False

def create_sample_selfie_data():
    """Create sample selfie data with proper image data."""
    try:
        logger.info("📸 Creating sample selfie data...")
        
        # Get selfies without image data
        query = """
        SELECT id, filename FROM diffusion.selfies 
        WHERE image_data IS NULL OR image_data = '' 
        LIMIT 10;
        """
        selfies = db_manager.execute_query(query)
        
        if len(selfies) == 0:
            logger.info("No selfies need image data")
            return True
        
        logger.info(f"Found {len(selfies)} selfies without image data")
        
        success_count = 0
        for _, row in selfies.iterrows():
            try:
                # Create synthetic face image
                image = create_synthetic_face_image()
                
                # Convert to bytes
                image_bytes = image_processor.image_to_bytes(image, 'JPEG')
                
                # Update database using fixed method
                if db_manager.update_selfie_with_image_data(
                    selfie_id=row['id'],
                    image_data=image_bytes,
                    width=512,
                    height=512
                ):
                    success_count += 1
                    logger.debug(f"✅ Updated selfie {row['id']}")
                
            except Exception as e:
                logger.warning(f"Failed to update selfie {row['id']}: {e}")
                continue
        
        logger.info(f"✅ Created image data for {success_count} selfies")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to create selfie data: {e}")
        return False

def create_synthetic_face_image() -> np.ndarray:
    """Create a synthetic face image for testing."""
    # Create base image with skin tone
    image = np.full((512, 512, 3), [220, 190, 160], dtype=np.uint8)
    
    # Add random background
    background_color = np.random.randint(200, 255, 3)
    image[:100, :] = background_color  # Top background
    image[400:, :] = background_color  # Bottom background
    image[:, :100] = background_color  # Left background
    image[:, 400:] = background_color  # Right background
    
    # Add face shape (ellipse)
    center = (256, 280)
    axes = (120, 150)
    face_color = [220, 190, 160]
    cv2.ellipse(image, center, axes, 0, 0, 360, face_color, -1)
    
    # Add eyes
    cv2.circle(image, (220, 240), 15, (80, 60, 40), -1)  # Left eye
    cv2.circle(image, (292, 240), 15, (80, 60, 40), -1)  # Right eye
    cv2.circle(image, (220, 240), 8, (20, 20, 20), -1)   # Left pupil
    cv2.circle(image, (292, 240), 8, (20, 20, 20), -1)   # Right pupil
    
    # Add eyebrows
    cv2.ellipse(image, (220, 220), (20, 8), 0, 0, 180, (100, 70, 50), -1)
    cv2.ellipse(image, (292, 220), (20, 8), 0, 0, 180, (100, 70, 50), -1)
    
    # Add nose
    nose_points = np.array([[256, 260], [250, 290], [262, 290]], np.int32)
    cv2.fillPoly(image, [nose_points], (200, 170, 140))
    
    # Add mouth
    cv2.ellipse(image, (256, 320), (25, 12), 0, 0, 180, (150, 100, 100), -1)
    
    # Add some texture/noise for realism
    noise = np.random.randint(-10, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def create_sample_glasses_data():
    """Create sample processed glasses data."""
    try:
        logger.info("🥽 Creating sample glasses data...")
        
        # Get a few glasses without processed data
        query = """
        SELECT f.id, f.title, f.main_image 
        FROM diffusion.frames f
        LEFT JOIN diffusion.processed_glasses pg ON f.id = pg.original_id
        WHERE pg.id IS NULL 
        AND f.main_image IS NOT NULL 
        AND f.main_image != ''
        LIMIT 5;
        """
        
        glasses = db_manager.execute_query(query)
        
        if len(glasses) == 0:
            logger.info("No glasses need processing")
            return True
        
        logger.info(f"Found {len(glasses)} glasses to process")
        
        success_count = 0
        for _, row in glasses.iterrows():
            try:
                # Create synthetic glasses image
                glasses_image = create_synthetic_glasses_image()
                
                # Convert to bytes
                image_bytes = image_processor.image_to_bytes(glasses_image, 'PNG')
                
                # Insert using fixed method
                if db_manager.insert_processed_glasses(
                    original_id=row['id'],
                    title=row['title'],
                    glasses_type='regular',
                    transparency_ratio=0.8,
                    has_transparency=True,
                    processed_width=512,
                    processed_height=512,
                    image_data=image_bytes
                ):
                    success_count += 1
                    logger.debug(f"✅ Processed glasses {row['title']}")
                
            except Exception as e:
                logger.warning(f"Failed to process glasses {row['title']}: {e}")
                continue
        
        logger.info(f"✅ Created {success_count} processed glasses")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to create glasses data: {e}")
        return False

def create_synthetic_glasses_image() -> np.ndarray:
    """Create a synthetic glasses image with transparency."""
    # Create RGBA image (with alpha channel)
    image = np.zeros((512, 512, 4), dtype=np.uint8)
    
    # Create glasses frame
    # Left lens
    cv2.circle(image, (180, 256), 80, (50, 50, 50, 255), 8)
    # Right lens  
    cv2.circle(image, (332, 256), 80, (50, 50, 50, 255), 8)
    # Bridge
    cv2.line(image, (260, 256), (252, 256), (50, 50, 50, 255), 6)
    # Left arm
    cv2.line(image, (100, 256), (50, 240), (50, 50, 50, 255), 6)
    # Right arm
    cv2.line(image, (412, 256), (462, 240), (50, 50, 50, 255), 6)
    
    # Add lens transparency effect
    cv2.circle(image, (180, 256), 75, (200, 220, 255, 30), -1)  # Slight blue tint
    cv2.circle(image, (332, 256), 75, (200, 220, 255, 30), -1)  # Slight blue tint
    
    return image

def verify_demo_readiness():
    """Verify that the system is ready for demo."""
    try:
        logger.info("🔍 Verifying demo readiness...")
        
        # Check usable selfies
        selfies_query = """
        SELECT COUNT(*) as count 
        FROM diffusion.selfies 
        WHERE face_detected = true 
        AND image_data IS NOT NULL;
        """
        selfies_result = db_manager.execute_query(selfies_query)
        usable_selfies = selfies_result.iloc[0]['count']
        
        # Check usable glasses
        glasses_query = """
        SELECT COUNT(*) as count 
        FROM diffusion.processed_glasses 
        WHERE has_transparency = true 
        AND image_data IS NOT NULL;
        """
        glasses_result = db_manager.execute_query(glasses_query)
        usable_glasses = glasses_result.iloc[0]['count']
        
        logger.info(f"📊 Demo readiness:")
        logger.info(f"   Usable selfies: {usable_selfies}")
        logger.info(f"   Usable glasses: {usable_glasses}")
        
        is_ready = usable_selfies > 0 and usable_glasses > 0
        
        if is_ready:
            logger.info("🎉 SYSTEM IS READY FOR DEMO!")
        else:
            logger.warning("⚠️ System not ready yet")
        
        return is_ready
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def main():
    """Main fix function."""
    try:
        logger.info("🔧 IMMEDIATE FIX FOR VIRTUAL GLASSES TRY-ON")
        logger.info("=" * 50)
        
        # Step 1: Test database connection
        logger.info("\n--- STEP 1: DATABASE CONNECTION TEST ---")
        if not test_database_connection():
            return 1
        
        # Step 2: Setup database schema
        logger.info("\n--- STEP 2: DATABASE SCHEMA SETUP ---")
        if not setup_database_schema():
            return 1
        
        # Step 3: Create sample selfie data
        logger.info("\n--- STEP 3: SAMPLE SELFIE DATA ---")
        selfies_success = create_sample_selfie_data()
        
        # Step 4: Create sample glasses data
        logger.info("\n--- STEP 4: SAMPLE GLASSES DATA ---")
        glasses_success = create_sample_glasses_data()
        
        # Step 5: Verify readiness
        logger.info("\n--- STEP 5: DEMO READINESS CHECK ---")
        demo_ready = verify_demo_readiness()
        
        # Summary
        logger.info("\n" + "=" * 50)
        if demo_ready:
            logger.info("🎉 IMMEDIATE FIX SUCCESSFUL!")
            logger.info("🚀 System is ready for demo")
            logger.info("📋 Next steps:")
            logger.info("   1. Run: python cli.py demo")
            logger.info("   2. Or: python cli.py pipeline --skip-download --skip-preprocessing")
        else:
            logger.info("⚠️ PARTIAL SUCCESS")
            logger.info("🔄 You may need to run the pipeline again")
            
        return 0 if demo_ready else 1
        
    except Exception as e:
        logger.error(f"❌ Immediate fix failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())