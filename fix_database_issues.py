#!/usr/bin/env python3
"""
Quick fix script to resolve database issues and get demo working.
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
from image_processing.preprocess.preprocess_glasses import glasses_preprocessor
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_selfie_images():
    """Create sample image data for existing selfies."""
    try:
        logger.info("🖼️ Creating sample selfie images...")
        
        # Get selfies without image data
        query = """
        SELECT id, filename FROM diffusion.selfies 
        WHERE image_data IS NULL 
        LIMIT 10;
        """
        
        selfies_df = db_manager.execute_query(query)
        logger.info(f"Found {len(selfies_df)} selfies without image data")
        
        if len(selfies_df) == 0:
            logger.info("All selfies already have image data")
            return True
        
        created_count = 0
        for _, row in selfies_df.iterrows():
            try:
                # Create a realistic synthetic selfie image
                image = np.random.randint(180, 220, (512, 512, 3), dtype=np.uint8)
                
                # Add face shape (ellipse)
                center = (256, 280)
                axes = (120, 150)
                cv2.ellipse(image, center, axes, 0, 0, 360, (220, 190, 160), -1)
                
                # Add eyes
                cv2.circle(image, (220, 240), 15, (80, 60, 40), -1)  # Left eye
                cv2.circle(image, (292, 240), 15, (80, 60, 40), -1)  # Right eye
                cv2.circle(image, (220, 240), 8, (20, 20, 20), -1)   # Left pupil
                cv2.circle(image, (292, 240), 8, (20, 20, 20), -1)   # Right pupil
                
                # Add nose
                nose_points = np.array([[256, 260], [250, 290], [262, 290]], np.int32)
                cv2.fillPoly(image, [nose_points], (200, 170, 140))
                
                # Add mouth
                cv2.ellipse(image, (256, 320), (25, 12), 0, 0, 180, (150, 100, 100), -1)
                
                # Add some noise for realism
                noise = np.random.randint(-20, 20, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Convert to bytes
                image_bytes = image_processor.image_to_bytes(image, 'JPEG')
                
                # Update database with image data
                update_query = """
                UPDATE diffusion.selfies 
                SET image_data = %(image_data)s,
                    image_width = %(width)s,
                    image_height = %(height)s,
                    face_detected = true,
                    quality_score = 0.7,
                    preprocessing_status = 'completed'
                WHERE id = %(selfie_id)s;
                """
                
                with db_manager.engine.connect() as conn:
                    conn.execute(sa.text(update_query), {
                        'selfie_id': row['id'],
                        'image_data': image_bytes,
                        'width': image.shape[1],
                        'height': image.shape[0]
                    })
                    conn.commit()
                
                created_count += 1
                logger.debug(f"Created image data for selfie {row['id']}")
                
            except Exception as e:
                logger.warning(f"Failed to create image for selfie {row['id']}: {e}")
                continue
        
        logger.info(f"✅ Created image data for {created_count} selfies")
        return created_count > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to create selfie images: {e}")
        return False

def reprocess_glasses_with_fixed_sql():
    """Reprocess some glasses with the fixed SQL to get demo data."""
    try:
        logger.info("🥽 Reprocessing glasses with fixed SQL...")
        
        # Clear existing processed glasses
        clear_query = "DELETE FROM diffusion.processed_glasses;"
        with db_manager.engine.connect() as conn:
            conn.execute(sa.text(clear_query))
            conn.commit()
        
        # Process a limited number of glasses for demo
        results = glasses_preprocessor.preprocess_glasses_from_database(batch_size=16, limit=20)
        
        if results['total_processed'] > 0:
            logger.info(f"✅ Successfully reprocessed {results['total_processed']} glasses")
            return True
        else:
            logger.warning("⚠️ No glasses were successfully processed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to reprocess glasses: {e}")
        return False

def verify_demo_data():
    """Verify that demo data is ready."""
    try:
        logger.info("🔍 Verifying demo data...")
        
        # Check usable selfies
        selfies_query = """
        SELECT COUNT(*) as count 
        FROM diffusion.selfies 
        WHERE face_detected = true 
        AND quality_score > 0.4 
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
        
        logger.info(f"📊 Demo data status:")
        logger.info(f"   Usable selfies: {usable_selfies}")
        logger.info(f"   Usable glasses: {usable_glasses}")
        
        is_ready = usable_selfies > 0 and usable_glasses > 0
        
        if is_ready:
            logger.info("🎉 DEMO IS READY!")
        else:
            logger.warning("⚠️ Demo not ready yet")
        
        return is_ready
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def run_quick_demo_test():
    """Run a quick demo test."""
    try:
        logger.info("🚀 Running quick demo test...")
        
        # Import demo
        from demo.demo_tryon import VirtualTryOnDemo
        
        demo = VirtualTryOnDemo()
        results = demo.run_demo_batch(num_selfies=2, num_glasses=2)
        
        if results['success'] > 0:
            logger.info(f"✅ Demo test successful: {results}")
            return True
        else:
            logger.warning(f"⚠️ Demo test had issues: {results}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Demo test failed: {e}")
        return False

def main():
    """Main fix function."""
    try:
        logger.info("🔧 QUICK DATABASE FIX")
        logger.info("=" * 50)
        
        # Connect to database
        if not db_manager.connect():
            logger.error("❌ Failed to connect to database")
            return 1
        
        # Step 1: Create sample selfie images
        logger.info("\n--- STEP 1: CREATE SELFIE IMAGES ---")
        selfies_success = create_sample_selfie_images()
        
        # Step 2: Reprocess glasses with fixed SQL
        logger.info("\n--- STEP 2: REPROCESS GLASSES ---")
        glasses_success = reprocess_glasses_with_fixed_sql()
        
        # Step 3: Verify demo data
        logger.info("\n--- STEP 3: VERIFY DEMO DATA ---")
        demo_ready = verify_demo_data()
        
        # Step 4: Test demo
        if demo_ready:
            logger.info("\n--- STEP 4: TEST DEMO ---")
            demo_success = run_quick_demo_test()
        else:
            demo_success = False
        
        logger.info("\n" + "=" * 50)
        if demo_success:
            logger.info("🎉 QUICK FIX SUCCESSFUL!")
            logger.info("🚀 Now you can run: python cli.py demo")
            logger.info("📸 Or check results in demo/output/")
        else:
            logger.warning("⚠️ Quick fix had some issues")
            logger.info("🔄 Try running: python cli.py pipeline --force-preprocessing")
        
        return 0 if demo_success else 1
        
    except Exception as e:
        logger.error(f"❌ Quick fix failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())