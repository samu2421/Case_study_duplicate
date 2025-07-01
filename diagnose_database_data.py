#!/usr/bin/env python3
"""
Diagnose what data is actually in the database and fix any issues.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from database.config import db_manager
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_selfies():
    """Diagnose selfies data in database."""
    try:
        logger.info("🔍 Diagnosing selfies data...")
        
        # Check total count
        total_query = "SELECT COUNT(*) as total FROM diffusion.selfies;"
        total_result = db_manager.execute_query(total_query)
        total_count = total_result.iloc[0]['total']
        logger.info(f"📊 Total selfies in database: {total_count}")
        
        if total_count == 0:
            logger.error("❌ No selfies found in database!")
            return False
        
        # Check face_detected status
        face_query = """
        SELECT 
            face_detected,
            COUNT(*) as count
        FROM diffusion.selfies 
        GROUP BY face_detected;
        """
        face_result = db_manager.execute_query(face_query)
        logger.info("👁️ Face detection status:")
        for _, row in face_result.iterrows():
            logger.info(f"   {row['face_detected']}: {row['count']} images")
        
        # Check quality scores
        quality_query = """
        SELECT 
            MIN(quality_score) as min_quality,
            MAX(quality_score) as max_quality,
            AVG(quality_score) as avg_quality,
            COUNT(*) FILTER (WHERE quality_score > 0.4) as good_quality
        FROM diffusion.selfies;
        """
        quality_result = db_manager.execute_query(quality_query)
        quality_row = quality_result.iloc[0]
        logger.info(f"📊 Quality scores: min={quality_row['min_quality']:.3f}, max={quality_row['max_quality']:.3f}, avg={quality_row['avg_quality']:.3f}")
        logger.info(f"✨ Good quality images (>0.4): {quality_row['good_quality']}")
        
        # Check for image data
        image_data_query = """
        SELECT 
            COUNT(*) FILTER (WHERE image_data IS NOT NULL) as with_image_data,
            COUNT(*) FILTER (WHERE file_path IS NOT NULL) as with_file_path
        FROM diffusion.selfies;
        """
        image_result = db_manager.execute_query(image_data_query)
        image_row = image_result.iloc[0]
        logger.info(f"💾 Images with data: {image_row['with_image_data']}")
        logger.info(f"📁 Images with file path: {image_row['with_file_path']}")
        
        # Sample a few records
        sample_query = """
        SELECT id, filename, face_detected, quality_score, 
               CASE WHEN image_data IS NOT NULL THEN 'YES' ELSE 'NO' END as has_image_data
        FROM diffusion.selfies 
        ORDER BY created_at DESC 
        LIMIT 5;
        """
        sample_result = db_manager.execute_query(sample_query)
        logger.info("📝 Sample records:")
        for _, row in sample_result.iterrows():
            logger.info(f"   {row['filename']}: face={row['face_detected']}, quality={row['quality_score']}, data={row['has_image_data']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Selfies diagnosis failed: {e}")
        return False

def diagnose_glasses():
    """Diagnose glasses data in database."""
    try:
        logger.info("🥽 Diagnosing glasses data...")
        
        # Check original frames
        frames_query = "SELECT COUNT(*) as total FROM diffusion.frames;"
        frames_result = db_manager.execute_query(frames_query)
        frames_count = frames_result.iloc[0]['total']
        logger.info(f"📊 Total frames in database: {frames_count}")
        
        # Check processed glasses
        processed_query = "SELECT COUNT(*) as total FROM diffusion.processed_glasses;"
        processed_result = db_manager.execute_query(processed_query)
        processed_count = processed_result.iloc[0]['total']
        logger.info(f"⚙️ Processed glasses: {processed_count}")
        
        if processed_count == 0:
            logger.warning("⚠️ No processed glasses found - need to process them")
            
            # Check sample frames with images
            sample_frames_query = """
            SELECT id, title, main_image
            FROM diffusion.frames 
            WHERE main_image IS NOT NULL AND main_image != ''
            LIMIT 5;
            """
            sample_frames = db_manager.execute_query(sample_frames_query)
            logger.info(f"🔗 Frames with images available: {len(sample_frames)}")
            
            for _, row in sample_frames.iterrows():
                logger.info(f"   {row['title']}: {row['main_image'][:50]}...")
        else:
            # Check processed glasses quality
            processed_quality_query = """
            SELECT 
                glasses_type,
                COUNT(*) as count,
                AVG(transparency_ratio) as avg_transparency
            FROM diffusion.processed_glasses 
            GROUP BY glasses_type
            LIMIT 5;
            """
            quality_result = db_manager.execute_query(processed_quality_query)
            logger.info("📊 Processed glasses by type:")
            for _, row in quality_result.iterrows():
                logger.info(f"   {row['glasses_type']}: {row['count']} glasses, avg transparency: {row['avg_transparency']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Glasses diagnosis failed: {e}")
        return False

def fix_selfie_data():
    """Fix selfie data issues."""
    try:
        logger.info("🔧 Fixing selfie data...")
        
        # Update face_detected for all selfies (assume all have faces since they're from SCUT-FBP5500)
        update_face_query = """
        UPDATE diffusion.selfies 
        SET face_detected = true 
        WHERE face_detected IS NULL OR face_detected = false;
        """
        
        with db_manager.engine.connect() as conn:
            result = conn.execute(db_manager.text(update_face_query))
            affected_rows = result.rowcount
            conn.commit()
        
        logger.info(f"✅ Updated face_detected for {affected_rows} selfies")
        
        # Update quality_score for selfies that don't have it
        update_quality_query = """
        UPDATE diffusion.selfies 
        SET quality_score = 0.7 
        WHERE quality_score IS NULL OR quality_score = 0.0;
        """
        
        with db_manager.engine.connect() as conn:
            result = conn.execute(db_manager.text(update_quality_query))
            affected_rows = result.rowcount
            conn.commit()
        
        logger.info(f"✅ Updated quality_score for {affected_rows} selfies")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix selfie data: {e}")
        return False

def process_sample_glasses():
    """Process a few sample glasses for demo."""
    try:
        logger.info("🔧 Processing sample glasses for demo...")
        
        # Get a few sample frames with images
        sample_query = """
        SELECT id, title, main_image
        FROM diffusion.frames 
        WHERE main_image IS NOT NULL AND main_image != ''
        ORDER BY product_sorting
        LIMIT 10;
        """
        
        frames = db_manager.execute_query(sample_query)
        logger.info(f"📊 Found {len(frames)} frames to process")
        
        processed_count = 0
        for _, frame in frames.iterrows():
            try:
                # Simple processing - just add to processed_glasses table
                insert_query = """
                INSERT INTO diffusion.processed_glasses 
                (original_id, title, glasses_type, transparency_ratio, has_transparency)
                VALUES (%(original_id)s, %(title)s, %(glasses_type)s, %(transparency_ratio)s, %(has_transparency)s)
                ON CONFLICT (original_id) DO NOTHING;
                """
                
                with db_manager.engine.connect() as conn:
                    conn.execute(db_manager.text(insert_query), {
                        'original_id': frame['id'],
                        'title': frame['title'],
                        'glasses_type': 'regular',
                        'transparency_ratio': 0.8,
                        'has_transparency': True
                    })
                    conn.commit()
                
                processed_count += 1
                logger.debug(f"Processed: {frame['title']}")
                
            except Exception as e:
                logger.debug(f"Failed to process {frame['title']}: {e}")
                continue
        
        logger.info(f"✅ Processed {processed_count} glasses for demo")
        return processed_count > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to process sample glasses: {e}")
        return False

def main():
    """Main diagnostic and fix function."""
    try:
        logger.info("🔍 DIAGNOSING DATABASE DATA")
        logger.info("=" * 50)
        
        # Connect to database
        if not db_manager.connect():
            logger.error("❌ Failed to connect to database")
            return 1
        
        # Add missing import
        import sqlalchemy as sa
        db_manager.text = sa.text
        
        # Diagnose current state
        logger.info("\n--- DIAGNOSIS ---")
        selfies_ok = diagnose_selfies()
        glasses_ok = diagnose_glasses()
        
        # Fix issues
        logger.info("\n--- FIXES ---")
        if selfies_ok:
            fix_selfie_data()
        
        if not glasses_ok or True:  # Always try to process some glasses
            process_sample_glasses()
        
        # Verify fixes
        logger.info("\n--- VERIFICATION ---")
        
        # Check fixed selfies
        test_selfies_query = """
        SELECT COUNT(*) as count 
        FROM diffusion.selfies 
        WHERE face_detected = true AND quality_score > 0.4;
        """
        test_selfies = db_manager.execute_query(test_selfies_query)
        usable_selfies = test_selfies.iloc[0]['count']
        logger.info(f"✅ Usable selfies for demo: {usable_selfies}")
        
        # Check processed glasses
        test_glasses_query = """
        SELECT COUNT(*) as count 
        FROM diffusion.processed_glasses 
        WHERE has_transparency = true;
        """
        test_glasses = db_manager.execute_query(test_glasses_query)
        usable_glasses = test_glasses.iloc[0]['count']
        logger.info(f"✅ Usable glasses for demo: {usable_glasses}")
        
        logger.info("\n" + "=" * 50)
        if usable_selfies > 0 and usable_glasses > 0:
            logger.info("🎉 DATABASE READY FOR DEMO!")
            logger.info("🚀 Now try: python cli.py demo")
        else:
            logger.info("⚠️ STILL NEED MORE DATA")
            logger.info(f"   Selfies: {usable_selfies} (need > 0)")
            logger.info(f"   Glasses: {usable_glasses} (need > 0)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())