"""
Glasses preprocessing pipeline for the Virtual Glasses Try-On system.
Handles background removal, transparency, and format conversion for glasses images.
"""

import cv2
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import pandas as pd
from PIL import Image
import io

from database.config import db_manager
from image_processing.utils.image_utils import image_processor
import sqlalchemy as sa

logger = logging.getLogger(__name__)

class GlassesPreprocessor:
    """Handles preprocessing of glasses images for virtual try-on."""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """Initialize glasses preprocessor."""
        self.target_size = target_size
        self.processed_count = 0
        
    def download_glasses_image(self, url: str) -> Optional[np.ndarray]:
        """Download glasses image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert to numpy array
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            
            if image is not None:
                # Convert BGR to RGB if needed
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to download glasses image from {url}: {e}")
            return None
    
    def detect_glasses_type(self, image: np.ndarray) -> str:
        """Detect the type/style of glasses from the image."""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Simple heuristics to classify glasses type
            height, width = gray.shape
            
            # Analyze the aspect ratio
            aspect_ratio = width / height
            
            # Analyze the image structure
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) >= 2:
                # Likely regular glasses with two lenses
                if aspect_ratio > 2.0:
                    return "wide_frame"
                elif aspect_ratio > 1.5:
                    return "regular"
                else:
                    return "round"
            elif len(contours) == 1:
                # Might be sunglasses or single piece
                return "sunglasses"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Failed to detect glasses type: {e}")
            return "unknown"
    
    def remove_background_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced background removal for glasses images."""
        try:
            if len(image.shape) == 2:
                # Grayscale image, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            if image.shape[2] == 3:
                # Add alpha channel
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            
            # Method 1: Color-based segmentation
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
            
            # Define range for white/light backgrounds
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Define range for black backgrounds
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv, lower_black, upper_black)
            
            # Combine masks
            background_mask = cv2.bitwise_or(white_mask, black_mask)
            
            # Method 2: Edge-based refinement
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # Dilate edges to create connected regions
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours to identify glasses regions
            contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask for glasses regions
            glasses_mask = np.zeros(gray.shape, dtype=np.uint8)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    cv2.fillPoly(glasses_mask, [contour], 255)
            
            # Combine with color-based mask
            final_mask = cv2.bitwise_and(cv2.bitwise_not(background_mask), glasses_mask)
            
            # Clean up the mask
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to alpha channel
            image[:, :, 3] = final_mask
            
            return image
            
        except Exception as e:
            logger.error(f"Advanced background removal failed: {e}")
            return image
    
    def preprocess_single_glasses(self, image_url: str, metadata: Dict) -> Optional[Dict]:
        """Preprocess a single glasses image."""
        try:
            # Download image
            image = self.download_glasses_image(image_url)
            if image is None:
                logger.warning(f"Failed to download image from {image_url}")
                return None
            
            # Detect glasses type
            glasses_type = self.detect_glasses_type(image)
            
            # Remove background and ensure transparency
            processed_image = self.remove_background_advanced(image)
            
            # Resize to target size
            resized_image = image_processor.resize_image(processed_image, self.target_size)
            
            # Enhance image quality while preserving transparency
            if resized_image.shape[2] == 4:
                # Separate RGB and alpha channels
                rgb_channels = resized_image[:, :, :3]
                alpha_channel = resized_image[:, :, 3]
                
                # Enhance RGB channels
                enhanced_rgb = image_processor.enhance_image_quality(rgb_channels)
                
                # Recombine with alpha
                enhanced_image = np.dstack([enhanced_rgb, alpha_channel])
            else:
                enhanced_image = image_processor.enhance_image_quality(resized_image)
            
            # Calculate transparency ratio
            if enhanced_image.shape[2] == 4:
                transparency_ratio = np.sum(enhanced_image[:, :, 3] > 0) / (enhanced_image.shape[0] * enhanced_image.shape[1])
            else:
                transparency_ratio = 1.0
            
            return {
                'processed_image': enhanced_image,
                'glasses_type': glasses_type,
                'transparency_ratio': transparency_ratio,
                'original_size': image.shape[:2],
                'processed_size': enhanced_image.shape[:2],
                'has_transparency': enhanced_image.shape[2] == 4,
                'url': image_url
            }
            
        except Exception as e:
            logger.error(f"Failed to preprocess glasses image: {e}")
            return None
    
    def preprocess_glasses_from_database(self, batch_size: int = 32, limit: Optional[int] = None) -> Dict[str, int]:
        """Preprocess glasses images from the database."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Get glasses data
            offset = 0
            total_processed = 0
            total_failed = 0
            processed_glasses = []
            
            while True:
                # Get batch of glasses
                glasses_batch = db_manager.get_glasses_batch(limit=batch_size, offset=offset)
                
                if len(glasses_batch) == 0:
                    break
                
                if limit and total_processed >= limit:
                    break
                
                logger.info(f"Processing glasses batch {offset//batch_size + 1}, items {offset}-{offset + len(glasses_batch)}")
                
                for _, glasses_row in tqdm(glasses_batch.iterrows(), desc="Processing glasses", total=len(glasses_batch)):
                    try:
                        if limit and total_processed >= limit:
                            break
                        
                        # Skip if no main_image URL
                        if pd.isna(glasses_row['main_image']) or not glasses_row['main_image']:
                            total_failed += 1
                            continue
                        
                        # Preprocess glasses image
                        result = self.preprocess_single_glasses(
                            glasses_row['main_image'], 
                            glasses_row.to_dict()
                        )
                        
                        if result is None:
                            total_failed += 1
                            continue
                        
                        # Store results
                        result['id'] = glasses_row['id']
                        result['title'] = glasses_row['title']
                        processed_glasses.append(result)
                        total_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process glasses {glasses_row['id']}: {e}")
                        total_failed += 1
                
                offset += batch_size
            
            # Save processed glasses metadata
            self._save_processed_glasses_metadata(processed_glasses)
            
            results = {
                'total_processed': total_processed,
                'total_failed': total_failed,
                'success_rate': total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
            }
            
            logger.info(f"Glasses preprocessing completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Glasses preprocessing failed: {e}")
            # Try to reset database connection if there are transaction issues
            if "transaction" in str(e).lower() or "rollback" in str(e).lower():
                logger.info("Attempting to reset database connection...")
                db_manager.reset_connection()
            return {'total_processed': 0, 'total_failed': 0, 'success_rate': 0}
    
    def _save_processed_glasses_metadata(self, processed_glasses: List[Dict]):
    """Save processed glasses metadata to database."""
    try:
        # Create processed glasses table if it doesn't exist
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
        
        with db_manager.engine.connect() as conn:
            try:
                conn.execute(sa.text(create_table_sql))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.warning(f"Table creation warning: {e}")
        
        # Insert processed glasses data
        success_count = 0
        for glasses_data in processed_glasses:
            try:
                # Convert image to bytes
                image_bytes = image_processor.image_to_bytes(glasses_data['processed_image'], 'PNG')
                
                # Use regular string (NOT f-string) - THIS IS THE KEY FIX
                insert_sql = """
                INSERT INTO diffusion.processed_glasses 
                (original_id, title, glasses_type, transparency_ratio, has_transparency,
                 processed_width, processed_height, image_data)
                VALUES (%(original_id)s, %(title)s, %(glasses_type)s, %(transparency_ratio)s,
                        %(has_transparency)s, %(width)s, %(height)s, %(image_data)s)
                ON CONFLICT (original_id) DO UPDATE SET
                    glasses_type = EXCLUDED.glasses_type,
                    transparency_ratio = EXCLUDED.transparency_ratio,
                    has_transparency = EXCLUDED.has_transparency,
                    processed_width = EXCLUDED.processed_width,
                    processed_height = EXCLUDED.processed_height,
                    image_data = EXCLUDED.image_data;
                """
                
                with db_manager.engine.connect() as conn:
                    try:
                        conn.execute(sa.text(insert_sql), {
                            'original_id': glasses_data['id'],
                            'title': glasses_data['title'],
                            'glasses_type': glasses_data['glasses_type'],
                            'transparency_ratio': glasses_data['transparency_ratio'],
                            'has_transparency': glasses_data['has_transparency'],
                            'width': glasses_data['processed_size'][1],
                            'height': glasses_data['processed_size'][0],
                            'image_data': image_bytes
                        })
                        conn.commit()
                        success_count += 1
                    except Exception as e:
                        conn.rollback()
                        logger.warning(f"Failed to save processed glasses {glasses_data['id']}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to process glasses data {glasses_data.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Saved {success_count} processed glasses to database")
        
    except Exception as e:
        logger.error(f"Failed to save processed glasses metadata: {e}")
    
    def get_glasses_for_training(self, glasses_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get processed glasses data for training."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Build query
            # Build query (use regular string)
            base_query = """
            SELECT pg.id, pg.original_id, pg.title, pg.glasses_type, 
                   pg.transparency_ratio, pg.has_transparency
            FROM diffusion.processed_glasses pg
            WHERE pg.has_transparency = true
            """
            
            if glasses_types:
                glasses_types_str = "', '".join(glasses_types)
                base_query += f" AND pg.glasses_type IN ('{glasses_types_str}')"
            
            base_query += " ORDER BY pg.transparency_ratio DESC;"
            
            result = db_manager.execute_query(base_query)
            logger.info(f"Retrieved {len(result)} glasses for training")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get glasses for training: {e}")
            return pd.DataFrame()
    
    def export_processed_glasses(self, output_dir: Path, limit: Optional[int] = None) -> bool:
        """Export processed glasses images to directory."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get processed glasses
            glasses_df = self.get_glasses_for_training()
            if len(glasses_df) == 0:
                logger.error("No processed glasses found for export")
                return False
            
            if limit:
                glasses_df = glasses_df.head(limit)
            
            export_count = 0
            for _, row in tqdm(glasses_df.iterrows(), desc="Exporting glasses", total=len(glasses_df)):
                try:
                    # Get processed image data
                    # Get processed image data (use regular string)
                    query = """
                    SELECT image_data FROM diffusion.processed_glasses 
                    WHERE id = %(glasses_id)s;
                    """
                    
                    result = db_manager.execute_query(query, {'glasses_id': row['id']})
                    if len(result) == 0:
                        continue
                    
                    image_data = result.iloc[0]['image_data']
                    if image_data is None:
                        continue
                    
                    # Save image
                    filename = f"{row['id']}_{row['glasses_type']}.png"
                    output_path = output_dir / filename
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    
                    export_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to export glasses {row['id']}: {e}")
            
            logger.info(f"Exported {export_count} glasses images to {output_dir}")
            return export_count > 0
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def analyze_glasses_dataset(self) -> Dict[str, any]:
        """Analyze the processed glasses dataset."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Get comprehensive statistics
            # Get comprehensive statistics (use regular string)
            analysis_query = """
            SELECT 
                COUNT(*) as total_glasses,
                COUNT(*) FILTER (WHERE has_transparency = true) as with_transparency,
                AVG(transparency_ratio) as avg_transparency_ratio,
                COUNT(DISTINCT glasses_type) as unique_types
            FROM diffusion.processed_glasses;
            """
            
            stats = db_manager.execute_query(analysis_query)
            
            # Get type distribution
            # Get type distribution (use regular string)
            type_query = """
            SELECT glasses_type, COUNT(*) as count
            FROM diffusion.processed_glasses
            GROUP BY glasses_type
            ORDER BY count DESC;
            """
            
            type_dist = db_manager.execute_query(type_query)
            
            analysis_result = {
                'overall_statistics': stats.to_dict('records')[0] if len(stats) > 0 else {},
                'type_distribution': type_dist.to_dict('records') if len(type_dist) > 0 else [],
                'recommendations': []
            }
            
            # Add recommendations
            if len(stats) > 0:
                stats_row = stats.iloc[0]
                if stats_row['total_glasses'] > 0:
                    transparency_rate = stats_row['with_transparency'] / stats_row['total_glasses']
                    
                    if transparency_rate < 0.7:
                        analysis_result['recommendations'].append(
                            f"Low transparency rate ({transparency_rate:.2%}). Consider improving background removal."
                        )
                    
                    if stats_row['avg_transparency_ratio'] and stats_row['avg_transparency_ratio'] < 0.3:
                        analysis_result['recommendations'].append(
                            f"Low average transparency ratio ({stats_row['avg_transparency_ratio']:.2f}). Check background removal quality."
                        )
            
            logger.info(f"Glasses dataset analysis completed: {analysis_result['overall_statistics']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Glasses dataset analysis failed: {e}")
            return {}

# Global glasses preprocessor instance
glasses_preprocessor = GlassesPreprocessor()