"""
Selfie preprocessing pipeline for the Virtual Glasses Try-On system.
Handles resizing, normalization, face detection, and quality assessment.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import pandas as pd

from database.config import db_manager
from image_processing.utils.image_utils import image_processor

logger = logging.getLogger(__name__)

class SelfiePreprocessor:
    """Handles preprocessing of selfie images for training and inference."""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512), quality_threshold: float = 0.3):
        """Initialize selfie preprocessor."""
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        self.processed_count = 0
        
    def preprocess_single_selfie(self, image_data: bytes, metadata: Dict) -> Optional[Dict]:
        """Preprocess a single selfie image."""
        try:
            # Load image from bytes
            image = image_processor.load_image(image_data)
            if image is None:
                logger.warning("Failed to load image from bytes")
                return None
            
            # Detect face
            face_bbox = image_processor.detect_face_region(image)
            if face_bbox is None:
                logger.debug("No face detected in image")
                # Still process the image but mark as no face detected
                face_detected = False
                processed_image = image
            else:
                face_detected = True
                # Crop face region with padding
                processed_image = image_processor.crop_face_region(image, face_bbox, expand_ratio=0.4)
            
            # Resize to target size
            resized_image = image_processor.resize_image(processed_image, self.target_size)
            
            # Enhance image quality
            enhanced_image = image_processor.enhance_image_quality(resized_image)
            
            # Calculate quality score
            quality_score = image_processor.calculate_image_quality_score(enhanced_image)
            
            # Normalize image
            normalized_image = image_processor.normalize_image(enhanced_image)
            
            return {
                'processed_image': enhanced_image,
                'normalized_image': normalized_image,
                'face_detected': face_detected,
                'face_bbox': face_bbox,
                'quality_score': quality_score,
                'original_size': image.shape[:2],
                'processed_size': enhanced_image.shape[:2]
            }
            
        except Exception as e:
            logger.error(f"Failed to preprocess selfie: {e}")
            return None
    
    def preprocess_batch_from_database(self, batch_size: int = 32, quality_filter: bool = True) -> Dict[str, int]:
        """Preprocess a batch of selfies from the database."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Get unprocessed selfies or all selfies for reprocessing
            offset = 0
            total_processed = 0
            total_failed = 0
            
            while True:
                # Get batch of selfies
                selfies_batch = db_manager.get_selfies_batch(limit=batch_size, offset=offset)
                
                if len(selfies_batch) == 0:
                    break
                
                logger.info(f"Processing batch {offset//batch_size + 1}, images {offset}-{offset + len(selfies_batch)}")
                
                for _, selfie_row in tqdm(selfies_batch.iterrows(), desc="Processing selfies", total=len(selfies_batch)):
                    try:
                        # Get image data
                        image_data = db_manager.get_selfie_image_data(selfie_row['id'])
                        if image_data is None:
                            logger.warning(f"No image data for selfie {selfie_row['id']}")
                            total_failed += 1
                            continue
                        
                        # Preprocess image
                        result = self.preprocess_single_selfie(image_data, selfie_row.to_dict())
                        if result is None:
                            total_failed += 1
                            continue
                        
                        # Apply quality filter
                        if quality_filter and result['quality_score'] < self.quality_threshold:
                            logger.debug(f"Skipping low quality image: {result['quality_score']}")
                            total_failed += 1
                            continue
                        
                        # Update metadata in database
                        update_metadata = {
                            'face_detected': result['face_detected'],
                            'face_landmarks': str(result['face_bbox']) if result['face_bbox'] else None,
                            'quality_score': result['quality_score']
                        }
                        
                        db_manager.update_selfie_metadata(selfie_row['id'], update_metadata)
                        total_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process selfie {selfie_row['id']}: {e}")
                        total_failed += 1
                
                offset += batch_size
            
            results = {
                'total_processed': total_processed,
                'total_failed': total_failed,
                'success_rate': total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
            }
            
            logger.info(f"Batch preprocessing completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}")
            return {'total_processed': 0, 'total_failed': 0, 'success_rate': 0}
    
    def get_training_data(self, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15), 
                         min_quality: float = 0.4) -> Dict[str, pd.DataFrame]:
        """Get training, validation, and test data splits from database."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Get high-quality selfies with faces
            query = f"""
            SELECT id, filename, face_detected, quality_score, age_group, gender
            FROM {db_manager.config['schema']}.selfies
            WHERE face_detected = true 
            AND quality_score >= {min_quality}
            ORDER BY quality_score DESC;
            """
            
            selfies_df = db_manager.execute_query(query)
            
            if len(selfies_df) == 0:
                logger.error("No suitable selfies found for training")
                return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
            
            # Shuffle the data
            selfies_df = selfies_df.sample(frac=1).reset_index(drop=True)
            
            # Calculate split indices
            total_samples = len(selfies_df)
            train_idx = int(total_samples * split_ratio[0])
            val_idx = int(total_samples * (split_ratio[0] + split_ratio[1]))
            
            # Create splits
            train_df = selfies_df[:train_idx]
            val_df = selfies_df[train_idx:val_idx]
            test_df = selfies_df[val_idx:]
            
            splits = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
            
            logger.info(f"Data splits created - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return splits
            
        except Exception as e:
            logger.error(f"Failed to create training data splits: {e}")
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
    
    def export_processed_images(self, output_dir: Path, split: str = 'train', 
                               limit: Optional[int] = None) -> bool:
        """Export processed images to directory for external use."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get data split
            splits = self.get_training_data()
            if split not in splits or len(splits[split]) == 0:
                logger.error(f"No data available for split: {split}")
                return False
            
            split_df = splits[split]
            if limit:
                split_df = split_df.head(limit)
            
            export_count = 0
            for _, row in tqdm(split_df.iterrows(), desc=f"Exporting {split} images", total=len(split_df)):
                try:
                    # Get image data
                    image_data = db_manager.get_selfie_image_data(row['id'])
                    if image_data is None:
                        continue
                    
                    # Load and process image
                    image = image_processor.load_image(image_data)
                    if image is None:
                        continue
                    
                    # Save to output directory
                    output_path = output_dir / f"{row['id']}_{row['filename']}"
                    if image_processor.save_image(image, output_path):
                        export_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to export image {row['id']}: {e}")
            
            logger.info(f"Exported {export_count} images to {output_dir}")
            return export_count > 0
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def analyze_dataset_quality(self) -> Dict[str, any]:
        """Analyze the quality and characteristics of the selfie dataset."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Get comprehensive statistics
            analysis_query = f"""
            SELECT 
                COUNT(*) as total_images,
                COUNT(*) FILTER (WHERE face_detected = true) as images_with_faces,
                COUNT(*) FILTER (WHERE quality_score >= 0.5) as high_quality_images,
                COUNT(*) FILTER (WHERE quality_score >= 0.3) as medium_quality_images,
                AVG(quality_score) as avg_quality_score,
                MIN(quality_score) as min_quality_score,
                MAX(quality_score) as max_quality_score,
                COUNT(DISTINCT gender) as unique_genders,
                COUNT(DISTINCT age_group) as unique_age_groups
            FROM {db_manager.config['schema']}.selfies;
            """
            
            stats = db_manager.execute_query(analysis_query)
            
            # Get gender distribution
            gender_query = f"""
            SELECT gender, COUNT(*) as count
            FROM {db_manager.config['schema']}.selfies
            WHERE gender IS NOT NULL
            GROUP BY gender;
            """
            
            gender_dist = db_manager.execute_query(gender_query)
            
            # Get age group distribution
            age_query = f"""
            SELECT age_group, COUNT(*) as count
            FROM {db_manager.config['schema']}.selfies
            WHERE age_group IS NOT NULL
            GROUP BY age_group;
            """
            
            age_dist = db_manager.execute_query(age_query)
            
            analysis_result = {
                'overall_statistics': stats.to_dict('records')[0] if len(stats) > 0 else {},
                'gender_distribution': gender_dist.to_dict('records') if len(gender_dist) > 0 else [],
                'age_distribution': age_dist.to_dict('records') if len(age_dist) > 0 else [],
                'recommendations': []
            }
            
            # Add recommendations based on analysis
            if len(stats) > 0:
                stats_row = stats.iloc[0]
                face_detection_rate = stats_row['images_with_faces'] / stats_row['total_images']
                
                if face_detection_rate < 0.8:
                    analysis_result['recommendations'].append(
                        f"Low face detection rate ({face_detection_rate:.2%}). Consider improving face detection or dataset quality."
                    )
                
                if stats_row['avg_quality_score'] < 0.5:
                    analysis_result['recommendations'].append(
                        f"Low average quality score ({stats_row['avg_quality_score']:.2f}). Consider filtering or enhancing images."
                    )
            
            logger.info(f"Dataset analysis completed: {analysis_result['overall_statistics']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            return {}

# Global selfie preprocessor instance
selfie_preprocessor = SelfiePreprocessor()