# image_processing/preprocess/preprocess_selfies.py
"""
Selfie preprocessing module for the Virtual Glasses Try-On project
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from tqdm import tqdm
import shutil

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfiePreprocessor:
    """Handles preprocessing of selfie images for the virtual try-on system"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 project_paths: Optional[ProjectPaths] = None):
        """
        Initialize the selfie preprocessor
        
        Args:
            db_manager: Database manager instance
            project_paths: Project paths instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.paths = project_paths or ProjectPaths()
        self.image_processor = ImageProcessor()
        
        # Ensure necessary directories exist
        self.paths.ensure_directories_exist()
        
        # Preprocessing parameters
        self.target_size = (512, 512)  # Standard size for processed selfies
        self.face_detection_enabled = True
        self.quality_checks_enabled = True
        
        logger.info("SelfiePreprocessor initialized")
    
    def load_selfies_from_directory(self, source_dir: Union[str, Path],
                                  recursive: bool = True) -> List[Path]:
        """
        Load selfie images from a directory
        
        Args:
            source_dir: Directory containing selfie images
            recursive: Whether to search subdirectories
            
        Returns:
            List of image file paths
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []
        
        # Supported image extensions
        image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp'}
        
        image_files = []
        for ext in image_extensions:
            if recursive:
                pattern = f"**/{ext}"
                files = list(source_dir.rglob(pattern))
            else:
                files = list(source_dir.glob(ext))
            image_files.extend(files)
        
        logger.info(f"Found {len(image_files)} image files in {source_dir}")
        return image_files
    
    def validate_selfie_quality(self, image: np.ndarray, 
                               image_path: Path) -> Dict[str, any]:
        """
        Validate the quality of a selfie image
        
        Args:
            image: Image as numpy array
            image_path: Path to the image file
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'face_detected': False,
            'face_count': 0,
            'image_score': 0.0
        }
        
        if image is None:
            validation_result['is_valid'] = False
            validation_result['issues'].append('Failed to load image')
            return validation_result
        
        height, width = image.shape[:2]
        
        # Check minimum resolution
        min_resolution = 224
        if height < min_resolution or width < min_resolution:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f'Resolution too low: {width}x{height}')
        
        # Check if image is too small in file size (might be corrupted)
        file_size_kb = image_path.stat().st_size // 1024
        if file_size_kb < 10:
            validation_result['is_valid'] = False
            validation_result['issues'].append('File size too small (possibly corrupted)')
        
        # Detect faces if enabled
        if self.face_detection_enabled:
            faces = self.image_processor.detect_faces(image)
            validation_result['face_count'] = len(faces)
            validation_result['face_detected'] = len(faces) > 0
            
            if len(faces) == 0:
                validation_result['issues'].append('No face detected')
            elif len(faces) > 1:
                validation_result['issues'].append(f'Multiple faces detected ({len(faces)})')
        
        # Calculate image quality score (simple version)
        # This is a placeholder - you can implement more sophisticated quality metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        validation_result['image_score'] = min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1
        
        # Low quality check
        if validation_result['image_score'] < 0.1:
            validation_result['issues'].append('Image appears blurry or low quality')
        
        # Overall validation
        if validation_result['issues'] and self.quality_checks_enabled:
            validation_result['is_valid'] = False
        
        return validation_result
    
    def preprocess_single_selfie(self, input_path: Path,
                                save_to_processed: bool = True) -> Dict[str, any]:
        """
        Preprocess a single selfie image
        
        Args:
            input_path: Path to the input selfie
            save_to_processed: Whether to save processed image
            
        Returns:
            Dictionary with preprocessing results
        """
        result = {
            'success': False,
            'input_path': str(input_path),
            'output_path': None,
            'validation': None,
            'database_saved': False,
            'error': None
        }
        
        try:
            # Load the image
            image = self.image_processor.load_image(input_path)
            if image is None:
                result['error'] = 'Failed to load image'
                return result
            
            # Validate image quality
            validation = self.validate_selfie_quality(image, input_path)
            result['validation'] = validation
            
            if not validation['is_valid'] and self.quality_checks_enabled:
                result['error'] = f"Quality validation failed: {', '.join(validation['issues'])}"
                return result
            
            # Preprocess the image
            processed_image = self._apply_preprocessing(image)
            
            # Prepare paths
            filename = input_path.name
            processed_filename = f"processed_{filename}"
            
            # Copy to raw directory
            raw_output_path = self.paths.get_selfie_path(filename, processed=False)
            if input_path != raw_output_path:
                shutil.copy2(input_path, raw_output_path)
            
            # Save processed image
            if save_to_processed:
                processed_output_path = self.paths.get_selfie_path(processed_filename, processed=True)
                success = self.image_processor.save_image(processed_image, processed_output_path)
                if not success:
                    result['error'] = 'Failed to save processed image'
                    return result
                result['output_path'] = str(processed_output_path)
            
            # Get image info
            image_info = self.image_processor.get_image_info(raw_output_path)
            
            # Save to database
            if image_info:
                db_success = self.db_manager.insert_selfie_record(
                    filename=filename,
                    file_path=str(raw_output_path),
                    file_size_kb=image_info['file_size_kb'],
                    image_width=image_info['width'],
                    image_height=image_info['height']
                )
                
                if db_success:
                    # Update preprocessing status
                    status = 'completed' if validation['is_valid'] else 'quality_issues'
                    self.db_manager.update_preprocessing_status(
                        filename=filename,
                        status=status,
                        face_detected=validation['face_detected']
                    )
                    result['database_saved'] = True
            
            result['success'] = True
            logger.info(f"Successfully preprocessed: {input_path.name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error preprocessing {input_path}: {e}")
        
        return result
    
    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image
        """
        # Resize to target size
        processed = self.image_processor.resize_image(image, self.target_size, maintain_aspect_ratio=True)
        
        # Apply slight enhancement (optional)
        processed = self.image_processor.enhance_image(
            processed,
            brightness=1.05,
            contrast=1.1,
            saturation=1.05
        )
        
        return processed
    
    def preprocess_batch(self, source_dir: Union[str, Path],
                        max_files: Optional[int] = None,
                        skip_existing: bool = True) -> Dict[str, any]:
        """
        Preprocess a batch of selfie images
        
        Args:
            source_dir: Directory containing selfie images
            max_files: Maximum number of files to process (None for all)
            skip_existing: Whether to skip files already in database
            
        Returns:
            Dictionary with batch processing results
        """
        source_dir = Path(source_dir)
        
        # Get list of image files
        image_files = self.load_selfies_from_directory(source_dir)
        
        if max_files:
            image_files = image_files[:max_files]
        
        # Get existing files from database if skipping
        existing_files = set()
        if skip_existing:
            try:
                existing_df = self.db_manager.get_selfies_data()
                existing_files = set(existing_df['filename'].tolist()) if not existing_df.empty else set()
            except Exception as e:
                logger.warning(f"Could not retrieve existing files: {e}")
        
        # Filter out existing files
        if skip_existing:
            image_files = [f for f in image_files if f.name not in existing_files]
            logger.info(f"Skipping {len(existing_files)} existing files")
        
        # Process files
        results = {
            'total_files': len(image_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'detailed_results': [],
            'summary': {}
        }
        
        logger.info(f"Processing {len(image_files)} selfie images...")
        
        for image_path in tqdm(image_files, desc="Processing selfies"):
            result = self.preprocess_single_selfie(image_path)
            results['detailed_results'].append(result)
            
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
                logger.warning(f"Failed to process {image_path.name}: {result['error']}")
        
        # Create summary
        results['summary'] = {
            'success_rate': results['successful'] / len(image_files) if image_files else 0,
            'total_processed': results['successful'] + results['failed'],
            'database_entries': sum(1 for r in results['detailed_results'] if r['database_saved'])
        }
        
        logger.info(f"Batch processing completed: {results['successful']}/{len(image_files)} successful")
        return results
    
    def get_preprocessing_stats(self) -> Dict[str, any]:
        """
        Get statistics about preprocessed selfies
        
        Returns:
            Dictionary with preprocessing statistics
        """
        try:
            selfies_df = self.db_manager.get_selfies_data()
            
            if selfies_df.empty:
                return {'total_selfies': 0, 'message': 'No selfies found in database'}
            
            stats = {
                'total_selfies': len(selfies_df),
                'preprocessing_status': selfies_df['preprocessing_status'].value_counts().to_dict(),
                'face_detection_stats': selfies_df['face_detected'].value_counts().to_dict(),
                'average_file_size_kb': selfies_df['file_size_kb'].mean() if 'file_size_kb' in selfies_df.columns else 0,
                'average_dimensions': {
                    'width': selfies_df['image_width'].mean() if 'image_width' in selfies_df.columns else 0,
                    'height': selfies_df['image_height'].mean() if 'image_height' in selfies_df.columns else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting preprocessing stats: {e}")
            return {'error': str(e)}


# Convenience functions
def preprocess_selfies_from_directory(source_dir: Union[str, Path],
                                    max_files: Optional[int] = None) -> Dict[str, any]:
    """
    Convenience function to preprocess selfies from a directory
    
    Args:
        source_dir: Directory containing selfie images
        max_files: Maximum number of files to process
        
    Returns:
        Processing results
    """
    preprocessor = SelfiePreprocessor()
    return preprocessor.preprocess_batch(source_dir, max_files=max_files)

# Example usage and testing
if __name__ == "__main__":
    print("Testing SelfiePreprocessor...")
    
    # Initialize preprocessor
    preprocessor = SelfiePreprocessor()
    
    # Test database connection
    if preprocessor.db_manager.test_connection():
        print(" Database connection successful")
        
        # Create selfies table if needed
        preprocessor.db_manager.create_selfies_table()
        
        # Get current stats
        stats = preprocessor.get_preprocessing_stats()
        print(f"Current preprocessing stats: {stats}")
        
        # Test with a sample directory (you would replace this with actual path)
        # sample_dir = Path("path/to/your/selfies")
        # if sample_dir.exists():
        #     results = preprocessor.preprocess_batch(sample_dir, max_files=5)
        #     print(f"Batch processing results: {results['summary']}")
        
        print(" SelfiePreprocessor test completed")
    else:
        print(" Database connection failed")