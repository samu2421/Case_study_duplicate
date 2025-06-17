# image_processing/preprocess/preprocess_glasses.py
"""
Glasses preprocessing module for the Virtual Glasses Try-On project
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image, ImageOps
import hashlib

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlassesPreprocessor:
    """Handles preprocessing of glasses images for the virtual try-on system"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 project_paths: Optional[ProjectPaths] = None):
        """
        Initialize the glasses preprocessor
        
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
        self.target_size = (512, 512)  # Standard size for processed glasses
        self.background_removal_enabled = True
        self.transparency_optimization = True
        
        logger.info("GlassesPreprocessor initialized")
    
    def download_glasses_from_database(self, limit: Optional[int] = None,
                                     skip_existing: bool = True) -> Dict[str, any]:
        """
        Download glasses images from database URLs
        
        Args:
            limit: Maximum number of glasses to download
            skip_existing: Whether to skip already downloaded images
            
        Returns:
            Dictionary with download results
        """
        # Get glasses data from database
        glasses_df = self.db_manager.get_glasses_data(limit=limit)
        
        if glasses_df.empty:
            logger.warning("No glasses data found in database")
            return {'total_glasses': 0, 'successful': 0, 'failed': 0}
        
        results = {
            'total_glasses': len(glasses_df),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'detailed_results': []
        }
        
        logger.info(f"Starting download of {len(glasses_df)} glasses...")
        
        for idx, row in tqdm(glasses_df.iterrows(), total=len(glasses_df), desc="Downloading glasses"):
            result = self._download_single_glasses(row, skip_existing)
            results['detailed_results'].append(result)
            
            if result['success']:
                results['successful'] += 1
            elif result['skipped']:
                results['skipped'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Download completed: {results['successful']} successful, {results['failed']} failed, {results['skipped']} skipped")
        return results
    
    def _download_single_glasses(self, glasses_row, skip_existing: bool = True) -> Dict[str, any]:
        """
        Download and preprocess a single glasses item
        
        Args:
            glasses_row: Row from glasses dataframe
            skip_existing: Whether to skip if file already exists
            
        Returns:
            Dictionary with download result
        """
        result = {
            'success': False,
            'skipped': False,
            'glasses_id': glasses_row.get('id', 'unknown'),
            'title': glasses_row.get('title', 'unknown'),
            'downloaded_files': [],
            'error': None
        }
        
        try:
            # Get image URLs
            main_image_url = glasses_row.get('main_image', '')
            additional_images_raw = glasses_row.get('additional_images', '')
            
            # Parse additional images
            additional_urls = []
            if additional_images_raw and isinstance(additional_images_raw, str):
                # Remove curly braces and split by comma
                additional_urls = [url.strip() for url in additional_images_raw.strip('{}').split(',') if url.strip()]
            
            # Combine all URLs
            all_urls = [main_image_url] + additional_urls
            all_urls = [url for url in all_urls if url]  # Remove empty URLs
            
            if not all_urls:
                result['error'] = 'No valid image URLs found'
                return result
            
            # Create glasses-specific directory
            glasses_id = str(glasses_row.get('id', 'unknown'))
            glasses_dir = self.paths.raw_glasses_dir / glasses_id
            glasses_dir.mkdir(exist_ok=True)
            
            # Download each image
            downloaded_count = 0
            for i, url in enumerate(all_urls):
                file_result = self._download_glasses_image(url, glasses_dir, i, skip_existing)
                if file_result['success']:
                    downloaded_count += 1
                    result['downloaded_files'].append(file_result['file_path'])
                elif file_result['skipped']:
                    result['downloaded_files'].append(file_result['file_path'])
            
            if downloaded_count > 0 or result['downloaded_files']:
                result['success'] = True
            else:
                result['error'] = 'No images could be downloaded'
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error downloading glasses {result['glasses_id']}: {e}")
        
        return result
    
    def _download_glasses_image(self, url: str, output_dir: Path, 
                              index: int, skip_existing: bool = True) -> Dict[str, any]:
        """
        Download a single glasses image
        
        Args:
            url: Image URL
            output_dir: Directory to save the image
            index: Image index for filename
            skip_existing: Whether to skip if file exists
            
        Returns:
            Dictionary with download result
        """
        result = {
            'success': False,
            'skipped': False,
            'file_path': None,
            'error': None
        }
        
        try:
            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            extension = self._get_extension_from_url(url)
            filename = f"glasses_{index}_{url_hash}.{extension}"
            file_path = output_dir / filename
            
            # Check if file already exists
            if skip_existing and file_path.exists():
                result['skipped'] = True
                result['file_path'] = str(file_path)
                return result
            
            # Download the image
            response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            # Save the image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            result['success'] = True
            result['file_path'] = str(file_path)
            
        except Exception as e:
            result['error'] = str(e)
            logger.debug(f"Failed to download {url}: {e}")
        
        return result
    
    def _get_extension_from_url(self, url: str) -> str:
        """Get file extension from URL"""
        try:
            # Try to get extension from URL
            ext = url.split('.')[-1].split('?')[0].lower()
            if ext in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
                return ext
        except:
            pass
        
        # Default to png for glasses (usually transparent)
        return 'png'
    
    def preprocess_downloaded_glasses(self, limit: Optional[int] = None) -> Dict[str, any]:
        """
        Preprocess downloaded glasses images
        
        Args:
            limit: Maximum number of glasses directories to process
            
        Returns:
            Dictionary with preprocessing results
        """
        glasses_dirs = list(self.paths.raw_glasses_dir.iterdir())
        glasses_dirs = [d for d in glasses_dirs if d.is_dir()]
        
        if limit:
            glasses_dirs = glasses_dirs[:limit]
        
        results = {
            'total_glasses': len(glasses_dirs),
            'successful': 0,
            'failed': 0,
            'detailed_results': []
        }
        
        logger.info(f"Preprocessing {len(glasses_dirs)} glasses directories...")
        
        for glasses_dir in tqdm(glasses_dirs, desc="Preprocessing glasses"):
            result = self._preprocess_single_glasses_directory(glasses_dir)
            results['detailed_results'].append(result)
            
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Preprocessing completed: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def _preprocess_single_glasses_directory(self, glasses_dir: Path) -> Dict[str, any]:
        """
        Preprocess all images in a glasses directory
        
        Args:
            glasses_dir: Directory containing glasses images
            
        Returns:
            Dictionary with preprocessing result
        """
        result = {
            'success': False,
            'glasses_id': glasses_dir.name,
            'processed_files': [],
            'main_processed_file': None,
            'error': None
        }
        
        try:
            # Get all image files in the directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
                image_files.extend(list(glasses_dir.glob(ext)))
            
            if not image_files:
                result['error'] = 'No image files found'
                return result
            
            # Create processed directory
            processed_dir = self.paths.processed_glasses_dir / glasses_dir.name
            processed_dir.mkdir(exist_ok=True)
            
            # Process each image
            processed_count = 0
            for i, image_file in enumerate(image_files):
                processed_result = self._preprocess_single_glasses_image(image_file, processed_dir, i)
                if processed_result['success']:
                    processed_count += 1
                    result['processed_files'].append(processed_result['output_path'])
                    
                    # Set first successful as main
                    if result['main_processed_file'] is None:
                        result['main_processed_file'] = processed_result['output_path']
            
            if processed_count > 0:
                result['success'] = True
            else:
                result['error'] = 'No images could be processed'
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error preprocessing glasses directory {glasses_dir}: {e}")
        
        return result
    
    def _preprocess_single_glasses_image(self, input_path: Path, 
                                       output_dir: Path, index: int) -> Dict[str, any]:
        """
        Preprocess a single glasses image
        
        Args:
            input_path: Path to input image
            output_dir: Directory for output
            index: Image index
            
        Returns:
            Dictionary with preprocessing result
        """
        result = {
            'success': False,
            'input_path': str(input_path),
            'output_path': None,
            'error': None
        }
        
        try:
            # Load image
            image = self.image_processor.load_image(input_path)
            if image is None:
                result['error'] = 'Failed to load image'
                return result
            
            # Apply preprocessing
            processed_image = self._apply_glasses_preprocessing(image)
            
            # Generate output filename
            output_filename = f"processed_glasses_{index}.png"  # Always save as PNG for transparency
            output_path = output_dir / output_filename
            
            # Save processed image
            success = self.image_processor.save_image(processed_image, output_path)
            if not success:
                result['error'] = 'Failed to save processed image'
                return result
            
            result['success'] = True
            result['output_path'] = str(output_path)
            
        except Exception as e:
            result['error'] = str(e)
            logger.debug(f"Error preprocessing glasses image {input_path}: {e}")
        
        return result
    
    def _apply_glasses_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to a glasses image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image with alpha channel
        """
        # Resize to target size
        processed = self.image_processor.resize_image(image, self.target_size, maintain_aspect_ratio=True)
        
        # Apply background removal if enabled
        if self.background_removal_enabled:
            processed = self._remove_background(processed)
        
        # Ensure image has alpha channel
        if processed.shape[2] == 3:  # BGR
            # Add alpha channel
            alpha = np.ones((processed.shape[0], processed.shape[1], 1), dtype=processed.dtype) * 255
            processed = np.concatenate([processed, alpha], axis=2)
        
        return processed
    
    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background from glasses image
        
        Args:
            image: Input image
            
        Returns:
            Image with background removed
        """
        try:
            # Method 1: Remove white/light backgrounds (common for product photos)
            processed_image = self.image_processor.create_alpha_channel(
                image, 
                background_color=(255, 255, 255),  # White background
                tolerance=40
            )
            
            # Method 2: Additional processing for better results
            # Convert to HSV for better color-based segmentation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for very light colors (typical backgrounds)
            lower_light = np.array([0, 0, 200])
            upper_light = np.array([180, 50, 255])
            light_mask = cv2.inRange(hsv, lower_light, upper_light)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)
            light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, kernel)
            
            # Combine with existing alpha if available
            if processed_image.shape[2] == 4:
                # Modify existing alpha channel
                alpha = processed_image[:, :, 3]
                alpha[light_mask == 255] = 0
                processed_image[:, :, 3] = alpha
            else:
                # Create new alpha channel
                alpha = np.ones(image.shape[:2], dtype=np.uint8) * 255
                alpha[light_mask == 255] = 0
                processed_image = np.dstack([image, alpha])
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Background removal failed: {e}")
            # Return original image with full alpha if background removal fails
            if image.shape[2] == 3:
                alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
                return np.concatenate([image, alpha], axis=2)
            return image
    
    def get_glasses_stats(self) -> Dict[str, any]:
        """
        Get statistics about downloaded and processed glasses
        
        Returns:
            Dictionary with glasses statistics
        """
        try:
            # Count raw glasses directories
            raw_dirs = list(self.paths.raw_glasses_dir.iterdir())
            raw_glasses_count = len([d for d in raw_dirs if d.is_dir()])
            
            # Count processed glasses directories
            processed_dirs = list(self.paths.processed_glasses_dir.iterdir())
            processed_glasses_count = len([d for d in processed_dirs if d.is_dir()])
            
            # Count total images
            total_raw_images = 0
            total_processed_images = 0
            
            for raw_dir in raw_dirs:
                if raw_dir.is_dir():
                    images = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
                        images.extend(list(raw_dir.glob(ext)))
                    total_raw_images += len(images)
            
            for processed_dir in processed_dirs:
                if processed_dir.is_dir():
                    images = list(processed_dir.glob('*.png'))
                    total_processed_images += len(images)
            
            stats = {
                'raw_glasses_directories': raw_glasses_count,
                'processed_glasses_directories': processed_glasses_count,
                'total_raw_images': total_raw_images,
                'total_processed_images': total_processed_images,
                'processing_completion_rate': processed_glasses_count / raw_glasses_count if raw_glasses_count > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting glasses stats: {e}")
            return {'error': str(e)}


# Convenience functions
def download_and_preprocess_glasses(limit: Optional[int] = None) -> Dict[str, any]:
    """
    Convenience function to download and preprocess glasses
    
    Args:
        limit: Maximum number of glasses to process
        
    Returns:
        Combined results from download and preprocessing
    """
    preprocessor = GlassesPreprocessor()
    
    # Download glasses
    download_results = preprocessor.download_glasses_from_database(limit=limit)
    
    # Preprocess downloaded glasses
    preprocessing_results = preprocessor.preprocess_downloaded_glasses(limit=limit)
    
    return {
        'download_results': download_results,
        'preprocessing_results': preprocessing_results,
        'combined_stats': preprocessor.get_glasses_stats()
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing GlassesPreprocessor...")
    
    # Initialize preprocessor
    preprocessor = GlassesPreprocessor()
    
    # Test database connection
    if preprocessor.db_manager.test_connection():
        print(" Database connection successful")
        
        # Get current stats
        stats = preprocessor.get_glasses_stats()
        print(f"Current glasses stats: {stats}")
        
        # Test downloading a few glasses (uncomment to run)
        # download_results = preprocessor.download_glasses_from_database(limit=3)
        # print(f"Download results: {download_results}")
        
        # Test preprocessing
        # preprocessing_results = preprocessor.preprocess_downloaded_glasses(limit=3)
        # print(f"Preprocessing results: {preprocessing_results}")
        
        print(" GlassesPreprocessor test completed")
    else:
        print(" Database connection failed")