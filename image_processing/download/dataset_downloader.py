"""
Dataset downloader for the Virtual Glasses Try-On system.
Downloads SCUT-FBP5500 dataset from Google Drive and stores in PostgreSQL.
"""

import gdown
import zipfile
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Dict, Any, List
import logging
from tqdm import tqdm
import pandas as pd
import re

from database.config import db_manager
from image_processing.utils.image_utils import image_processor

logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloads and processes YOUR SCUT-FBP5500 selfie dataset from Google Drive."""
    
    def __init__(self, google_drive_file_id: str = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"):
        """Initialize dataset downloader with YOUR Google Drive file."""
        self.google_drive_file_id = google_drive_file_id  # YOUR specific file ID
        self.zip_filename = "SCUT-FBP5500_v2.1.zip"  # YOUR specific zip file
        self.temp_dir = None
        
    def download_dataset(self, extract_path: Optional[Path] = None) -> Optional[Path]:
        """Download and extract the SCUT-FBP5500 dataset from Google Drive."""
        try:
            # Create temporary directory if extract_path not provided
            if extract_path is None:
                self.temp_dir = Path(tempfile.mkdtemp())
                extract_path = self.temp_dir
            else:
                extract_path = Path(extract_path)
                extract_path.mkdir(parents=True, exist_ok=True)
            
            zip_path = extract_path / self.zip_filename
            
            logger.info(f"Downloading dataset to {zip_path}")
            
            # Download from Google Drive
            gdown.download(
                f"https://drive.google.com/uc?id={self.google_drive_file_id}",
                str(zip_path),
                quiet=False
            )
            
            logger.info("Extracting dataset...")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Remove the zip file to save space
            zip_path.unlink()
            
            # Find the extracted folder
            dataset_folders = [d for d in extract_path.iterdir() if d.is_dir()]
            if dataset_folders:
                dataset_path = dataset_folders[0]
                logger.info(f"Dataset extracted to: {dataset_path}")
                return dataset_path
            else:
                logger.error("No dataset folder found after extraction")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return None
    
    def parse_dataset_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """Parse the SCUT-FBP5500 dataset structure and metadata."""
        try:
            dataset_info = {
                'images_path': None,
                'metadata_files': [],
                'total_images': 0,
                'image_files': []
            }
            
            # Look for common dataset structures
            possible_image_dirs = ['Images', 'images', 'SCUT-FBP5500_v2', 'Data']
            for img_dir in possible_image_dirs:
                img_path = dataset_path / img_dir
                if img_path.exists():
                    dataset_info['images_path'] = img_path
                    break
            
            if dataset_info['images_path'] is None:
                # If no standard directory found, use the dataset_path itself
                dataset_info['images_path'] = dataset_path
            
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(dataset_info['images_path'].glob(f"**/*{ext}"))
                image_files.extend(dataset_info['images_path'].glob(f"**/*{ext.upper()}"))
            
            dataset_info['image_files'] = image_files
            dataset_info['total_images'] = len(image_files)
            
            # Look for metadata files
            metadata_extensions = {'.txt', '.csv', '.xlsx', '.mat'}
            for ext in metadata_extensions:
                dataset_info['metadata_files'].extend(dataset_path.glob(f"**/*{ext}"))
            
            logger.info(f"Found {dataset_info['total_images']} images in dataset")
            logger.info(f"Found {len(dataset_info['metadata_files'])} metadata files")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Failed to parse dataset structure: {e}")
            return {}
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from SCUT-FBP5500 filename patterns."""
        metadata = {
            'age_group': None,
            'gender': None,
            'ethnicity': None,
            'quality_score': 0.5  # Default neutral score
        }
        
        try:
            # SCUT-FBP5500 naming patterns (adjust based on actual dataset structure)
            filename_lower = filename.lower()
            
            # Extract gender if encoded in filename
            if any(x in filename_lower for x in ['male', 'm_', '_m_']):
                metadata['gender'] = 'male'
            elif any(x in filename_lower for x in ['female', 'f_', '_f_']):
                metadata['gender'] = 'female'
            
            # Extract age group if encoded
            if any(x in filename_lower for x in ['young', 'child', 'teen']):
                metadata['age_group'] = 'young'
            elif any(x in filename_lower for x in ['adult', 'middle']):
                metadata['age_group'] = 'adult'
            elif any(x in filename_lower for x in ['old', 'senior']):
                metadata['age_group'] = 'senior'
            
            # Extract numeric IDs or scores if present
            numbers = re.findall(r'\d+', filename)
            if numbers:
                # Use first number as a quality indicator (normalized)
                first_num = int(numbers[0])
                metadata['quality_score'] = min(first_num / 100.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata from filename {filename}: {e}")
        
        return metadata
    
    def process_and_store_image(self, image_path: Path, batch_metadata: Dict[str, Any]) -> bool:
        """Process a single image and store it in PostgreSQL."""
        try:
            # Load image
            image = image_processor.load_image(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return False
            
            # Extract metadata from filename
            file_metadata = self.extract_metadata_from_filename(image_path.name)
            
            # Detect face in image
            face_bbox = image_processor.detect_face_region(image)
            face_detected = face_bbox is not None
            
            # Calculate quality score
            quality_score = image_processor.calculate_image_quality_score(image)
            
            # Resize image for storage
            resized_image = image_processor.resize_image(image, (512, 512))
            
            # Convert to bytes for database storage
            image_bytes = image_processor.image_to_bytes(resized_image)
            
            # Prepare metadata for database
            db_metadata = {
                'file_path': str(image_path),
                'width': resized_image.shape[1],
                'height': resized_image.shape[0],
                'face_detected': face_detected,
                'age_group': file_metadata.get('age_group'),
                'gender': file_metadata.get('gender'),
                'quality_score': max(quality_score, file_metadata.get('quality_score', 0.0))
            }
            
            # Insert into database
            selfie_id = db_manager.insert_selfie(
                filename=image_path.name,
                image_data=image_bytes,
                metadata=db_metadata
            )
            
            logger.debug(f"Stored image {image_path.name} with ID: {selfie_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return False
    
    def store_dataset_in_database(self, dataset_path: Path, batch_size: int = 50) -> Dict[str, int]:
        """Store YOUR dataset in PostgreSQL database (no local storage needed)."""
        try:
            # Ensure database connection
            if not db_manager.engine:
                db_manager.connect()
            
            # Create selfies table if it doesn't exist
            db_manager.create_selfies_table()
            
            # Parse YOUR dataset structure
            dataset_info = self.parse_dataset_structure(dataset_path)
            image_files = dataset_info.get('image_files', [])
            
            if not image_files:
                logger.error("No image files found in YOUR dataset")
                return {'total': 0, 'success': 0, 'failed': 0}
            
            # Process images in batches and store directly in PostgreSQL
            results = {'total': len(image_files), 'success': 0, 'failed': 0}
            
            logger.info(f"Processing {len(image_files)} images from YOUR Google Drive dataset...")
            
            for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
                batch_files = image_files[i:i + batch_size]
                
                for image_path in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}", leave=False):
                    if self.process_and_store_image(image_path, dataset_info):
                        results['success'] += 1
                    else:
                        results['failed'] += 1
            
            logger.info(f"YOUR dataset storage completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to store dataset in database: {e}")
            return {'total': 0, 'success': 0, 'failed': 0}
    
    def download_and_store_dataset(self, extract_path: Optional[Path] = None) -> Dict[str, int]:
        """Complete pipeline: download YOUR Google Drive dataset and store in PostgreSQL."""
        try:
            logger.info("Starting YOUR Google Drive dataset download and PostgreSQL storage pipeline...")
            logger.info(f"Using YOUR Google Drive file ID: {self.google_drive_file_id}")
            logger.info(f"Using YOUR zip file: {self.zip_filename}")
            
            # Download and extract YOUR dataset
            dataset_path = self.download_dataset(extract_path)
            if dataset_path is None:
                logger.error("Failed to download YOUR dataset")
                return {'total': 0, 'success': 0, 'failed': 0}
            
            # Store YOUR dataset in PostgreSQL (no local PC storage)
            results = self.store_dataset_in_database(dataset_path)
            
            # Cleanup temporary files (we only keep data in PostgreSQL)
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary files - data is now in PostgreSQL only")
            
            return results
            
        except Exception as e:
            logger.error(f"YOUR dataset pipeline failed: {e}")
            return {'total': 0, 'success': 0, 'failed': 0}
    
    def verify_dataset_in_database(self) -> Dict[str, Any]:
        """Verify that the dataset was stored correctly in the database."""
        try:
            if not db_manager.engine:
                db_manager.connect()
            
            # Get dataset statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_images,
                COUNT(*) FILTER (WHERE face_detected = true) as faces_detected,
                AVG(quality_score) as avg_quality,
                COUNT(DISTINCT gender) as unique_genders,
                COUNT(DISTINCT age_group) as unique_age_groups
            FROM {db_manager.config['schema']}.selfies;
            """
            
            stats = db_manager.execute_query(stats_query)
            
            # Get sample records
            sample_query = f"""
            SELECT filename, face_detected, gender, age_group, quality_score
            FROM {db_manager.config['schema']}.selfies
            ORDER BY RANDOM()
            LIMIT 5;
            """
            
            samples = db_manager.execute_query(sample_query)
            
            verification_results = {
                'statistics': stats.to_dict('records')[0] if len(stats) > 0 else {},
                'sample_records': samples.to_dict('records') if len(samples) > 0 else [],
                'verification_passed': stats.iloc[0]['total_images'] > 0 if len(stats) > 0 else False
            }
            
            logger.info(f"Dataset verification: {verification_results['statistics']}")
            return verification_results
            
        except Exception as e:
            logger.error(f"Dataset verification failed: {e}")
            return {'verification_passed': False}

# Global dataset downloader instance using YOUR Google Drive file
dataset_downloader = DatasetDownloader("1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf")