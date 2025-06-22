"""
Download selfie dataset from Google Drive and store in PostgreSQL
"""
import requests
import zipfile
import gdown
from pathlib import Path
from typing import List, Dict, Optional, Any
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

class SelfieDatasetDownloader:
    """Downloads and manages selfie datasets from Google Drive"""
    
    def __init__(self):
        """Initialize the downloader"""
        self.db_manager = DatabaseManager()
        self.paths = ProjectPaths()
        self.image_processor = ImageProcessor()
        
        # Ensure directories exist
        self.paths.ensure_directories_exist()
        
        # Common Google Drive selfie datasets (you can update these URLs)
        self.dataset_urls = {
            'celeba': 'https://drive.google.com/uc?id=YOUR_CELEBA_ID',
            'ffhq': 'https://drive.google.com/uc?id=YOUR_FFHQ_ID',
            'custom': 'https://drive.google.com/uc?id=YOUR_CUSTOM_DATASET_ID'
        }
        
        logger.info("SelfieDatasetDownloader initialized")
    
    def download_from_google_drive(self, drive_url: str, dataset_name: str = "custom") -> Path:
        """
        Download dataset from Google Drive
        
        Args:
            drive_url: Google Drive URL or file ID
            dataset_name: Name for the dataset
            
        Returns:
            Path to downloaded dataset
        """
        logger.info(f"Downloading {dataset_name} dataset from Google Drive...")
        
        # Create download directory
        download_dir = self.paths.data_dir / "downloads"
        download_dir.mkdir(exist_ok=True)
        
        # Extract file ID from URL if needed
        if "drive.google.com" in drive_url:
            if "/file/d/" in drive_url:
                file_id = drive_url.split("/file/d/")[1].split("/")[0]
            elif "id=" in drive_url:
                file_id = drive_url.split("id=")[1].split("&")[0]
            else:
                raise ValueError("Could not extract file ID from Google Drive URL")
        else:
            file_id = drive_url  # Assume it's already a file ID
        
        # Download using gdown
        output_path = download_dir / f"{dataset_name}_dataset.zip"
        
        try:
            logger.info(f"Downloading to: {output_path}")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output_path), quiet=False)
            
            if output_path.exists():
                logger.info(f"✅ Download completed: {output_path}")
                return output_path
            else:
                raise Exception("Download failed - file not found")
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Try alternative download method
            return self._download_with_requests(drive_url, output_path)
    
    def _download_with_requests(self, url: str, output_path: Path) -> Path:
        """Alternative download method using requests"""
        logger.info("Trying alternative download method...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"✅ Alternative download completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Alternative download failed: {e}")
            raise
    
    def extract_dataset(self, zip_path: Path, extract_to: Optional[Path] = None) -> Path:
        """
        Extract downloaded dataset
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            
        Returns:
            Path to extracted dataset
        """
        if extract_to is None:
            extract_to = self.paths.raw_selfies_dir
        
        logger.info(f"Extracting dataset to: {extract_to}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            logger.info("✅ Extraction completed")
            return extract_to
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    def organize_extracted_files(self, extract_path: Path) -> List[Path]:
        """
        Organize extracted files and find all image files
        
        Args:
            extract_path: Path where files were extracted
            
        Returns:
            List of image file paths
        """
        logger.info("Organizing extracted files...")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            # Search recursively for images
            files = list(extract_path.rglob(f"*{ext}"))
            files.extend(list(extract_path.rglob(f"*{ext.upper()}")))
            image_files.extend(files)
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Move images to organized structure if needed
        organized_dir = self.paths.raw_selfies_dir / "dataset_images"
        organized_dir.mkdir(exist_ok=True)
        
        organized_files = []
        for i, img_file in enumerate(tqdm(image_files, desc="Organizing files")):
            # Create organized filename
            new_filename = f"selfie_{i+1:06d}{img_file.suffix.lower()}"
            new_path = organized_dir / new_filename
            
            # Copy file to organized location
            if not new_path.exists():
                shutil.copy2(img_file, new_path)
            
            organized_files.append(new_path)
        
        logger.info(f"✅ Organized {len(organized_files)} files")
        return organized_files
    
    def upload_to_database(self, image_files: List[Path], batch_size: int = 100) -> Dict[str, Any]:
        """
        Upload selfie metadata to PostgreSQL database
        
        Args:
            image_files: List of image file paths
            batch_size: Number of files to process in each batch
            
        Returns:
            Upload results
        """
        logger.info(f"Uploading {len(image_files)} selfies to database...")
        
        results = {
            'total_files': len(image_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        # Process in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="Uploading batches"):
            batch = image_files[i:i+batch_size]
            
            for img_path in batch:
                try:
                    # Get image info
                    image_info = self.image_processor.get_image_info(img_path)
                    
                    if image_info is None:
                        results['failed'] += 1
                        results['errors'].append(f"Could not read: {img_path.name}")
                        continue
                    
                    # Check if already exists
                    existing_df = self.db_manager.get_selfies_data()
                    if not existing_df.empty and img_path.name in existing_df['filename'].values:
                        results['skipped'] += 1
                        continue
                    
                    # Insert to database
                    success = self.db_manager.insert_selfie_record(
                        filename=img_path.name,
                        file_path=str(img_path),
                        file_size_kb=image_info['file_size_kb'],
                        image_width=image_info['width'],
                        image_height=image_info['height']
                    )
                    
                    if success:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"DB insert failed: {img_path.name}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error processing {img_path.name}: {str(e)}")
        
        logger.info(f"✅ Upload completed: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def download_and_setup_dataset(self, drive_url: str, dataset_name: str = "selfies") -> Dict[str, Any]:
        """
        Complete pipeline: download, extract, organize, and upload dataset
        
        Args:
            drive_url: Google Drive URL or file ID
            dataset_name: Name for the dataset
            
        Returns:
            Complete setup results
        """
        logger.info(f"🚀 Starting complete dataset setup for: {dataset_name}")
        
        try:
            # Step 1: Download
            zip_path = self.download_from_google_drive(drive_url, dataset_name)
            
            # Step 2: Extract
            extract_path = self.extract_dataset(zip_path)
            
            # Step 3: Organize
            image_files = self.organize_extracted_files(extract_path)
            
            # Step 4: Upload to database
            upload_results = self.upload_to_database(image_files)
            
            # Step 5: Cleanup
            if zip_path.exists():
                zip_path.unlink()  # Remove zip file
                logger.info("Cleaned up download files")
            
            # Final results
            final_results = {
                'dataset_name': dataset_name,
                'total_images_found': len(image_files),
                'database_upload': upload_results,
                'setup_completed': True
            }
            
            logger.info("🎉 Dataset setup completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Dataset setup failed: {e}")
            return {
                'dataset_name': dataset_name,
                'setup_completed': False,
                'error': str(e)
            }
    
    def download_sample_datasets(self) -> Dict[str, Any]:
        """Download common sample datasets for testing"""
        logger.info("Downloading sample datasets...")
        
        # You can add your specific Google Drive URLs here
        sample_urls = {
            'celeba_sample': 'your_celeba_sample_id',
            'custom_selfies': 'your_custom_dataset_id'
        }
        
        results = {}
        
        for dataset_name, file_id in sample_urls.items():
            try:
                result = self.download_and_setup_dataset(file_id, dataset_name)
                results[dataset_name] = result
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        return results


# Convenience functions
def download_selfies_from_drive(drive_url: str, dataset_name: str = "selfies") -> Dict[str, Any]:
    """
    Convenience function to download selfies from Google Drive
    
    Args:
        drive_url: Google Drive URL or file ID
        dataset_name: Name for the dataset
        
    Returns:
        Setup results
    """
    downloader = SelfieDatasetDownloader()
    return downloader.download_and_setup_dataset(drive_url, dataset_name)

# Example usage and testing
if __name__ == "__main__":
    print("🔍 Testing SelfieDatasetDownloader...")
    
    # Initialize downloader
    downloader = SelfieDatasetDownloader()
    
    # Test database connection
    if downloader.db_manager.test_connection():
        print("✅ Database connection successful")
        
        # Get current selfie stats
        current_stats = downloader.db_manager.get_selfies_data()
        print(f"Current selfies in database: {len(current_stats)}")
        
        # Instructions for user
        print("\n📋 To download your dataset:")
        print("1. Share your Google Drive file/folder publicly")
        print("2. Copy the file ID or full URL")
        print("3. Run: python -c \"")
        print("   from image_processing.download.download_selfies import download_selfies_from_drive")
        print("   result = download_selfies_from_drive('YOUR_GOOGLE_DRIVE_URL_OR_ID', 'my_selfies')")
        print("   print(result)")
        print("   \"")
        
        print("\n📋 Alternative - use the pipeline:")
        print("   python scripts/run_pipeline.py download-selfies --drive-url YOUR_URL")
        
    else:
        print("❌ Database connection failed")
    
    print("✅ SelfieDatasetDownloader ready")