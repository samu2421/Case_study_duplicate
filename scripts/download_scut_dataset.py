"""
Download and setup SCUT-FBP5500 dataset for virtual glasses try-on
"""
import os
import gdown
import zipfile
from pathlib import Path
from typing import Dict, Any, List
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SCUTDatasetDownloader:
    """Downloads and manages SCUT-FBP5500 dataset"""
    
    def __init__(self):
        """Initialize the downloader"""
        self.db_manager = DatabaseManager()
        self.paths = ProjectPaths()
        self.image_processor = ImageProcessor()
        
        # Load environment variables
        load_dotenv()
        
        # Dataset configuration
        self.file_id = os.getenv("GOOGLE_DRIVE_FILE_ID")
        self.zip_filename = os.getenv("GOOGLE_DRIVE_ZIP_FILE_NAME", "SCUT-FBP5500_v2.1.zip")
        self.selected_folder = "SCUT-FBP5500_v2/Images"
        
        if not self.file_id:
            raise ValueError("GOOGLE_DRIVE_FILE_ID not found in .env file")
        
        # Ensure directories exist
        self.paths.ensure_directories_exist()
        
        logger.info("SCUTDatasetDownloader initialized")
        logger.info(f"File ID: {self.file_id}")
        logger.info(f"Zip filename: {self.zip_filename}")
    
    def download_dataset(self) -> Path:
        """
        Download SCUT-FBP5500 dataset from Google Drive
        
        Returns:
            Path to downloaded zip file
        """
        logger.info("🔽 Downloading SCUT-FBP5500 dataset from Google Drive...")
        
        # Create download directory
        download_dir = self.paths.data_dir / "downloads"
        download_dir.mkdir(exist_ok=True)
        
        zip_path = download_dir / self.zip_filename
        zip_url = f"https://drive.google.com/uc?id={self.file_id}"
        
        logger.info(f"Download URL: {zip_url}")
        logger.info(f"Saving to: {zip_path}")
        
        # Download if not exists
        if not zip_path.exists():
            try:
                logger.info("Downloading... This may take a few minutes for 5500 images.")
                gdown.download(zip_url, str(zip_path), quiet=False)
                logger.info("✅ Download completed successfully!")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise
        else:
            logger.info("📁 Zip file already exists, skipping download.")
        
        return zip_path
    
    def extract_images(self, zip_path: Path) -> List[Path]:
        """
        Extract images from SCUT-FBP5500 zip file
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            List of extracted image paths
        """
        logger.info("📂 Extracting SCUT-FBP5500 images...")
        
        extract_path = self.paths.raw_selfies_dir
        extracted_files = []
        
        # Supported image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all files in the Images folder
                image_files_in_zip = [
                    f for f in zip_ref.namelist() 
                    if f.startswith(f"{self.selected_folder}/") and 
                    f.lower().endswith(image_extensions)
                ]
                
                logger.info(f"Found {len(image_files_in_zip)} images in zip file")
                
                # Extract images with progress bar
                for file_path in tqdm(image_files_in_zip, desc="Extracting images"):
                    try:
                        zip_ref.extract(file_path, extract_path)
                        
                        # Get the extracted file path
                        extracted_file = extract_path / file_path
                        if extracted_file.exists():
                            extracted_files.append(extracted_file)
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract {file_path}: {e}")
                        continue
                
                logger.info(f"✅ Successfully extracted {len(extracted_files)} images")
                logger.info(f"📁 Images extracted to: {extract_path / self.selected_folder}")
                
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
        
        return extracted_files
    
    def organize_images(self, extracted_files: List[Path]) -> List[Path]:
        """
        Organize extracted images with better naming
        
        Args:
            extracted_files: List of extracted image paths
            
        Returns:
            List of organized image paths
        """
        logger.info("🗂️ Organizing images with standardized naming...")
        
        # Create organized directory
        organized_dir = self.paths.raw_selfies_dir / "scut_dataset"
        organized_dir.mkdir(exist_ok=True)
        
        organized_files = []
        
        for i, img_path in enumerate(tqdm(extracted_files, desc="Organizing files")):
            try:
                # Create standardized filename
                # Extract original name and category info
                original_name = img_path.name
                category = self._determine_category(original_name)
                
                # New filename: scut_category_index_originalname
                new_filename = f"scut_{category}_{i+1:04d}_{original_name}"
                new_path = organized_dir / new_filename
                
                # Copy to organized location
                if not new_path.exists():
                    import shutil
                    shutil.copy2(img_path, new_path)
                
                organized_files.append(new_path)
                
            except Exception as e:
                logger.warning(f"Failed to organize {img_path}: {e}")
                continue
        
        logger.info(f"✅ Organized {len(organized_files)} images")
        return organized_files
    
    def _determine_category(self, filename: str) -> str:
        """Determine image category from filename"""
        filename_upper = filename.upper()
        
        if filename_upper.startswith('AF'):
            return 'asian_female'
        elif filename_upper.startswith('AM'):
            return 'asian_male'
        elif filename_upper.startswith('CF'):
            return 'caucasian_female'
        elif filename_upper.startswith('CM'):
            return 'caucasian_male'
        else:
            return 'unknown'
    
    def upload_to_database(self, organized_files: List[Path]) -> Dict[str, Any]:
        """
        Upload SCUT dataset metadata to PostgreSQL
        
        Args:
            organized_files: List of organized image paths
            
        Returns:
            Upload results
        """
        logger.info(f"📊 Uploading {len(organized_files)} images to database...")
        
        results = {
            'total_files': len(organized_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'categories': {}
        }
        
        # Process in batches for better performance
        batch_size = 100
        
        for i in tqdm(range(0, len(organized_files), batch_size), desc="Uploading to database"):
            batch = organized_files[i:i+batch_size]
            
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
                        
                        # Track categories
                        category = self._determine_category(img_path.name)
                        results['categories'][category] = results['categories'].get(category, 0) + 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"DB insert failed: {img_path.name}")
                
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error processing {img_path.name}: {str(e)}")
        
        logger.info(f"✅ Database upload completed:")
        logger.info(f"   Successful: {results['successful']}")
        logger.info(f"   Failed: {results['failed']}")
        logger.info(f"   Skipped: {results['skipped']}")
        logger.info(f"   Categories: {results['categories']}")
        
        return results
    
    def setup_complete_dataset(self) -> Dict[str, Any]:
        """
        Complete pipeline: download, extract, organize, and upload SCUT dataset
        
        Returns:
            Complete setup results
        """
        logger.info("🚀 Starting complete SCUT-FBP5500 dataset setup...")
        
        try:
            # Step 1: Download
            zip_path = self.download_dataset()
            
            # Step 2: Extract
            extracted_files = self.extract_images(zip_path)
            
            # Step 3: Organize
            organized_files = self.organize_images(extracted_files)
            
            # Step 4: Upload to database
            upload_results = self.upload_to_database(organized_files)
            
            # Step 5: Cleanup (optional)
            cleanup_zip = input("Delete zip file to save space? (y/n): ").lower() == 'y'
            if cleanup_zip and zip_path.exists():
                zip_path.unlink()
                logger.info("🗑️ Cleaned up zip file")
            
            # Final results
            final_results = {
                'dataset_name': 'SCUT-FBP5500',
                'total_images_extracted': len(extracted_files),
                'total_images_organized': len(organized_files),
                'database_upload': upload_results,
                'setup_completed': True,
                'categories_found': upload_results['categories']
            }
            
            logger.info("🎉 SCUT-FBP5500 dataset setup completed successfully!")
            logger.info(f"📈 Dataset statistics:")
            logger.info(f"   Total images: {len(organized_files)}")
            logger.info(f"   In database: {upload_results['successful']}")
            logger.info(f"   Categories: {upload_results['categories']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Dataset setup failed: {e}")
            return {
                'dataset_name': 'SCUT-FBP5500',
                'setup_completed': False,
                'error': str(e)
            }


def main():
    """Main function for SCUT dataset download"""
    print("🎯 SCUT-FBP5500 Dataset Setup for Virtual Glasses Try-On")
    print("="*60)
    
    try:
        # Initialize downloader
        downloader = SCUTDatasetDownloader()
        
        # Test database connection first
        if not downloader.db_manager.test_connection():
            print("❌ Database connection failed. Check your .env configuration.")
            return
        
        print("✅ Database connection successful")
        print(f"📁 Project root: {downloader.paths.project_root}")
        print(f"🆔 File ID: {downloader.file_id}")
        
        # Confirm download
        print(f"\n📋 About to download SCUT-FBP5500 dataset:")
        print(f"   • 5,500 high-quality facial images")
        print(f"   • Multiple demographics (Asian/Caucasian, Male/Female)")
        print(f"   • Perfect for virtual glasses try-on training")
        print(f"   • Will be stored in PostgreSQL database")
        
        confirm = input("\nProceed with download? (y/n): ").lower()
        if confirm != 'y':
            print("Download cancelled.")
            return
        
        # Run complete setup
        results = downloader.setup_complete_dataset()
        
        if results['setup_completed']:
            print("\n" + "="*60)
            print("🎉 SUCCESS! SCUT-FBP5500 Dataset Ready")
            print("="*60)
            print(f"✅ Images organized: {results['total_images_organized']}")
            print(f"✅ Database records: {results['database_upload']['successful']}")
            print(f"✅ Categories found: {list(results['categories_found'].keys())}")
            
            print("\n📋 Next steps:")
            print("1. Run: python scripts/run_pipeline.py analyze")
            print("2. Run: python scripts/run_pipeline.py download-glasses --limit 20")
            print("3. Run: python scripts/run_pipeline.py train --epochs 10")
            print("4. Run: python scripts/run_pipeline.py demo")
            
        else:
            print(f"\n❌ Setup failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()