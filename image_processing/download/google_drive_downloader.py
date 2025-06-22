# image_processing/download/google_drive_downloader.py
"""
Google Drive dataset downloader for SCUT-FBP5500 selfie dataset
Downloads and processes the selfie dataset directly into PostgreSQL
"""
import requests
import zipfile
import io
from pathlib import Path
from typing import Optional, Dict, List
import logging
from tqdm import tqdm
import cv2
import numpy as np
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveDatasetDownloader:
    """Downloads and processes SCUT-FBP5500 dataset from Google Drive"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the downloader
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.paths = ProjectPaths()
        self.image_processor = ImageProcessor()
        
        # Dataset configuration
        self.google_drive_file_id = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
        self.zip_file_name = "SCUT-FBP5500_v2.1.zip"
        
        # Ensure directories exist
        self.paths.ensure_directories_exist()
        
        logger.info("GoogleDriveDatasetDownloader initialized")
    
    def download_from_google_drive(self, file_id: str, destination: Path) -> bool:
        """
        Download file from Google Drive with improved handling
        
        Args:
            file_id: Google Drive file ID
            destination: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading from Google Drive: {file_id}")
            
            # Method 1: Try direct download URL first
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            session = requests.Session()
            
            # First request
            response = session.get(download_url, stream=True)
            
            # Check if we got a confirmation page
            if "download_warning" in response.text or response.headers.get('content-type', '').startswith('text/html'):
                logger.info("Large file detected, getting confirmation token...")
                
                # Extract confirmation token from the response
                token = None
                for line in response.text.split('\n'):
                    if 'confirm=' in line and 'download' in line:
                        # Extract token using more robust method
                        import re
                        token_match = re.search(r'confirm=([a-zA-Z0-9\-_]+)', line)
                        if token_match:
                            token = token_match.group(1)
                            break
                
                if not token:
                    # Try alternative token extraction
                    import re
                    token_matches = re.findall(r'"([a-zA-Z0-9\-_]{25,})"', response.text)
                    if token_matches:
                        token = token_matches[0]  # Use first long token found
                
                if token:
                    logger.info(f"Using confirmation token: {token[:10]}...")
                    confirmed_url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                    response = session.get(confirmed_url, stream=True)
                else:
                    logger.warning("Could not extract confirmation token, trying alternative method...")
                    # Try alternative method using gdown-style approach
                    return self._download_with_alternative_method(file_id, destination)
            
            # Check response
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code}: {response.reason}")
                return False
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                logger.error("Received HTML instead of file - download may have failed")
                logger.info(f"Response content preview: {response.text[:500]}")
                return self._download_with_alternative_method(file_id, destination)
            
            # Check content length
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size < 1000000:  # Less than 1MB is suspicious for this dataset
                logger.warning(f"File size seems too small: {total_size} bytes")
                logger.info("This might not be the actual file, trying alternative method...")
                return self._download_with_alternative_method(file_id, destination)
            
            # Download the file
            logger.info(f"Downloading {total_size / (1024**3):.2f}GB file...")
            
            with open(destination, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Download without progress bar if size unknown
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            # Verify download
            final_size = destination.stat().st_size
            logger.info(f"✅ Download completed: {destination}")
            logger.info(f"   File size: {final_size / (1024**2):.1f}MB")
            
            if final_size < 1000000:  # Less than 1MB
                logger.error("Downloaded file is too small - likely not the correct file")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return self._download_with_alternative_method(file_id, destination)
    
    def _download_with_alternative_method(self, file_id: str, destination: Path) -> bool:
        """
        Alternative download method using different approach
        
        Args:
            file_id: Google Drive file ID
            destination: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Trying alternative download method...")
            
            # Try using gdown if available
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, str(destination), quiet=False)
                
                if destination.exists() and destination.stat().st_size > 1000000:
                    logger.info("✅ Alternative download successful with gdown")
                    return True
                    
            except ImportError:
                logger.info("gdown not available, trying manual method...")
            except Exception as e:
                logger.debug(f"gdown failed: {e}")
            
            # Manual alternative approach
            session = requests.Session()
            
            # Try the old-style download URL
            urls_to_try = [
                f"https://drive.google.com/uc?export=download&id={file_id}",
                f"https://docs.google.com/uc?export=download&id={file_id}",
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
            ]
            
            for url in urls_to_try:
                try:
                    logger.info(f"Trying URL: {url}")
                    response = session.get(url, stream=True, timeout=60)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'text/html' not in content_type:
                            # This looks like a binary file
                            with open(destination, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            
                            if destination.stat().st_size > 1000000:
                                logger.info("✅ Alternative download successful")
                                return True
                            
                except Exception as e:
                    logger.debug(f"URL {url} failed: {e}")
                    continue
            
            # If all methods fail, provide instructions for manual download
            logger.error("❌ All automatic download methods failed")
            self._provide_manual_download_instructions(file_id, destination)
            return False
            
        except Exception as e:
            logger.error(f"Alternative download method failed: {e}")
            return False
    
    def _provide_manual_download_instructions(self, file_id: str, destination: Path):
        """Provide instructions for manual download"""
        logger.info("\n" + "="*60)
        logger.info("📥 MANUAL DOWNLOAD REQUIRED")
        logger.info("="*60)
        logger.info("The automatic download failed. Please download manually:")
        logger.info(f"1. Go to: https://drive.google.com/file/d/{file_id}/view")
        logger.info("2. Click 'Download' button")
        logger.info(f"3. Save the file as: {destination}")
        logger.info("4. Then re-run the command")
        logger.info("\nAlternatively, you can:")
        logger.info("1. Use a smaller dataset limit: --limit 100")
        logger.info("2. Or use the demo with synthetic data for testing")
        logger.info("="*60)
    
    def extract_and_analyze_dataset(self, zip_path: Path) -> Dict:
        """
        Extract and analyze the SCUT-FBP5500 dataset
        
        Args:
            zip_path: Path to the downloaded zip file
            
        Returns:
            Dataset analysis information
        """
        try:
            logger.info("Extracting and analyzing dataset...")
            
            analysis = {
                'total_files': 0,
                'image_files': 0,
                'annotation_files': 0,
                'file_types': {},
                'image_stats': {
                    'min_width': float('inf'),
                    'max_width': 0,
                    'min_height': float('inf'),
                    'max_height': 0,
                    'avg_width': 0,
                    'avg_height': 0
                },
                'sample_images': []
            }
            
            width_sum = 0
            height_sum = 0
            image_count = 0
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                analysis['total_files'] = len(file_list)
                
                logger.info(f"Found {len(file_list)} files in dataset")
                
                # Analyze file types
                for filename in file_list:
                    ext = Path(filename).suffix.lower()
                    analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                    
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        analysis['image_files'] += 1
                    elif ext in ['.txt', '.csv', '.xlsx']:
                        analysis['annotation_files'] += 1
                
                # Analyze sample images
                image_files = [f for f in file_list if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                sample_size = min(50, len(image_files))  # Analyze first 50 images
                
                logger.info(f"Analyzing {sample_size} sample images...")
                
                for filename in tqdm(image_files[:sample_size], desc="Analyzing images"):
                    try:
                        # Read image from zip
                        with zip_ref.open(filename) as img_file:
                            img_data = img_file.read()
                            
                        # Convert to numpy array
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            h, w = img.shape[:2]
                            
                            # Update statistics
                            analysis['image_stats']['min_width'] = min(analysis['image_stats']['min_width'], w)
                            analysis['image_stats']['max_width'] = max(analysis['image_stats']['max_width'], w)
                            analysis['image_stats']['min_height'] = min(analysis['image_stats']['min_height'], h)
                            analysis['image_stats']['max_height'] = max(analysis['image_stats']['max_height'], h)
                            
                            width_sum += w
                            height_sum += h
                            image_count += 1
                            
                            # Store sample info
                            analysis['sample_images'].append({
                                'filename': filename,
                                'width': w,
                                'height': h,
                                'size_kb': len(img_data) // 1024
                            })
                    
                    except Exception as e:
                        logger.debug(f"Could not analyze {filename}: {e}")
                        continue
            
            # Calculate averages
            if image_count > 0:
                analysis['image_stats']['avg_width'] = width_sum / image_count
                analysis['image_stats']['avg_height'] = height_sum / image_count
            
            logger.info("✅ Dataset analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Dataset extraction/analysis failed: {e}")
            return {}
    
    def create_enhanced_selfies_table(self) -> bool:
        """
        Create enhanced selfies table with BLOB storage for images
        
        Returns:
            True if successful, False otherwise
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.db_manager.schema}.selfies_dataset (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL UNIQUE,
            original_path TEXT,
            image_data BYTEA,  -- Store actual image data
            image_width INTEGER,
            image_height INTEGER,
            file_size_kb INTEGER,
            beauty_score FLOAT,  -- SCUT-FBP5500 beauty annotations
            face_detected BOOLEAN DEFAULT FALSE,
            face_bbox_x INTEGER,
            face_bbox_y INTEGER,
            face_bbox_width INTEGER,
            face_bbox_height INTEGER,
            preprocessing_status VARCHAR(50) DEFAULT 'pending',
            dataset_split VARCHAR(20) DEFAULT 'train',  -- train/val/test
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create index for faster queries
        CREATE INDEX IF NOT EXISTS idx_selfies_dataset_split ON {self.db_manager.schema}.selfies_dataset(dataset_split);
        CREATE INDEX IF NOT EXISTS idx_selfies_dataset_beauty ON {self.db_manager.schema}.selfies_dataset(beauty_score);
        CREATE INDEX IF NOT EXISTS idx_selfies_dataset_face ON {self.db_manager.schema}.selfies_dataset(face_detected);
        """
        
        try:
            with self.db_manager.engine.connect() as connection:
                # Execute each statement separately
                for statement in create_table_sql.split(';'):
                    if statement.strip():
                        connection.execute(text(statement))
                connection.commit()
                
            logger.info("✅ Enhanced selfies table created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced selfies table: {e}")
            return False
    
    def process_and_store_dataset(self, zip_path: Path, limit: Optional[int] = None) -> Dict:
        """
        Process SCUT-FBP5500 dataset and store in PostgreSQL
        
        Args:
            zip_path: Path to the dataset zip file
            limit: Maximum number of images to process (None for all)
            
        Returns:
            Processing results
        """
        try:
            from sqlalchemy import text
            
            logger.info("Processing and storing dataset in PostgreSQL...")
            
            results = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'annotations_found': False
            }
            
            # Create enhanced table
            if not self.create_enhanced_selfies_table():
                return {'error': 'Failed to create database table'}
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                image_files = [f for f in file_list if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                
                if limit:
                    image_files = image_files[:limit]
                
                # Look for annotation files
                annotation_files = [f for f in file_list if 'train_test_files' in f or 'All_labels' in f or f.endswith('.txt')]
                if annotation_files:
                    results['annotations_found'] = True
                    logger.info(f"Found annotation files: {annotation_files}")
                
                logger.info(f"Processing {len(image_files)} images...")
                
                # Process images in batches
                batch_size = 100
                for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
                    batch_files = image_files[i:i+batch_size]
                    batch_results = self._process_image_batch(zip_ref, batch_files)
                    
                    results['successful'] += batch_results['successful']
                    results['failed'] += batch_results['failed']
                    results['skipped'] += batch_results['skipped']
                    results['total_processed'] += len(batch_files)
            
            logger.info(f"✅ Dataset processing completed: {results['successful']}/{results['total_processed']} successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Dataset processing failed: {e}")
            return {'error': str(e)}
    
    def _process_image_batch(self, zip_ref: zipfile.ZipFile, image_files: List[str]) -> Dict:
        """Process a batch of images"""
        batch_results = {'successful': 0, 'failed': 0, 'skipped': 0}
        
        for filename in image_files:
            try:
                # Read image from zip
                with zip_ref.open(filename) as img_file:
                    img_data = img_file.read()
                
                # Decode image
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    batch_results['failed'] += 1
                    continue
                
                h, w = img.shape[:2]
                file_size_kb = len(img_data) // 1024
                
                # Detect face
                faces = self.image_processor.detect_faces(img)
                face_detected = len(faces) > 0
                face_bbox = faces[0] if faces else (0, 0, 0, 0)
                
                # Extract beauty score if available (from filename or annotations)
                beauty_score = self._extract_beauty_score(filename)
                
                # Determine dataset split (80% train, 10% val, 10% test)
                rand_val = random.random()
                if rand_val < 0.8:
                    dataset_split = 'train'
                elif rand_val < 0.9:
                    dataset_split = 'val'
                else:
                    dataset_split = 'test'
                
                # Insert into database
                insert_sql = f"""
                INSERT INTO {self.db_manager.schema}.selfies_dataset 
                (filename, original_path, image_data, image_width, image_height, file_size_kb,
                 beauty_score, face_detected, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
                 dataset_split, preprocessing_status)
                VALUES (:filename, :original_path, :image_data, :image_width, :image_height, :file_size_kb,
                        :beauty_score, :face_detected, :face_bbox_x, :face_bbox_y, :face_bbox_width, :face_bbox_height,
                        :dataset_split, :preprocessing_status)
                ON CONFLICT (filename) DO UPDATE SET
                    image_data = EXCLUDED.image_data,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                with self.db_manager.engine.connect() as connection:
                    connection.execute(text(insert_sql), {
                        'filename': Path(filename).name,
                        'original_path': filename,
                        'image_data': img_data,
                        'image_width': w,
                        'image_height': h,
                        'file_size_kb': file_size_kb,
                        'beauty_score': beauty_score,
                        'face_detected': face_detected,
                        'face_bbox_x': face_bbox[0],
                        'face_bbox_y': face_bbox[1],
                        'face_bbox_width': face_bbox[2],
                        'face_bbox_height': face_bbox[3],
                        'dataset_split': dataset_split,
                        'preprocessing_status': 'completed'
                    })
                    connection.commit()
                
                batch_results['successful'] += 1
                
            except Exception as e:
                logger.debug(f"Failed to process {filename}: {e}")
                batch_results['failed'] += 1
        
        return batch_results
    
    def _extract_beauty_score(self, filename: str) -> Optional[float]:
        """Extract beauty score from filename or return None"""
        try:
            # SCUT-FBP5500 filename format often includes score
            # This is a placeholder - adjust based on actual dataset structure
            name = Path(filename).stem
            if '_' in name:
                parts = name.split('_')
                for part in parts:
                    try:
                        score = float(part)
                        if 0 <= score <= 5:  # SCUT-FBP5500 uses 1-5 scale
                            return score
                    except ValueError:
                        continue
        except Exception:
            pass
        
        return None
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the stored dataset"""
        try:
            from sqlalchemy import text
            
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_images,
                COUNT(CASE WHEN face_detected = true THEN 1 END) as faces_detected,
                AVG(image_width) as avg_width,
                AVG(image_height) as avg_height,
                AVG(beauty_score) as avg_beauty_score,
                COUNT(CASE WHEN dataset_split = 'train' THEN 1 END) as train_count,
                COUNT(CASE WHEN dataset_split = 'val' THEN 1 END) as val_count,
                COUNT(CASE WHEN dataset_split = 'test' THEN 1 END) as test_count
            FROM {self.db_manager.schema}.selfies_dataset
            """
            
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(text(stats_sql)).fetchone()
                
                return {
                    'total_images': result[0],
                    'faces_detected': result[1],
                    'face_detection_rate': result[1] / result[0] if result[0] > 0 else 0,
                    'avg_width': round(result[2], 1) if result[2] else 0,
                    'avg_height': round(result[3], 1) if result[3] else 0,
                    'avg_beauty_score': round(result[4], 2) if result[4] else None,
                    'train_count': result[5],
                    'val_count': result[6],
                    'test_count': result[7]
                }
        
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {'error': str(e)}
    
    def create_synthetic_dataset_fallback(self, limit: int = 100) -> Dict:
        """
        Create synthetic dataset as fallback when real dataset download fails
        
        Args:
            limit: Number of synthetic images to create
            
        Returns:
            Dataset creation results
        """
        logger.info("🎨 Creating synthetic dataset as fallback...")
        
        try:
            # Create synthetic selfie images
            results = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'synthetic': True
            }
            
            # Create enhanced table
            if not self.create_enhanced_selfies_table():
                return {'error': 'Failed to create database table'}
            
            logger.info(f"Generating {limit} synthetic selfie images...")
            
            for i in tqdm(range(limit), desc="Creating synthetic selfies"):
                try:
                    # Create a realistic synthetic face image
                    synthetic_image = self._create_synthetic_face_image(i)
                    
                    # Convert to bytes for storage
                    import cv2
                    _, img_encoded = cv2.imencode('.jpg', synthetic_image)
                    img_data = img_encoded.tobytes()
                    
                    # Generate metadata
                    filename = f"synthetic_selfie_{i:04d}.jpg"
                    beauty_score = np.random.uniform(2.0, 4.5)  # Random beauty score
                    face_bbox = (64, 32, 128, 160)  # Standard face region
                    
                    # Determine split
                    rand_val = np.random.random()
                    if rand_val < 0.8:
                        dataset_split = 'train'
                    elif rand_val < 0.9:
                        dataset_split = 'val'
                    else:
                        dataset_split = 'test'
                    
                    # Insert into database
                    insert_sql = f"""
                    INSERT INTO {self.db_manager.schema}.selfies_dataset 
                    (filename, original_path, image_data, image_width, image_height, file_size_kb,
                     beauty_score, face_detected, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
                     dataset_split, preprocessing_status)
                    VALUES (:filename, :original_path, :image_data, :image_width, :image_height, :file_size_kb,
                            :beauty_score, :face_detected, :face_bbox_x, :face_bbox_y, :face_bbox_width, :face_bbox_height,
                            :dataset_split, :preprocessing_status)
                    ON CONFLICT (filename) DO UPDATE SET
                        image_data = EXCLUDED.image_data,
                        updated_at = CURRENT_TIMESTAMP
                    """
                    
                    with self.db_manager.engine.connect() as connection:
                        connection.execute(text(insert_sql), {
                            'filename': filename,
                            'original_path': f'synthetic/{filename}',
                            'image_data': img_data,
                            'image_width': 256,
                            'image_height': 256,
                            'file_size_kb': len(img_data) // 1024,
                            'beauty_score': beauty_score,
                            'face_detected': True,
                            'face_bbox_x': face_bbox[0],
                            'face_bbox_y': face_bbox[1],
                            'face_bbox_width': face_bbox[2],
                            'face_bbox_height': face_bbox[3],
                            'dataset_split': dataset_split,
                            'preprocessing_status': 'completed'
                        })
                        connection.commit()
                    
                    results['successful'] += 1
                    results['total_processed'] += 1
                    
                except Exception as e:
                    logger.debug(f"Failed to create synthetic image {i}: {e}")
                    results['failed'] += 1
                    results['total_processed'] += 1
            
            logger.info(f"✅ Synthetic dataset created: {results['successful']}/{results['total_processed']} images")
            return results
            
        except Exception as e:
            logger.error(f"❌ Synthetic dataset creation failed: {e}")
            return {'error': str(e)}
    
    def _create_synthetic_face_image(self, seed: int) -> np.ndarray:
        """
        Create a synthetic face image for training
        
        Args:
            seed: Random seed for variation
            
        Returns:
            Synthetic face image as numpy array
        """
        np.random.seed(seed)
        
        # Create base image
        img = np.ones((256, 256, 3), dtype=np.uint8) * np.random.randint(200, 240)
        
        # Face parameters with variation
        face_center_x = 128 + np.random.randint(-10, 10)
        face_center_y = 128 + np.random.randint(-10, 10)
        face_width = 100 + np.random.randint(-20, 20)
        face_height = 130 + np.random.randint(-20, 20)
        
        # Skin color variation
        skin_base = np.random.randint(180, 220)
        skin_color = (skin_base - 30, skin_base - 10, skin_base)
        
        # Draw face oval
        cv2.ellipse(img, (face_center_x, face_center_y), (face_width//2, face_height//2), 
                   0, 0, 360, skin_color, -1)
        
        # Eyes
        eye_y = face_center_y - 20 + np.random.randint(-5, 5)
        left_eye_x = face_center_x - 25 + np.random.randint(-5, 5)
        right_eye_x = face_center_x + 25 + np.random.randint(-5, 5)
        
        # Eye whites
        cv2.ellipse(img, (left_eye_x, eye_y), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (right_eye_x, eye_y), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        
        # Pupils
        pupil_color = (20, 20, 20)
        cv2.circle(img, (left_eye_x, eye_y), 6, pupil_color, -1)
        cv2.circle(img, (right_eye_x, eye_y), 6, pupil_color, -1)
        
        # Eyebrows
        brow_color = tuple(int(c * 0.4) for c in skin_color)
        cv2.ellipse(img, (left_eye_x, eye_y - 15), (15, 5), 0, 0, 180, brow_color, -1)
        cv2.ellipse(img, (right_eye_x, eye_y - 15), (15, 5), 0, 0, 180, brow_color, -1)
        
        # Nose
        nose_color = tuple(int(c * 0.9) for c in skin_color)
        nose_points = np.array([
            [face_center_x - 6, face_center_y - 5],
            [face_center_x + 6, face_center_y - 5],
            [face_center_x + 8, face_center_y + 15],
            [face_center_x - 8, face_center_y + 15]
        ], np.int32)
        cv2.fillPoly(img, [nose_points], nose_color)
        
        # Mouth
        mouth_y = face_center_y + 35 + np.random.randint(-5, 5)
        mouth_color = (120, 80, 80)
        cv2.ellipse(img, (face_center_x, mouth_y), (18, 8), 0, 0, 360, mouth_color, -1)
        
        # Hair
        hair_colors = [(60, 40, 20), (40, 30, 20), (20, 20, 20), (100, 80, 60)]
        hair_color = hair_colors[seed % len(hair_colors)]
        cv2.ellipse(img, (face_center_x, face_center_y - 60), (face_width//2 + 20, 40), 
                   0, 0, 180, hair_color, -1)
        
        # Add some noise for realism
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img

    def download_and_process_complete_dataset(self, limit: Optional[int] = None) -> Dict:
        """
        Complete workflow: download, extract, and process SCUT-FBP5500 dataset
        
        Args:
            limit: Maximum number of images to process
            
        Returns:
            Complete workflow results
        """
        logger.info("🚀 Starting complete SCUT-FBP5500 dataset workflow...")
        
        workflow_results = {
            'download': {'status': 'pending'},
            'analysis': {'status': 'pending'},
            'processing': {'status': 'pending'},
            'final_stats': {'status': 'pending'}
        }
        
        try:
            # Step 1: Download dataset
            zip_path = self.paths.data_raw_dir / self.zip_file_name
            
            if not zip_path.exists():
                logger.info("Step 1: Downloading dataset from Google Drive...")
                download_success = self.download_from_google_drive(self.google_drive_file_id, zip_path)
                
                if download_success and zip_path.exists() and zip_path.stat().st_size > 1000000:
                    workflow_results['download'] = {'status': 'success', 'path': str(zip_path)}
                else:
                    logger.warning("❌ Google Drive download failed or file too small")
                    logger.info("🎨 Falling back to synthetic dataset generation...")
                    
                    # Use synthetic dataset as fallback
                    synthetic_results = self.create_synthetic_dataset_fallback(limit=limit)
                    
                    if 'error' not in synthetic_results:
                        workflow_results['download'] = {'status': 'synthetic_fallback', 'data': synthetic_results}
                        workflow_results['analysis'] = {'status': 'skipped', 'reason': 'using_synthetic_data'}
                        workflow_results['processing'] = {'status': 'synthetic', 'data': synthetic_results}
                        
                        # Get final statistics
                        final_stats = self.get_dataset_stats()
                        workflow_results['final_stats'] = {'status': 'success', 'data': final_stats}
                        
                        logger.info("✅ Synthetic dataset workflow completed successfully!")
                        return workflow_results
                    else:
                        workflow_results['download'] = {'status': 'failed', 'error': 'Both real and synthetic dataset creation failed'}
                        return workflow_results
            else:
                logger.info("Dataset already downloaded, skipping download step")
                workflow_results['download'] = {'status': 'skipped', 'path': str(zip_path)}
            
            # Step 2: Analyze dataset
            logger.info("Step 2: Analyzing dataset structure...")
            analysis = self.extract_and_analyze_dataset(zip_path)
            
            if analysis:
                workflow_results['analysis'] = {'status': 'success', 'data': analysis}
                logger.info(f"Analysis: {analysis['image_files']} images, {analysis['annotation_files']} annotation files")
            else:
                workflow_results['analysis'] = {'status': 'failed', 'error': 'Analysis failed'}
                return workflow_results
            
            # Step 3: Process and store in database
            logger.info("Step 3: Processing and storing in PostgreSQL...")
            processing = self.process_and_store_dataset(zip_path, limit=limit)
            
            if 'error' not in processing:
                workflow_results['processing'] = {'status': 'success', 'data': processing}
                logger.info(f"Processing: {processing['successful']}/{processing['total_processed']} images stored")
            else:
                workflow_results['processing'] = {'status': 'failed', 'error': processing['error']}
                return workflow_results
            
            # Step 4: Get final statistics
            logger.info("Step 4: Getting final dataset statistics...")
            final_stats = self.get_dataset_stats()
            
            if 'error' not in final_stats:
                workflow_results['final_stats'] = {'status': 'success', 'data': final_stats}
                logger.info(f"Final stats: {final_stats['total_images']} images with {final_stats['face_detection_rate']:.1%} face detection rate")
            else:
                workflow_results['final_stats'] = {'status': 'failed', 'error': final_stats['error']}
            
            logger.info("🎉 Complete dataset workflow finished!")
            return workflow_results
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            workflow_results['error'] = str(e)
            return workflow_results


# Convenience functions
def download_scut_fbp5500_dataset(limit: Optional[int] = None) -> Dict:
    """
    Convenience function to download and process SCUT-FBP5500 dataset
    
    Args:
        limit: Maximum number of images to process
        
    Returns:
        Workflow results
    """
    downloader = GoogleDriveDatasetDownloader()
    return downloader.download_and_process_complete_dataset(limit=limit)

# Example usage and testing
if __name__ == "__main__":
    print("Testing GoogleDriveDatasetDownloader...")
    
    # Initialize downloader
    downloader = GoogleDriveDatasetDownloader()
    
    # Test database connection
    if downloader.db_manager.test_connection():
        print("✅ Database connection successful")
        
        # Test with small sample
        print("Running workflow with 10 sample images...")
        results = downloader.download_and_process_complete_dataset(limit=10)
        
        print("Workflow Results:")
        for step, result in results.items():
            if isinstance(result, dict) and 'status' in result:
                print(f"  {step}: {result['status']}")
        
        # Get final stats
        if results.get('final_stats', {}).get('status') == 'success':
            stats = results['final_stats']['data']
            print(f"\nDataset Statistics:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Face detection rate: {stats['face_detection_rate']:.1%}")
            print(f"  Train/Val/Test split: {stats['train_count']}/{stats['val_count']}/{stats['test_count']}")
        
        print("✅ GoogleDriveDatasetDownloader test completed")
    else:
        print("❌ Database connection failed")