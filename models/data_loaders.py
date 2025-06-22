# models/data_loaders.py
"""
Data loaders for training the Virtual Glasses Try-On model
Loads data directly from PostgreSQL database
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
from sqlalchemy import text
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfiesDataset(Dataset):
    """Dataset for loading selfies from PostgreSQL"""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 split: str = 'train',
                 transform=None,
                 image_size: Tuple[int, int] = (512, 512),
                 limit: Optional[int] = None):
        """
        Initialize selfies dataset
        
        Args:
            db_manager: Database manager instance
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transforms
            image_size: Target image size
            limit: Maximum number of images to load
        """
        self.db_manager = db_manager
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.image_processor = ImageProcessor()
        
        # Load metadata from database
        self.selfies_metadata = self._load_selfies_metadata(limit)
        
        logger.info(f"SelfiesDataset initialized: {len(self.selfies_metadata)} {split} images")
    
    def _load_selfies_metadata(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load selfies metadata from database"""
        try:
            query = f"""
            SELECT id, filename, image_width, image_height, beauty_score, 
                   face_detected, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height
            FROM {self.db_manager.schema}.selfies_dataset 
            WHERE dataset_split = :split
            ORDER BY id
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            with self.db_manager.engine.connect() as connection:
                df = pd.read_sql(query, connection, params={'split': self.split})
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load selfies metadata: {e}")
            return pd.DataFrame()
    
    def __len__(self) -> int:
        return len(self.selfies_metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single selfie item"""
        try:
            row = self.selfies_metadata.iloc[idx]
            
            # Load image from database
            image = self._load_image_from_db(row['id'])
            
            if image is None:
                # Return a dummy image if loading fails
                image = np.zeros((*self.image_size, 3), dtype=np.uint8)
            
            # Resize image
            image = self.image_processor.resize_image(image, self.image_size, maintain_aspect_ratio=True)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            # Face bbox (normalized)
            face_bbox = None
            if row['face_detected'] and pd.notna(row['face_bbox_x']):
                face_bbox = torch.tensor([
                    row['face_bbox_x'] / row['image_width'],
                    row['face_bbox_y'] / row['image_height'],
                    row['face_bbox_width'] / row['image_width'],
                    row['face_bbox_height'] / row['image_height']
                ], dtype=torch.float32)
            else:
                face_bbox = torch.zeros(4, dtype=torch.float32)
            
            # Beauty score (normalized to 0-1)
            beauty_score = torch.tensor(row['beauty_score'] / 5.0 if pd.notna(row['beauty_score']) else 0.5, dtype=torch.float32)
            
            item = {
                'image': image_tensor,
                'face_bbox': face_bbox,
                'beauty_score': beauty_score,
                'face_detected': torch.tensor(row['face_detected'], dtype=torch.bool),
                'filename': row['filename'],
                'id': row['id']
            }
            
            if self.transform:
                item = self.transform(item)
            
            return item
            
        except Exception as e:
            logger.debug(f"Failed to load item {idx}: {e}")
            # Return dummy data
            return {
                'image': torch.zeros(3, *self.image_size, dtype=torch.float32),
                'face_bbox': torch.zeros(4, dtype=torch.float32),
                'beauty_score': torch.tensor(0.5, dtype=torch.float32),
                'face_detected': torch.tensor(False, dtype=torch.bool),
                'filename': 'error',
                'id': -1
            }
    
    def _load_image_from_db(self, image_id: int) -> Optional[np.ndarray]:
        """Load image data from database"""
        try:
            query = f"""
            SELECT image_data FROM {self.db_manager.schema}.selfies_dataset 
            WHERE id = :image_id
            """
            
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(text(query), {'image_id': image_id}).fetchone()
                
                if result and result[0]:
                    # Decode image from binary data
                    img_data = result[0]
                    nparr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return image
                    
            return None
            
        except Exception as e:
            logger.debug(f"Failed to load image {image_id}: {e}")
            return None


class GlassesDataset(Dataset):
    """Dataset for loading glasses from PostgreSQL"""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 transform=None,
                 image_size: Tuple[int, int] = (512, 512),
                 style_filter: Optional[List[str]] = None,
                 limit: Optional[int] = None):
        """
        Initialize glasses dataset
        
        Args:
            db_manager: Database manager instance
            transform: Optional transforms
            image_size: Target image size
            style_filter: Filter by glass styles
            limit: Maximum number of glasses to load
        """
        self.db_manager = db_manager
        self.transform = transform
        self.image_size = image_size
        self.image_processor = ImageProcessor()
        
        # Load metadata from database
        self.glasses_metadata = self._load_glasses_metadata(style_filter, limit)
        
        logger.info(f"GlassesDataset initialized: {len(self.glasses_metadata)} glasses")
    
    def _load_glasses_metadata(self, style_filter: Optional[List[str]] = None, 
                              limit: Optional[int] = None) -> pd.DataFrame:
        """Load glasses metadata from database"""
        try:
            query = f"""
            SELECT ge.id, ge.title, ge.brand, ge.style_category, ge.frame_material,
                   ge.primary_color, ge.best_front_view_url, ge.main_image_url,
                   f.main_image, f.additional_images
            FROM {self.db_manager.schema}.glasses_enhanced ge
            JOIN {self.db_manager.schema}.frames f ON ge.id = f.id
            WHERE ge.analysis_completed = true
            """
            
            params = {}
            if style_filter:
                placeholders = ', '.join([f':style_{i}' for i in range(len(style_filter))])
                query += f" AND ge.style_category IN ({placeholders})"
                for i, style in enumerate(style_filter):
                    params[f'style_{i}'] = style
            
            query += " ORDER BY ge.id"
            
            if limit:
                query += f" LIMIT {limit}"
            
            with self.db_manager.engine.connect() as connection:
                df = pd.read_sql(query, connection, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load glasses metadata: {e}")
            # Fallback to original frames table
            try:
                query = f"""
                SELECT id, title, main_image, additional_images
                FROM {self.db_manager.schema}.frames
                ORDER BY id
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                with self.db_manager.engine.connect() as connection:
                    df = pd.read_sql(query, connection)
                    
                # Add default values for missing columns
                df['brand'] = df['title'].str.split().str[0]
                df['style_category'] = 'unknown'
                df['frame_material'] = 'unknown'
                df['primary_color'] = 'unknown'
                df['best_front_view_url'] = df['main_image']
                
                return df
                
            except Exception as e2:
                logger.error(f"Fallback glasses loading failed: {e2}")
                return pd.DataFrame()
    
    def __len__(self) -> int:
        return len(self.glasses_metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single glasses item"""
        try:
            row = self.glasses_metadata.iloc[idx]
            
            # Load glasses image (prefer front view)
            image_url = row.get('best_front_view_url') or row.get('main_image')
            image = self._load_image_from_url(image_url)
            
            if image is None:
                # Return a dummy image if loading fails
                image = np.zeros((*self.image_size, 4), dtype=np.uint8)  # RGBA for glasses
            
            # Ensure RGBA format for glasses (transparency)
            if image.shape[2] == 3:
                alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255
                image = np.concatenate([image, alpha], axis=2)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            # Style category (one-hot encoded)
            style_categories = ['round', 'square', 'aviator', 'cat_eye', 'wayfarer', 
                              'oversized', 'sport', 'vintage', 'rimless', 'half_rim', 'unknown']
            style = row.get('style_category', 'unknown')
            style_idx = style_categories.index(style) if style in style_categories else -1
            style_onehot = torch.zeros(len(style_categories), dtype=torch.float32)
            if style_idx >= 0:
                style_onehot[style_idx] = 1.0
            
            item = {
                'image': image_tensor,
                'style_onehot': style_onehot,
                'style_category': style,
                'brand': row.get('brand', ''),
                'title': row.get('title', ''),
                'primary_color': row.get('primary_color', 'unknown'),
                'id': row['id']
            }
            
            if self.transform:
                item = self.transform(item)
            
            return item
            
        except Exception as e:
            logger.debug(f"Failed to load glasses item {idx}: {e}")
            # Return dummy data
            return {
                'image': torch.zeros(4, *self.image_size, dtype=torch.float32),
                'style_onehot': torch.zeros(11, dtype=torch.float32),
                'style_category': 'unknown',
                'brand': 'error',
                'title': 'error',
                'primary_color': 'unknown',
                'id': 'error'
            }
    
    def _load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load image from URL"""
        try:
            return self.image_processor.load_image_from_url(url)
        except Exception as e:
            logger.debug(f"Failed to load image from URL {url}: {e}")
            return None


class VirtualTryOnDataset(Dataset):
    """Combined dataset for virtual try-on training"""
    
    def __init__(self,
                 selfies_dataset: SelfiesDataset,
                 glasses_dataset: GlassesDataset,
                 pairs_per_epoch: int = 1000,
                 random_pairing: bool = True):
        """
        Initialize virtual try-on dataset
        
        Args:
            selfies_dataset: Selfies dataset
            glasses_dataset: Glasses dataset
            pairs_per_epoch: Number of selfie-glasses pairs per epoch
            random_pairing: Whether to use random pairing
        """
        self.selfies_dataset = selfies_dataset
        self.glasses_dataset = glasses_dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.random_pairing = random_pairing
        
        logger.info(f"VirtualTryOnDataset initialized: {pairs_per_epoch} pairs per epoch")
    
    def __len__(self) -> int:
        return self.pairs_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a selfie-glasses pair"""
        try:
            if self.random_pairing:
                # Random pairing
                selfie_idx = random.randint(0, len(self.selfies_dataset) - 1)
                glasses_idx = random.randint(0, len(self.glasses_dataset) - 1)
            else:
                # Sequential pairing with wraparound
                selfie_idx = idx % len(self.selfies_dataset)
                glasses_idx = idx % len(self.glasses_dataset)
            
            # Get selfie and glasses
            selfie = self.selfies_dataset[selfie_idx]
            glasses = self.glasses_dataset[glasses_idx]
            
            # Combine into training pair
            item = {
                # Input images
                'selfie_image': selfie['image'],
                'glasses_image': glasses['image'],
                
                # Face information
                'face_bbox': selfie['face_bbox'],
                'face_detected': selfie['face_detected'],
                
                # Style information
                'glasses_style': glasses['style_onehot'],
                'glasses_category': glasses['style_category'],
                
                # Metadata
                'selfie_id': selfie['id'],
                'glasses_id': glasses['id'],
                'beauty_score': selfie['beauty_score'],
            }
            
            return item
            
        except Exception as e:
            logger.debug(f"Failed to create pair {idx}: {e}")
            # Return dummy pair
            return {
                'selfie_image': torch.zeros(3, 512, 512, dtype=torch.float32),
                'glasses_image': torch.zeros(4, 512, 512, dtype=torch.float32),
                'face_bbox': torch.zeros(4, dtype=torch.float32),
                'face_detected': torch.tensor(False, dtype=torch.bool),
                'glasses_style': torch.zeros(11, dtype=torch.float32),
                'glasses_category': 'unknown',
                'selfie_id': -1,
                'glasses_id': 'error',
                'beauty_score': torch.tensor(0.5, dtype=torch.float32),
            }


def create_data_loaders(db_manager: DatabaseManager,
                       batch_size: int = 8,
                       image_size: Tuple[int, int] = (512, 512),
                       selfies_limit: Optional[int] = None,
                       glasses_limit: Optional[int] = None,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        db_manager: Database manager
        batch_size: Batch size for training
        image_size: Target image size
        selfies_limit: Limit on number of selfies
        glasses_limit: Limit on number of glasses
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    try:
        # Create selfies datasets
        train_selfies = SelfiesDataset(db_manager, split='train', image_size=image_size, limit=selfies_limit)
        val_selfies = SelfiesDataset(db_manager, split='val', image_size=image_size, limit=selfies_limit//5 if selfies_limit else None)
        test_selfies = SelfiesDataset(db_manager, split='test', image_size=image_size, limit=selfies_limit//5 if selfies_limit else None)
        
        # Create glasses dataset (shared across splits)
        glasses_dataset = GlassesDataset(db_manager, image_size=image_size, limit=glasses_limit)
        
        # Create virtual try-on datasets
        train_dataset = VirtualTryOnDataset(train_selfies, glasses_dataset, pairs_per_epoch=len(train_selfies) * 2)
        val_dataset = VirtualTryOnDataset(val_selfies, glasses_dataset, pairs_per_epoch=len(val_selfies), random_pairing=False)
        test_dataset = VirtualTryOnDataset(test_selfies, glasses_dataset, pairs_per_epoch=len(test_selfies), random_pairing=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"   Train: {len(train_dataset)} pairs, {len(train_loader)} batches")
        logger.info(f"   Val: {len(val_dataset)} pairs, {len(val_loader)} batches")
        logger.info(f"   Test: {len(test_dataset)} pairs, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    print("Testing Training Data Loaders...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    if db_manager.test_connection():
        print("✅ Database connection successful")
        
        try:
            # Test selfies dataset
            print("Testing SelfiesDataset...")
            selfies_dataset = SelfiesDataset(db_manager, split='train', limit=5)
            print(f"✅ SelfiesDataset: {len(selfies_dataset)} images")
            
            if len(selfies_dataset) > 0:
                sample = selfies_dataset[0]
                print(f"   Sample selfie: {sample['image'].shape}, face_detected: {sample['face_detected']}")
            
            # Test glasses dataset
            print("Testing GlassesDataset...")
            glasses_dataset = GlassesDataset(db_manager, limit=5)
            print(f"✅ GlassesDataset: {len(glasses_dataset)} glasses")
            
            if len(glasses_dataset) > 0:
                sample = glasses_dataset[0]
                print(f"   Sample glasses: {sample['image'].shape}, style: {sample['style_category']}")
            
            # Test virtual try-on dataset
            if len(selfies_dataset) > 0 and len(glasses_dataset) > 0:
                print("Testing VirtualTryOnDataset...")
                tryon_dataset = VirtualTryOnDataset(selfies_dataset, glasses_dataset, pairs_per_epoch=3)
                print(f"✅ VirtualTryOnDataset: {len(tryon_dataset)} pairs")
                
                sample = tryon_dataset[0]
                print(f"   Sample pair: selfie {sample['selfie_image'].shape}, glasses {sample['glasses_image'].shape}")
                
                # Test data loader
                print("Testing DataLoader...")
                loader = DataLoader(tryon_dataset, batch_size=2, shuffle=True)
                batch = next(iter(loader))
                print(f"✅ DataLoader batch: {batch['selfie_image'].shape}")
            
            print("✅ Training data loaders test completed")
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Database connection failed")