"""
Training script for the Virtual Glasses Try-On alignment module.
This script demonstrates how to fine-tune the alignment module for better performance.
"""

import sys
import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cv2
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.config import db_manager
from image_processing.utils.image_utils import image_processor
from models.hybrid_model import HybridVirtualTryOnModel, GlassesAlignmentModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VirtualTryOnDataset(Dataset):
    """Dataset for training the virtual try-on alignment module."""
    
    def __init__(self, selfie_ids: List[str], glasses_ids: List[str], 
                 mode: str = 'train', augment: bool = True):
        """
        Initialize dataset.
        
        Args:
            selfie_ids: List of selfie IDs from database
            glasses_ids: List of glasses IDs from database
            mode: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
        """
        self.selfie_ids = selfie_ids
        self.glasses_ids = glasses_ids
        self.mode = mode
        self.augment = augment
        self.hybrid_model = HybridVirtualTryOnModel()
        
        # Create all possible combinations
        self.combinations = []
        for selfie_id in selfie_ids:
            for glasses_id in glasses_ids:
                self.combinations.append((selfie_id, glasses_id))
        
        logger.info(f"Created {mode} dataset with {len(self.combinations)} combinations")
    
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        try:
            selfie_id, glasses_id = self.combinations[idx]
            
            # Load selfie
            selfie_data = self._load_selfie(selfie_id)
            if selfie_data is None:
                # Return a dummy sample if loading fails
                return self._get_dummy_sample()
            
            # Load glasses
            glasses_data = self._load_glasses(glasses_id)
            if glasses_data is None:
                return self._get_dummy_sample()
            
            # Process with hybrid model to get features
            selfie_result = self.hybrid_model.process_selfie(selfie_data['image'])
            glasses_result = self.hybrid_model.process_glasses(glasses_data['image'])
            
            if not selfie_result or not glasses_result:
                return self._get_dummy_sample()
            
            # Generate target transformation (for training, we use synthetic targets)
            target_transform = self._generate_target_transform(selfie_data, glasses_data)
            target_blend_weight = self._generate_target_blend_weight()
            
            return {
                'face_features': selfie_result['face_features'],
                'glasses_features': glasses_result['glasses_features'],
                'target_transform': torch.tensor(target_transform, dtype=torch.float32),
                'target_blend_weight': torch.tensor([target_blend_weight], dtype=torch.float32),
                'selfie_id': selfie_id,
                'glasses_id': glasses_id
            }
            
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _load_selfie(self, selfie_id: str) -> Optional[Dict]:
        """Load selfie data from database."""
        try:
            image_data = db_manager.get_selfie_image_data(selfie_id)
            if image_data is None:
                return None
            
            image = image_processor.load_image(image_data)
            if image is None:
                return None
            
            return {'id': selfie_id, 'image': image}
            
        except Exception as e:
            logger.warning(f"Failed to load selfie {selfie_id}: {e}")
            return None
    
    def _load_glasses(self, glasses_id: str) -> Optional[Dict]:
        """Load glasses data from database."""
        try:
            query = f"""
            SELECT image_data FROM {db_manager.config['schema']}.processed_glasses 
            WHERE id = %(glasses_id)s;
            """
            
            result = db_manager.execute_query(query, {'glasses_id': glasses_id})
            if len(result) == 0:
                return None
            
            image_data = result.iloc[0]['image_data']
            if image_data is None:
                return None
            
            image = image_processor.load_image(image_data)
            if image is None:
                return None
            
            return {'id': glasses_id, 'image': image}
            
        except Exception as e:
            logger.warning(f"Failed to load glasses {glasses_id}: {e}")
            return None
    
    def _generate_target_transform(self, selfie_data: Dict, glasses_data: Dict) -> np.ndarray:
        """Generate synthetic target transformation for training."""
        # For training, we generate random but reasonable transformations
        # In a real scenario, you would have ground truth transformations
        
        # Random scale (0.8 to 1.2)
        scale = random.uniform(0.8, 1.2)
        
        # Random rotation (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        angle_rad = np.deg2rad(angle)
        
        # Random translation (-20 to 20 pixels)
        tx = random.uniform(-20, 20)
        ty = random.uniform(-20, 20)
        
        # Create affine transformation matrix
        cos_a = np.cos(angle_rad) * scale
        sin_a = np.sin(angle_rad) * scale
        
        transform_matrix = np.array([
            [cos_a, -sin_a, tx],
            [sin_a, cos_a, ty]
        ], dtype=np.float32)
        
        return transform_matrix.flatten()  # Return as 6-element vector
    
    def _generate_target_blend_weight(self) -> float:
        """Generate target blend weight for training."""
        # Random blend weight between 0.3 and 0.9
        return random.uniform(0.3, 0.9)
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for error cases."""
        return {
            'face_features': torch.zeros(768),
            'glasses_features': torch.zeros(768),
            'target_transform': torch.zeros(6),
            'target_blend_weight': torch.tensor([0.5]),
            'selfie_id': 'dummy',
            'glasses_id': 'dummy'
        }

class TryOnLoss(nn.Module):
    """Custom loss function for virtual try-on training."""
    
    def __init__(self, transform_weight: float = 1.0, blend_weight: float = 0.5):
        """Initialize loss function."""
        super().__init__()
        self.transform_weight = transform_weight
        self.blend_weight = blend_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss."""
        # Transformation loss
        transform_loss = self.mse_loss(
            predictions['transform_params'], 
            targets['target_transform']
        )
        
        # Blend weight loss
        blend_loss = self.mse_loss(
            predictions['blend_weight'], 
            targets['target_blend_weight']
        )
        
        # Combined loss
        total_loss = (self.transform_weight * transform_loss + 
                     self.blend_weight * blend_loss)
        
        return total_loss

class VirtualTryOnTrainer:
    """Trainer for the virtual try-on alignment module."""
    
    def __init__(self, device: str = "auto"):
        """Initialize trainer."""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GlassesAlignmentModule().to(self.device)
        self.criterion = TryOnLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_optimization(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.7
        )
        
        logger.info(f"Optimization setup: Adam(lr={learning_rate}, wd={weight_decay})")
    
    def prepare_datasets(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training datasets."""
        try:
            # Get selfie and glasses IDs
            selfie_splits = self._get_data_splits('selfies', train_ratio, val_ratio)
            glasses_splits = self._get_data_splits('glasses', train_ratio, val_ratio)
            
            # Create datasets
            train_dataset = VirtualTryOnDataset(
                selfie_splits['train'], 
                glasses_splits['train'], 
                mode='train', 
                augment=True
            )
            
            val_dataset = VirtualTryOnDataset(
                selfie_splits['val'], 
                glasses_splits['val'], 
                mode='val', 
                augment=False
            )
            
            test_dataset = VirtualTryOnDataset(
                selfie_splits['test'], 
                glasses_splits['test'], 
                mode='test', 
                augment=False
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
            
            logger.info(f"Datasets prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise
    
    def _get_data_splits(self, data_type: str, train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
        """Get data splits for selfies or glasses."""
        try:
            if data_type == 'selfies':
                query = f"""
                SELECT id FROM {db_manager.config['schema']}.selfies 
                WHERE face_detected = true AND quality_score > 0.4
                ORDER BY RANDOM();
                """
            else:  # glasses
                query = f"""
                SELECT id FROM {db_manager.config['schema']}.processed_glasses 
                WHERE has_transparency = true
                ORDER BY RANDOM();
                """
            
            data_df = db_manager.execute_query(query)
            ids = data_df['id'].tolist()
            
            # Split data
            n_total = len(ids)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            splits = {
                'train': ids[:n_train],
                'val': ids[n_train:n_train + n_val],
                'test': ids[n_train + n_val:]
            }
            
            logger.info(f"{data_type.capitalize()} splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
            return splits
            
        except Exception as e:
            logger.error(f"Failed to get {data_type} splits: {e}")
            return {'train': [], 'val': [], 'test': []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            try:
                # Move to device
                face_features = batch['face_features'].to(self.device)
                glasses_features = batch['glasses_features'].to(self.device)
                target_transform = batch['target_transform'].to(self.device)
                target_blend_weight = batch['target_blend_weight'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(face_features, glasses_features)
                
                # Compute loss
                targets = {
                    'target_transform': target_transform,
                    'target_blend_weight': target_blend_weight
                }
                
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            except Exception as e:
                logger.warning(f"Batch training failed: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Move to device
                    face_features = batch['face_features'].to(self.device)
                    glasses_features = batch['glasses_features'].to(self.device)
                    target_transform = batch['target_transform'].to(self.device)
                    target_blend_weight = batch['target_blend_weight'].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(face_features, glasses_features)
                    
                    # Compute loss
                    targets = {
                        'target_transform': target_transform,
                        'target_blend_weight': target_blend_weight
                    }
                    
                    loss = self.criterion(predictions, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Batch validation failed: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(self, epochs: int = 20, save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """Train the model."""
        try:
            logger.info(f"Starting training for {epochs} epochs")
            
            # Setup optimization
            self.setup_optimization()
            
            # Prepare datasets
            train_loader, val_loader, test_loader = self.prepare_datasets()
            
            # Training loop
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                # Train
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Update scheduler
                self.scheduler.step()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if save_path:
                        self.save_model(save_path)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Final test
            test_loss = self.validate(test_loader)
            logger.info(f"Final Test Loss: {test_loss:.4f}")
            
            training_history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'test_loss': test_loss
            }
            
            return training_history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}
    
    def save_model(self, save_path: Path):
        """Save model checkpoint."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'best_val_loss': self.best_val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Virtual Try-On Alignment Module")
    
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-path", type=str, default="models/alignment_weights.pth", 
                       help="Path to save trained model")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    
    return parser.parse_args()

def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup database connection
        if not db_manager.connect():
            logger.error("Failed to connect to database")
            return 1
        
        # Create trainer
        trainer = VirtualTryOnTrainer(device=args.device)
        
        # Start training
        logger.info("Starting Virtual Try-On alignment module training")
        training_history = trainer.train(
            epochs=args.epochs,
            save_path=Path(args.save_path)
        )
        
        if training_history:
            logger.info("Training completed successfully!")
            logger.info(f"Final train loss: {training_history['train_losses'][-1]:.4f}")
            logger.info(f"Final val loss: {training_history['val_losses'][-1]:.4f}")
            logger.info(f"Test loss: {training_history['test_loss']:.4f}")
            return 0
        else:
            logger.error("Training failed")
            return 1
            
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())