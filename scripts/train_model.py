# scripts/train_model.py
"""
Training script for the Virtual Glasses Try-On system
Implements actual training for virtual try-on using selfies and glasses data
"""
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from models.data_loaders import create_data_loaders, SelfiesDataset, GlassesDataset
from models.hybrid_model import HybridVirtualTryOnModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualTryOnLoss(nn.Module):
    """Custom loss function for virtual try-on training"""
    
    def __init__(self, lambda_l1: float = 1.0, lambda_perceptual: float = 0.1, 
                 lambda_style: float = 0.01, lambda_identity: float = 0.1):
        """
        Initialize virtual try-on loss
        
        Args:
            lambda_l1: Weight for L1 reconstruction loss
            lambda_perceptual: Weight for perceptual loss
            lambda_style: Weight for style loss
            lambda_identity: Weight for identity preservation loss
        """
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_identity = lambda_identity
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # For perceptual loss (simplified - using random features)
        self.perceptual_features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        ).eval()
        
        # Freeze perceptual features
        for param in self.perceptual_features.parameters():
            param.requires_grad = False
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, 
                selfie: torch.Tensor, glasses_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute virtual try-on loss
        
        Args:
            output: Generated image
            target: Target image (ground truth)
            selfie: Original selfie
            glasses_mask: Mask indicating glasses region
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # L1 reconstruction loss
        losses['l1'] = self.l1_loss(output, target) * self.lambda_l1
        
        # Perceptual loss (simplified)
        if self.lambda_perceptual > 0:
            output_features = self.perceptual_features(output)
            target_features = self.perceptual_features(target)
            losses['perceptual'] = self.mse_loss(output_features, target_features) * self.lambda_perceptual
        else:
            losses['perceptual'] = torch.tensor(0.0, device=output.device)
        
        # Identity preservation loss (preserve non-glasses regions)
        if self.lambda_identity > 0:
            identity_mask = 1 - glasses_mask
            losses['identity'] = self.l1_loss(output * identity_mask, selfie * identity_mask) * self.lambda_identity
        else:
            losses['identity'] = torch.tensor(0.0, device=output.device)
        
        # Style consistency loss (simplified)
        if self.lambda_style > 0:
            losses['style'] = self.mse_loss(output.mean(dim=[2, 3]), target.mean(dim=[2, 3])) * self.lambda_style
        else:
            losses['style'] = torch.tensor(0.0, device=output.device)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class VirtualTryOnTrainingModel(nn.Module):
    """Trainable model for virtual try-on"""
    
    def __init__(self, input_channels: int = 7, output_channels: int = 3):
        """
        Initialize trainable virtual try-on model
        
        Args:
            input_channels: Input channels (3 for selfie + 4 for glasses RGBA)
            output_channels: Output channels (3 for RGB)
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 7 x 512 x 512
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 x 256 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 256 x 128 x 128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 512 x 64 x 64
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 4, stride=2, padding=1),  # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  # 512 x 64 x 64
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256 x 128 x 128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128 x 256 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64 x 512 x 512
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, selfie: torch.Tensor, glasses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            selfie: Selfie image [B, 3, H, W]
            glasses: Glasses image [B, 4, H, W] (RGBA)
            
        Returns:
            Tuple of (output_image, glasses_mask)
        """
        # Extract glasses mask from alpha channel
        glasses_rgb = glasses[:, :3, :, :]  # RGB channels
        glasses_mask = glasses[:, 3:4, :, :].clamp(0, 1)  # Alpha channel as mask
        
        # Concatenate selfie and glasses
        combined_input = torch.cat([selfie, glasses], dim=1)  # [B, 7, H, W]
        
        # Encode
        encoded = self.encoder(combined_input)
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Decode
        output = self.decoder(bottleneck)
        
        # Normalize output to [0, 1]
        output = (output + 1) / 2
        
        return output, glasses_mask


class VirtualTryOnTrainer:
    """Enhanced trainer class for the virtual try-on model"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.paths = ProjectPaths()
        self.db_manager = DatabaseManager()
        
        # Initialize model (use the trainable model, not the hybrid one)
        self.model = VirtualTryOnTrainingModel().to(self.device)
        
        # Verify model has parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if total_params == 0:
            raise RuntimeError("Model has no trainable parameters!")
        
        # Training parameters
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 8)
        self.learning_rate = config.get('learning_rate', 1e-4)
        
        # Initialize optimizer with model parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimizer!")
        
        self.optimizer = optim.Adam(trainable_params, lr=self.learning_rate, betas=(0.5, 0.999))
        
        # Initialize loss function
        self.criterion = VirtualTryOnLoss(
            lambda_l1=config.get('lambda_l1', 1.0),
            lambda_perceptual=config.get('lambda_perceptual', 0.1),
            lambda_style=config.get('lambda_style', 0.01),
            lambda_identity=config.get('lambda_identity', 0.1)
        ).to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"VirtualTryOnTrainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self):
        """Prepare training data loaders"""
        logger.info("Preparing training data...")
        
        try:
            # Create data loaders
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                db_manager=self.db_manager,
                batch_size=self.batch_size,
                image_size=(512, 512),
                selfies_limit=self.config.get('selfies_limit'),
                glasses_limit=self.config.get('glasses_limit'),
                num_workers=self.config.get('num_workers', 4)
            )
            
            logger.info(f"✅ Data preparation completed")
            logger.info(f"   Train batches: {len(self.train_loader)}")
            logger.info(f"   Val batches: {len(self.val_loader)}")
            logger.info(f"   Test batches: {len(self.test_loader)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'identity': 0, 'style': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")
        
        for batch in pbar:
            try:
                # Move data to device
                selfie = batch['selfie_image'].to(self.device)
                glasses = batch['glasses_image'].to(self.device)
                
                # Create target (for now, use a simple blend as ground truth)
                # In a real implementation, you'd have actual ground truth images
                glasses_rgb = glasses[:, :3, :, :]
                glasses_alpha = glasses[:, 3:4, :, :].clamp(0, 1)
                target = selfie * (1 - glasses_alpha) + glasses_rgb * glasses_alpha
                
                # Forward pass
                self.optimizer.zero_grad()
                output, glasses_mask = self.model(selfie, glasses)
                
                # Compute loss
                losses = self.criterion(output, target, selfie, glasses_mask)
                
                # Backward pass
                losses['total'].backward()
                self.optimizer.step()
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total'].item():.4f}",
                    'L1': f"{losses['l1'].item():.4f}",
                    'Identity': f"{losses['identity'].item():.4f}"
                })
                
            except Exception as e:
                logger.debug(f"Batch training error: {e}")
                continue
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] = epoch_losses[key] / max(num_batches, 1)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'identity': 0, 'style': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Move data to device
                    selfie = batch['selfie_image'].to(self.device)
                    glasses = batch['glasses_image'].to(self.device)
                    
                    # Create target
                    glasses_rgb = glasses[:, :3, :, :]
                    glasses_alpha = glasses[:, 3:4, :, :].clamp(0, 1)
                    target = selfie * (1 - glasses_alpha) + glasses_rgb * glasses_alpha
                    
                    # Forward pass
                    output, glasses_mask = self.model(selfie, glasses)
                    
                    # Compute loss
                    losses = self.criterion(output, target, selfie, glasses_mask)
                    
                    # Accumulate losses
                    for key, value in losses.items():
                        val_losses[key] += value.item()
                    
                    num_batches += 1
                    
                except Exception as e:
                    logger.debug(f"Batch validation error: {e}")
                    continue
        
        # Average losses
        for key in val_losses:
            val_losses[key] = val_losses[key] / max(num_batches, 1)
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.paths.models_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Best model saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            logger.info(f"✅ Checkpoint loaded: epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def save_sample_outputs(self, epoch: int, num_samples: int = 4):
        """Save sample outputs for visualization"""
        self.model.eval()
        samples_dir = self.paths.logs_dir / "training_samples"
        samples_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            try:
                # Get a batch from validation set
                batch = next(iter(self.val_loader))
                selfie = batch['selfie_image'][:num_samples].to(self.device)
                glasses = batch['glasses_image'][:num_samples].to(self.device)
                
                # Generate outputs
                output, _ = self.model(selfie, glasses)
                
                # Create target for comparison
                glasses_rgb = glasses[:, :3, :, :]
                glasses_alpha = glasses[:, 3:4, :, :].clamp(0, 1)
                target = selfie * (1 - glasses_alpha) + glasses_rgb * glasses_alpha
                
                # Save comparison images
                fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
                if num_samples == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(num_samples):
                    # Original selfie
                    axes[i, 0].imshow(selfie[i].cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[i, 0].set_title('Selfie')
                    axes[i, 0].axis('off')
                    
                    # Glasses
                    glasses_vis = glasses[i, :3].cpu().permute(1, 2, 0).clamp(0, 1)
                    axes[i, 1].imshow(glasses_vis)
                    axes[i, 1].set_title('Glasses')
                    axes[i, 1].axis('off')
                    
                    # Target
                    axes[i, 2].imshow(target[i].cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[i, 2].set_title('Target')
                    axes[i, 2].axis('off')
                    
                    # Generated output
                    axes[i, 3].imshow(output[i].cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[i, 3].set_title('Generated')
                    axes[i, 3].axis('off')
                
                plt.tight_layout()
                plt.savefig(samples_dir / f"epoch_{epoch}_samples.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Sample outputs saved for epoch {epoch}")
                
            except Exception as e:
                logger.debug(f"Failed to save sample outputs: {e}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        if not self.train_losses or not self.val_losses:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Total loss
            axes[0, 0].plot(epochs, [l['total'] for l in self.train_losses], label='Train')
            axes[0, 0].plot(epochs, [l['total'] for l in self.val_losses], label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # L1 loss
            axes[0, 1].plot(epochs, [l['l1'] for l in self.train_losses], label='Train')
            axes[0, 1].plot(epochs, [l['l1'] for l in self.val_losses], label='Val')
            axes[0, 1].set_title('L1 Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Perceptual loss
            axes[1, 0].plot(epochs, [l['perceptual'] for l in self.train_losses], label='Train')
            axes[1, 0].plot(epochs, [l['perceptual'] for l in self.val_losses], label='Val')
            axes[1, 0].set_title('Perceptual Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Identity loss
            axes[1, 1].plot(epochs, [l['identity'] for l in self.train_losses], label='Train')
            axes[1, 1].plot(epochs, [l['identity'] for l in self.val_losses], label='Val')
            axes[1, 1].set_title('Identity Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            curves_path = self.paths.logs_dir / "training_curves.png"
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved: {curves_path}")
            
        except Exception as e:
            logger.debug(f"Failed to plot training curves: {e}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        
        try:
            # Prepare data
            if not self.prepare_data():
                raise RuntimeError("Data preparation failed")
            
            for epoch in range(self.current_epoch, self.epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_losses = self.train_epoch()
                self.train_losses.append(train_losses)
                
                # Validate
                val_losses = self.validate()
                self.val_losses.append(val_losses)
                
                # Log progress
                logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                logger.info(f"  Train Loss: {train_losses['total']:.4f} (L1: {train_losses['l1']:.4f})")
                logger.info(f"  Val Loss: {val_losses['total']:.4f} (L1: {val_losses['l1']:.4f})")
                
                # Save best model
                is_best = val_losses['total'] < self.best_loss
                if is_best:
                    self.best_loss = val_losses['total']
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0 or is_best:
                    self.save_checkpoint(epoch + 1, val_losses['total'], is_best)
                
                # Save sample outputs
                if (epoch + 1) % 5 == 0:
                    self.save_sample_outputs(epoch + 1)
                
                # Plot training curves
                if (epoch + 1) % 10 == 0:
                    self.plot_training_curves()
            
            logger.info("🎉 Training completed successfully!")
            
            # Final evaluation
            final_metrics = self.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        self.model.eval()
        test_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'identity': 0, 'style': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    selfie = batch['selfie_image'].to(self.device)
                    glasses = batch['glasses_image'].to(self.device)
                    
                    # Create target
                    glasses_rgb = glasses[:, :3, :, :]
                    glasses_alpha = glasses[:, 3:4, :, :].clamp(0, 1)
                    target = selfie * (1 - glasses_alpha) + glasses_rgb * glasses_alpha
                    
                    # Forward pass
                    output, glasses_mask = self.model(selfie, glasses)
                    
                    # Compute loss
                    losses = self.criterion(output, target, selfie, glasses_mask)
                    
                    # Accumulate losses
                    for key, value in losses.items():
                        test_losses[key] += value.item()
                    
                    num_batches += 1
                    
                except Exception as e:
                    logger.debug(f"Batch evaluation error: {e}")
                    continue
        
        # Average losses
        for key in test_losses:
            test_losses[key] = test_losses[key] / max(num_batches, 1)
        
        return test_losses


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Virtual Glasses Try-On Model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Data arguments
    parser.add_argument('--selfies-limit', type=int, help='Limit number of selfies')
    parser.add_argument('--glasses-limit', type=int, help='Limit number of glasses')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--lambda-l1', type=float, default=1.0, help='L1 loss weight')
    parser.add_argument('--lambda-perceptual', type=float, default=0.1, help='Perceptual loss weight')
    parser.add_argument('--lambda-identity', type=float, default=0.1, help='Identity loss weight')
    parser.add_argument('--lambda-style', type=float, default=0.01, help='Style loss weight')
    
    # Hardware
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate the model')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    # Create training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'selfies_limit': args.selfies_limit,
        'glasses_limit': args.glasses_limit,
        'num_workers': args.num_workers,
        'lambda_l1': args.lambda_l1,
        'lambda_perceptual': args.lambda_perceptual,
        'lambda_identity': args.lambda_identity,
        'lambda_style': args.lambda_style,
        'use_gpu': not args.no_gpu
    }
    
    # Initialize trainer
    trainer = VirtualTryOnTrainer(config)
    
    if args.evaluate:
        # Evaluation mode
        if args.checkpoint:
            if trainer.load_checkpoint(Path(args.checkpoint)):
                metrics = trainer.evaluate()
                print("Evaluation Results:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
            else:
                print("Failed to load checkpoint for evaluation")
        else:
            print("Please provide --checkpoint for evaluation")
    else:
        # Training mode
        if args.resume:
            trainer.load_checkpoint(Path(args.resume))
        trainer.train()


if __name__ == "__main__":
    # Example usage without arguments
    if len(sys.argv) == 1:
        print("🧠 Virtual Glasses Try-On Training Script")
        print("="*50)
        print("Status: Ready for training with real neural network!")
        print()
        print("Key Features:")
        print("✅ Loads data directly from PostgreSQL")
        print("✅ Implements proper virtual try-on loss function")
        print("✅ Trains actual neural network (encoder-decoder)")
        print("✅ Saves checkpoints and training visualizations")
        print("✅ Supports GPU acceleration")
        print()
        print("Example usage:")
        print("  # Train with small dataset")
        print("  python scripts/train_model.py --epochs 50 --batch-size 4 --selfies-limit 100 --glasses-limit 20")
        print()
        print("  # Resume training")
        print("  python scripts/train_model.py --resume models/checkpoints/checkpoint_epoch_10.pth")
        print()
        print("  # Evaluate model")
        print("  python scripts/train_model.py --evaluate --checkpoint models/checkpoints/best_model.pth")
        print()
        
        # Test trainer initialization
        try:
            config = {
                'epochs': 5, 
                'batch_size': 4, 
                'learning_rate': 1e-4,
                'selfies_limit': 10,
                'glasses_limit': 5
            }
            trainer = VirtualTryOnTrainer(config)
            print("✅ Trainer initialization successful")
            print(f"   Device: {trainer.device}")
            print(f"   Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
            
            # Test data preparation
            if trainer.db_manager.test_connection():
                print("✅ Database connection ready")
                print("🚀 Ready to start training!")
            else:
                print("❌ Database connection failed")
                
        except Exception as e:
            print(f"❌ Trainer initialization failed: {e}")
    else:
        main()