"""
Configuration management for the Virtual Glasses Try-On system.
Centralizes all configuration settings and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "152.53.12.68"
    port: int = 4000
    user: str = "student_diff"
    password: str = "diff_pass"
    database: str = "postgres"
    schema: str = "diffusion"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", cls.host),
            port=int(os.getenv("DB_PORT", cls.port)),
            user=os.getenv("DB_USER", cls.user),
            password=os.getenv("DB_PASSWORD", cls.password),
            database=os.getenv("DB_DATABASE", cls.database),
            schema=os.getenv("DB_SCHEMA", cls.schema)
        )
    
    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class ModelConfig:
    """AI model configuration settings."""
    sam_model_type: str = "vit_h"  # vit_h, vit_l, vit_b
    dino_model_name: str = "dinov2_vitb14"
    device: str = "auto"  # auto, cpu, cuda
    torch_home: str = "./models/.cache/torch"
    hf_home: str = "./models/.cache/huggingface"
    alignment_weights_path: str = "models/alignment_weights.pth"
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create config from environment variables."""
        return cls(
            sam_model_type=os.getenv("SAM_MODEL_TYPE", cls.sam_model_type),
            dino_model_name=os.getenv("DINO_MODEL_NAME", cls.dino_model_name),
            device=os.getenv("MODEL_DEVICE", cls.device),
            torch_home=os.getenv("TORCH_HOME", cls.torch_home),
            hf_home=os.getenv("HF_HOME", cls.hf_home),
            alignment_weights_path=os.getenv("ALIGNMENT_WEIGHTS_PATH", cls.alignment_weights_path)
        )

@dataclass
class ProcessingConfig:
    """Image processing configuration settings."""
    target_image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 16
    quality_threshold: float = 0.4
    transparency_threshold: float = 0.3
    max_dataset_size: Optional[int] = None
    num_workers: int = 2
    
    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        """Create config from environment variables."""
        return cls(
            target_image_size=(
                int(os.getenv("TARGET_IMAGE_WIDTH", cls.target_image_size[0])),
                int(os.getenv("TARGET_IMAGE_HEIGHT", cls.target_image_size[1]))
            ),
            batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", cls.batch_size)),
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", cls.quality_threshold)),
            transparency_threshold=float(os.getenv("TRANSPARENCY_THRESHOLD", cls.transparency_threshold)),
            max_dataset_size=int(os.getenv("MAX_DATASET_SIZE")) if os.getenv("MAX_DATASET_SIZE") else None,
            num_workers=int(os.getenv("NUM_WORKERS", cls.num_workers))
        )

@dataclass
class DatasetConfig:
    """Dataset configuration settings for YOUR Google Drive dataset."""
    google_drive_file_id: str = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"  # YOUR specific file ID
    zip_filename: str = "SCUT-FBP5500_v2.1.zip"  # YOUR specific zip file
    dataset_name: str = "SCUT-FBP5500"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    @classmethod
    def from_env(cls) -> 'DatasetConfig':
        """Create config from environment variables."""
        return cls(
            google_drive_file_id=os.getenv("GOOGLE_DRIVE_FILE_ID", cls.google_drive_file_id),
            zip_filename=os.getenv("ZIP_FILENAME", cls.zip_filename),
            dataset_name=os.getenv("DATASET_NAME", cls.dataset_name),
            train_ratio=float(os.getenv("TRAIN_RATIO", cls.train_ratio)),
            val_ratio=float(os.getenv("VAL_RATIO", cls.val_ratio)),
            test_ratio=float(os.getenv("TEST_RATIO", cls.test_ratio))
        )

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 16
    save_interval: int = 5
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'TrainingConfig':
        """Create config from environment variables."""
        return cls(
            learning_rate=float(os.getenv("LEARNING_RATE", cls.learning_rate)),
            weight_decay=float(os.getenv("WEIGHT_DECAY", cls.weight_decay)),
            epochs=int(os.getenv("EPOCHS", cls.epochs)),
            batch_size=int(os.getenv("TRAIN_BATCH_SIZE", cls.batch_size)),
            save_interval=int(os.getenv("SAVE_INTERVAL", cls.save_interval)),
            early_stopping_patience=int(os.getenv("EARLY_STOPPING_PATIENCE", cls.early_stopping_patience)),
            gradient_clip_norm=float(os.getenv("GRADIENT_CLIP_NORM", cls.gradient_clip_norm))
        )

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "./logs/application.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create config from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", cls.level),
            format=os.getenv("LOG_FORMAT", cls.format),
            file_path=os.getenv("LOG_FILE", cls.file_path),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", cls.max_file_size)),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", cls.backup_count))
        )

@dataclass
class PathConfig:
    """Project path configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    demo_dir: Path = field(default_factory=lambda: Path("demo"))
    notebooks_dir: Path = field(default_factory=lambda: Path("notebooks"))
    
    def __post_init__(self):
        """Convert relative paths to absolute paths."""
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.project_root / self.models_dir
        if not self.logs_dir.is_absolute():
            self.logs_dir = self.project_root / self.logs_dir
        if not self.demo_dir.is_absolute():
            self.demo_dir = self.project_root / self.demo_dir
        if not self.notebooks_dir.is_absolute():
            self.notebooks_dir = self.project_root / self.notebooks_dir
    
    def ensure_directories(self):
        """Create all necessary directories."""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.models_dir / "sam_checkpoints",
            self.models_dir / "dino_checkpoints",
            self.logs_dir,
            self.demo_dir / "output",
            self.notebooks_dir / "outputs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

class AppConfig:
    """Main application configuration class."""
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.database = DatabaseConfig.from_env()
        self.models = ModelConfig.from_env()
        self.processing = ProcessingConfig.from_env()
        self.dataset = DatasetConfig.from_env()
        self.training = TrainingConfig.from_env()
        self.logging = LoggingConfig.from_env()
        self.paths = PathConfig()
        
        # Ensure directories exist
        self.paths.ensure_directories()
        
        # Set environment variables for models
        os.environ["TORCH_HOME"] = str(self.models.torch_home)
        os.environ["HF_HOME"] = str(self.models.hf_home)
    
    def setup_logging(self):
        """Setup logging configuration."""
        import logging.handlers
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.logging.level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = Path(self.logging.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.logging.max_file_size,
            backupCount=self.logging.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info("Logging configuration initialized")
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate data split ratios
        total_ratio = self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            errors.append(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate quality thresholds
        if not 0.0 <= self.processing.quality_threshold <= 1.0:
            errors.append(f"Quality threshold must be between 0.0 and 1.0, got {self.processing.quality_threshold}")
        
        # Validate batch sizes
        if self.processing.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.processing.batch_size}")
        
        # Validate learning rate
        if self.training.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {self.training.learning_rate}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "database": self.database.__dict__,
            "models": self.models.__dict__,
            "processing": self.processing.__dict__,
            "dataset": self.dataset.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()}
        }
    
    def save_config(self, output_path: Path):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {output_path}")

# Global configuration instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

def setup_environment():
    """Setup the complete environment with configuration and logging."""
    # Setup logging
    config.setup_logging()
    
    # Validate configuration
    if not config.validate_config():
        raise ValueError("Configuration validation failed")
    
    # Log configuration summary
    logger.info("Environment setup completed")
    logger.info(f"Project root: {config.paths.project_root}")
    logger.info(f"Database: {config.database.host}:{config.database.port}")
    logger.info(f"Model device: {config.models.device}")
    logger.info(f"Batch size: {config.processing.batch_size}")
    
    return config

if __name__ == "__main__":
    # Test configuration
    try:
        config = setup_environment()
        print("✅ Configuration setup successful")
        
        # Save configuration for reference
        config_file = config.paths.project_root / "config" / "current_config.json"
        config_file.parent.mkdir(exist_ok=True)
        config.save_config(config_file)
        print(f"✅ Configuration saved to {config_file}")
        
    except Exception as e:
        print(f"❌ Configuration setup failed: {e}")
        exit(1)